import torch
import numpy as np
import matplotlib.pyplot as plt
from gsplat import rasterization
from gs_loader.kernel_loader import general_gaussian_loader, Kernel
from gs_loader.feature_mapper import feature_lift_mapper
from typing import List


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized RGB [0,1] to 0th spherical harmonic coefficient.
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def normalize_tensor(x: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalize tensor to [0, 1] along specified dims.
    """
    x = x.detach()
    mn = x.min(dim=dim, keepdim=True).values if dim is not None else x.min()
    mx = x.max(dim=dim, keepdim=True).values if dim is not None else x.max()
    return (x - mn) / (mx - mn + eps)


class Renderer:
    """
    High-level renderer for Gaussian splats with multiple modes:
      - RGB, Attention, Segmentation, Feature, Feature_PCA, Weight_Filtered, Mask
    """

    def __init__(
        self,
        gaussian_loader: general_gaussian_loader,
        feature_mapper: feature_lift_mapper,
        colormap_name: str = "turbo"
    ) -> None:
        kernels: Kernel = gaussian_loader.load()
        self._init_geometry(kernels)
        self.feature_mapper = feature_mapper
        self.cmap = plt.get_cmap(colormap_name)

        # Precompute feature PCA colors
        self.feature_pca = self._compute_feature_pca(kernels.feature)
        self.segmentation_mask = None
        self.attention_color = None
        self.attention_score_forall = None
        self.global_scale_value = 1

        print("Renderer setup complete.")

    def _init_geometry(self, kernels: Kernel) -> None:
        geom = kernels.geometry
        # ensure float32 and no grad
        self.colors = kernels.color.detach().to(torch.float32)
        self.means = geom['means'].detach().to(torch.float32)
        self.quats = geom['quats'].detach().to(torch.float32)
        self.scales = torch.exp(geom['scales'].detach().to(torch.float32))
        self.opacities = torch.sigmoid(geom['opacities'].detach().to(torch.float32)).squeeze(-1)
        self.feature_id = kernels.feature_id
        self.feature = kernels.feature.detach().to(torch.float32)
        if kernels.attention_weight != None:
            self.attention_weight = kernels.attention_weight.detach().to(torch.float32)
        else:
            self.attention_weight = None

    def _compute_feature_pca(self, feature: torch.Tensor) -> torch.Tensor:
        feat = feature.detach().to(torch.float32)
        self.feature = self.feature_mapper.pre_project_mapping(feat)
        mean = self.feature.mean(dim=0, keepdim=True)
        Xc = self.feature - mean
        cov = (Xc.t() @ Xc) / (Xc.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)[:3]
        components = eigvecs[:, idx]
        coords = Xc @ components
        normed = normalize_tensor(coords, dim=0)
        return rgb_to_sh(normed)

    def attention_score(
        self,
        texts: List[str],
        filtering: bool = True,
        sh: bool = True,
        mask: bool = False
    ) -> torch.Tensor:
        """
        Convert Gaussian embedding to attention scores related to multiple texts.
        Args:
            texts: list of strings to calculate attention scores for.
            filtering: usually True, clamps below-mean values.
            sh: if True, convert RGB heatmap to spherical-harmonic color.
            mask: if True, only update segmentation_mask and return None.
        Returns:
            attention color tensor if mask=False, else None.
        """
        with torch.no_grad():
            # 1) encode texts
            txt_feat = self.feature_mapper.text_mapping(texts).detach().to(torch.float32)
            # compute raw scores [n_features, n_texts]
            scores = torch.einsum("nc,mc->nm", self.feature, txt_feat)
            self.attention_score_forall = scores

            # segmentation: last text is positive class
            if scores.shape[1] > 1:
                max_idx = scores.argmax(dim=-1)
                self.segmentation_mask = (max_idx == (scores.shape[1] - 1))

            if mask:
                return

            # select last text's scores
            att = scores[:, -1]
            # 3) filter
            if filtering:
                att = torch.clamp(att, min=att.mean().item())
            # 4) normalize
            att_min, att_max = att.min(), att.max()
            if att_min == att_max:
                att_norm = torch.zeros_like(att)
            else:
                att_norm = (att - att_min) / (att_max - att_min)

            # 5) heatmap
            rgba = self.cmap(att_norm.cpu().numpy())
            rgb = torch.tensor(rgba[..., :3], dtype=torch.float32, device=att.device)

            # 6) assign
            if sh:
                self.attention_color = rgb_to_sh(rgb)
            else:
                self.attention_color = rgb
            return self.attention_color

    def segmentation(
        self,
        positive_words: str,
        background_words: str,
        mask: bool = True
    ) -> torch.Tensor:
        """
        Run segmentation: compute mask for positive vs background.
        Args:
            positive_words: target class description
            background_words: comma-separated negatives
            mask: always True here (populates self.segmentation_mask)
        Returns:
            segmentation mask tensor
        """
        # split background by commas
        bg_list = [w.strip() for w in background_words.split(',') if w.strip()]
        # compose queries: background(s) first, then positive
        queries = bg_list + [positive_words]
        self.attention_score(queries, filtering=False, sh=False, mask=mask)
        return self.segmentation_mask

    def weight_filtering(self, ratio: float = 0.5) -> None:
        if self.attention_weight == None:
            return
        n = len(self.attention_weight)
        k = int(n * min(max(ratio, 0.0), 1.0))
        _, idx = torch.topk(self.attention_weight, k)
        self.weighted_indices = idx

    def mask_heatmap(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().to(torch.float32)
        last = tensor[..., -1]
        last_n = normalize_tensor(last)
        argmax = tensor.argmax(dim=-1)
        mask = (argmax == (tensor.shape[-1] - 1)).float()
        heat = (last_n * mask).cpu().numpy()
        rgba = self.cmap(heat)
        return torch.tensor(rgba[..., :3], dtype=torch.float32)

    def render(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        mode: str,
        H: int,
        W: int
    ) -> torch.Tensor:
        view = torch.tensor(w2c, dtype=torch.float32, device=self.means.device, requires_grad=False).unsqueeze(0)
        Ks   = torch.tensor(K,   dtype=torch.float32, device=self.means.device, requires_grad=False).unsqueeze(0)

        colors, means, quats, scales, opac, sh_deg = self._mode_config(mode)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.means.device)
        means  = torch.as_tensor(means,  dtype=torch.float32, device=self.means.device)
        quats  = torch.as_tensor(quats,  dtype=torch.float32, device=self.means.device)
        scales = torch.as_tensor(scales, dtype=torch.float32, device=self.means.device)*self.global_scale_value
        opac   = torch.as_tensor(opac,   dtype=torch.float32, device=self.means.device)

        rendered, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opac,
            colors=colors,
            viewmats=view,
            Ks=Ks,
            width=W,
            height=H,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode='RGB',
            sh_degree=sh_deg,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="antialiased",
        )
        img = rendered.squeeze(0)
        if mode == 'Mask':
            return self.mask_heatmap(img)
        if mode == 'Metrics_Mask':
            return img, alphas
        return img

    def _mode_config(self, mode: str):
        if mode == 'RGB':
            return self.colors, self.means, self.quats, self.scales, self.opacities, 2
        if mode == 'Attention':
            return self.attention_color.unsqueeze(1), *self._select_feature_subset(), 0
        if mode == 'Segmentation':
            if self.segmentation_mask is None:
                raise RuntimeError("Segmentation mask not computed. Call `segmentation(...)` first.")
            return self._filter_by_mask(self.segmentation_mask, 2)
        if mode == 'Feature_PCA':
            return self.feature_pca.unsqueeze(1), *self._select_feature_subset(), 0
        if mode == 'Feature':
            return self.feature, self.means, self.quats, self.scales, self.opacities, None
        if mode == 'Weight_Filtered':
            idx = self.weighted_indices
            return self.colors[idx], *self._select_feature_subset(idx), 2
        if mode.endswith('Mask'):
            return self.attention_score_forall, *self._select_feature_subset(), None
        raise ValueError(f"Unknown render mode: {mode}")


    def set_gaussian_scale(self, s):
        self.global_scale_value = s

    def _select_feature_subset(self, idx = None, colors = False):
        if idx == None:
            idx = self.feature_id if self.feature_id is not None else slice(None)
        if colors: 
            return (
                self.colors[idx],
                self.means[idx],
                self.quats[idx],
                self.scales[idx],
                self.opacities[idx]
            )
        else:
            return (
                self.means[idx],
                self.quats[idx],
                self.scales[idx],
                self.opacities[idx]
            )

    def _filter_by_mask(self, mask: torch.Tensor, sh_degree: int):
        if self.feature_id != None:
            colors, means, quats, scales, opacities = self._select_feature_subset(colors=True)

        idx = mask.nonzero(as_tuple=True)[0]
        return (
            colors[idx],
            means[idx],
            quats[idx],
            scales[idx],
            opacities[idx],
            sh_degree
        )
