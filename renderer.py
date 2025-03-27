from gsplat import rasterization
from pathlib import Path
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from featup.featurizers.maskclip.clip import tokenize # maskclip tokenizer
import torch


import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class renderer:
    def __init__(self, gaussian_ckpt: Path, feature_path: Path, text_model: str = 'maskclip')-> None:
        self.cmap = plt.get_cmap("turbo")
        
        self.model_load(gaussian_ckpt)
        self.feature: torch.Tensor = torch.load(feature_path).cuda()
        if len(self.feature) != len(self.opacities):
            print("frequency filtered feature")
            self.filter_index: torch.Tensor = torch.load(str(feature_path).split('.')[0]+'_id.pt').detach().cuda()
            self.filter_index = torch.tensor(self.filter_index, dtype=torch.long)

            print(self.filter_index.shape)
        else: 
            self.filter_index = None

        self.feature = F.normalize(self.feature.float(), dim=1)


        self.text_feature = None




        if text_model == 'maskclip':
            model = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).cuda()
            self.tokenizer = tokenize
            self.text_model = model
        else: 
            print(" currently, we only support maskClip model from featUp")
            raise NotImplementedError
        
        print("set up renderer accomplsihed")

    
    def model_load(self, model_location: str):
        '''
            used to load PLY and ckpt
        '''

        if model_location.endswith('ply'):
            plydata = PlyData.read(model_location)
            max_sh_degree = 3
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])), axis=1)
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self.means = torch.tensor(xyz, dtype=torch.float, device="cuda")
            self.quats = F.normalize(torch.tensor(rots, dtype=torch.float, device="cuda"), p=2, dim=-1)
            self.scales = torch.tensor(scales, dtype=torch.float, device="cuda")
            self.opacities = torch.tensor(opacities, dtype=torch.float, device="cuda")
            sh0 = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            shN = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            self.color = torch.cat((sh0, shN), dim=1)
            torch.cuda.empty_cache()

        elif model_location.endswith('ckpt'):
            model_config = SplatfactoModelConfig()
            model: SplatfactoModel = model_config.setup(scene_box = None, num_train_data = 1)
            state_dict: dict = torch.load(model_location)['pipeline']
            state_dict = {".".join(key.split('.')[1:]): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)

            self.means: torch.Tensor = model.means.cuda()
            self.quats: torch.Tensor = model.quats.cuda()
            self.opacities: torch.Tensor = model.opacities.cuda()
            self.scales: torch.Tensor = model.scales.cuda()
            self.color: torch.Tensor = torch.cat((model.features_dc[:, None, :], model.features_rest), dim=1).cuda()
    

    def attention_score(self, text: str, filtering: bool = True, sh: bool = True)-> torch.Tensor:
        """
            Convert Gaussian embedding to attention score related to text
            Args: 
                text: str that you want to related 
                filtering: usually set true, to make the score visualized
            Returns: RGB color for visualization or return SH level for visualization
        """ 
        self.text_feature:torch.Tensor = self.text_model.model.model.encode_text(self.tokenizer(text).to(self.means.device)).float().squeeze()
        self.text_feature = F.normalize(self.text_feature, dim=0)




        attention = torch.einsum("nc,c->n", self.feature, self.text_feature)

        if filtering:
            attention = self.filtering(attention_score=attention)
        
        att_min, att_max = attention.min(), attention.max()
        if att_min == att_max:
            # If they are the same, attention is constant -> make normalized zero
            attention_norm = torch.zeros_like(attention)
        else:
            attention_norm = (attention - att_min) / (att_max - att_min)
            
        # 6) Convert to numpy and get colormap RGBA, shape [N, 4]
        hr_heatmap_rgba = self.cmap(attention_norm.detach().cpu().numpy())

        # 7) Return only the first three channels (RGB), shape [N, 3]
        hr_heatmap_rgb = hr_heatmap_rgba[..., :3]
        if sh:
            self.attention_color = torch.tensor(RGB2SH(hr_heatmap_rgb), dtype=torch.float32).cuda()
        else:
            self.attention_color = torch.tensor(hr_heatmap_rgb, dtype=torch.float32).cuda()
        print("attention score calculation accomplished")

        print(self.attention_color.shape)


    def filtering(self, attention_score: torch.Tensor)->torch.Tensor:
        """
        Clips (clamps) all values in x to lie within [mean - 2*std, mean + 2*std].
        """
        mean = attention_score.mean()
        lower_bound = mean 

        # Clamp the values to [lower_bound, upper_bound]
        return attention_score.clamp(min=lower_bound.item())
    

    def render(self, w2c: np.ndarray, k: np.ndarray, mode: str, H: int,W: int)-> torch.Tensor:
        """
            input camera location, should give out the rendered result depends on mode
        """
        w2c = torch.tensor(w2c).cuda().unsqueeze(0).to(torch.float32)
        k = torch.tensor(k).cuda().unsqueeze(0).to(torch.float32)

        if mode == 'RGB':
            colors = self.color
            sh_degree = 2
            means = self.means
            quats = self.quats
            scales = torch.exp(self.scales)
            opacities = torch.sigmoid(self.opacities).squeeze(-1)
        elif mode == 'Attention': 
            colors = self.attention_color.unsqueeze(1)
            sh_degree = 0
            if self.filter_index != None: 
                means = self.means[self.filter_index]
                quats = self.quats[self.filter_index]
                scales = torch.exp(self.scales[self.filter_index])
                opacities = torch.sigmoid(self.opacities[self.filter_index]).squeeze(-1)
            else: 
                means = self.means
                quats = self.quats
                scales = torch.exp(self.scales)
                opacities = torch.sigmoid(self.opacities).squeeze(-1)
        else:
            raise NotImplementedError
        
        render, _, _ = rasterization(
            means=means,
            quats=quats,  # rasterization does normalization internally
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=w2c,  # [1, 4, 4]
            Ks=k,  # [1, 3, 3]
            width=W,
            height=H,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode='RGB',
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="antialiased",  
        )

        return render.squeeze()
