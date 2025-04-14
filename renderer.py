from turtle import color
from typing import List
from gsplat import rasterization
from pathlib import Path
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from featup.featurizers.maskclip.clip import tokenize # maskclip tokenizer
import torch


import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from sklearn.decomposition import PCA

from kernel_loader import general_gaussian_loader
from feature_mapper import feature_lift_mapper
from kernel_loader import Kernel

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class renderer:
    def __init__(self, gaussian_loader: general_gaussian_loader, feature_mapper:feature_lift_mapper)-> None:
        self.cmap = plt.get_cmap("turbo")
        
        kernels: Kernel = gaussian_loader.load()
        self.colors = kernels.color
        self.means = kernels.geometry['means']
        self.quats = kernels.geometry['quats']
        self.scales = torch.exp(kernels.geometry['scales'])
        self.opacities = torch.sigmoid(kernels.geometry['opacities']).squeeze(-1)
        self.feature_id = kernels.feature_id
        self.feature = kernels.feature
        self.feature_mapper = feature_mapper

        self.feature = self.feature_mapper.pre_project_mapping(self.feature)


        pca = PCA(n_components=3)
        self.feature_pca =  torch.tensor(pca.fit_transform(self.feature.cpu().numpy())).cuda()
        # Min-Max normalization for each channel (component)
        # Apply min-max normalization per channel
        min_vals = self.feature_pca.min(dim=(0), keepdim=True).values  # Min along H, W (spatial dimensions)
        max_vals = self.feature_pca.max(dim=(0), keepdim=True).values  # Max along H, W

        # Normalize each channel to [0, 1]
        self.feature_pca = RGB2SH((self.feature_pca - min_vals) / (max_vals - min_vals))

        self.segmentation_mask = None
        
        print("set up renderer accomplsihed")


    def attention_score(self, texts: list, filtering: bool = True, sh: bool = True) -> torch.Tensor:
        """
        Convert Gaussian embedding to attention scores related to multiple texts.
        Args: 
            texts: list of strings to calculate attention scores for.
            filtering: usually set true, to make the score visualized.
        Returns: 
            Attention scores in RGB for visualization or SH level for visualization.
        """

        # 1) Encode multiple texts
        text_features = self.feature_mapper.text_mapping(texts)

        # 2) Calculate attention scores between each feature and the text features
        attention = torch.einsum("nc,mc->nm", self.feature, text_features)  # Shape: [n, m], n texts, m features

        if attention.shape[1] > 1: # input multiple words
            max_attention_scores = attention.argmax(dim=-1)  # Shape: [n], find the index of max score for each text 
            # The last one is the segmentation target
            
            # If the argmax is the last one
            self.segmentation_mask = max_attention_scores == (attention.shape[1]-1)

        # 3) Apply filtering if necessary
        attention = attention[:, -1]
        if filtering:
            attention = self.filtering(attention_score=attention)

        # 4) Scale the attention scores
        att_min, att_max = attention.min(), attention.max()
        if att_min == att_max:
            attention_norm = torch.zeros_like(attention)
        else:
            attention_norm = (attention - att_min) / (att_max - att_min)
            #attention_norm = torch.sigmoid(2 * (attention_norm - 0.5)) #  We will see if it is better or not

        # 5) Generate heatmap for visualization
        hr_heatmap_rgba = self.cmap(attention_norm.detach().cpu().numpy())  # Shape: [n, m, 4]

        # 6) Only keep RGB channels (first 3 channels)
        hr_heatmap_rgb = hr_heatmap_rgba[..., :3]  # Shape: [n, m, 3]


        # 8) Assign to self.attention_color based on SH flag
        if sh:
            self.attention_color = torch.tensor(RGB2SH(hr_heatmap_rgb), dtype=torch.float32).cuda()
        else:
            self.attention_color = torch.tensor(hr_heatmap_rgb, dtype=torch.float32).cuda()

        # Return the segmentation mask and attention scores
        return self.attention_color

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
            sh_degree = 2
            colors = self.colors
            means = self.means
            quats = self.quats
            scales = self.scales
            opacities = self.opacities
        elif mode == 'Attention': 
            sh_degree = 0
            colors = self.attention_color.unsqueeze(1)
            if self.feature_id != None: 
                means = self.means[self.feature_id]
                quats = self.quats[self.feature_id]
                scales = self.scales[self.feature_id]
                opacities = self.opacities[self.feature_id]
            else: 
                means = self.means
                quats = self.quats
                scales = self.scales
                opacities = self.opacities
        elif mode == 'Segmentation':
            sh_degree = 2
            colors = self.colors[self.segmentation_mask]
            means = self.means[self.segmentation_mask]
            quats = self.quats[self.segmentation_mask]
            scales = self.scales[self.segmentation_mask]
            opacities = self.opacities[self.segmentation_mask]
        elif mode == "Feature_PCA":
            sh_degree = 0
            colors = self.feature_pca.unsqueeze(1)
            if self.feature_id != None: 
                means = self.means[self.feature_id]
                quats = self.quats[self.feature_id]
                scales = self.scales[self.feature_id]
                opacities = self.opacities[self.feature_id]
            else: 
                means = self.means
                quats = self.quats
                scales = self.scales
                opacities = self.opacities
        elif mode == "Feature": # rendering feature to 2D
            sh_degree = None
            colors = self.feature
            means = self.means
            quats = self.quats
            scales = self.scales
            opacities = self.opacities
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

    def segmentation(self, positive_words: str, background_words: str):
        words = background_words.split(',')
        words.append(positive_words)

        print(words)
        self.attention_score(words)