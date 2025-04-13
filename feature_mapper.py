from distutils.command.config import config
import torch
from plyfile import PlyData
import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
import os
import torch.nn.functional as F

from featup.featurizers.maskclip.clip import tokenize  # Maskclip tokenizer



"""
We find out that each feature splatting method has this transfering method, either 
do elemntary wise transfering. For example, in feature splat, it has a 1d convolution laryer
convert c' channel feature to c channel, and project. 

Or after projection, decode H*W*C' to H*W*C. 

Another function that is frequently used it so called text_mapping, convert query text to something

We name it as:
1. pre-project-mapping 
2. post-project-mapping
3. text_mapping
"""


@dataclass
class base_feature_mapper_config:
    text_tokenizer: Optional[str] = None # Path or identifier for tokenizer
    text_encoder: Optional[str] = None  # Path to text encoder model
    gaussian_feature_decoder: Optional[str] = None  # Path to pre-project-mapping model
    feature_map_decoder: Optional[str] = None  # Path to post-project-mapping model


class base_feature_mapper(ABC):
    config: base_feature_mapper_config

    def __init__(self, config: base_feature_mapper_config) -> None:
        super().__init__()
        self.config = config
        self._setup()

    def _setup(self) -> None:
        """
        Setup function to load tokenizer and optionally 3 models:
        - text_encoder
        - gaussian_feature_decoder
        - feature_map_decoder

        If any model path is None, the corresponding module will not be loaded.
        """
        self.tokenizer = self._load_tokenizer(self.config.text_tokenizer)
        self.text_encoder = self._load_model(self.config.text_encoder)
        self.gaussian_feature_decoder = self._load_model(self.config.gaussian_feature_decoder)
        self.feature_map_decoder = self._load_model(self.config.feature_map_decoder)

    def _load_tokenizer(self, tokenizer_id: str):
        # Placeholder: replace with actual tokenizer loading
        print(f"Loading tokenizer: {tokenizer_id}")
        return tokenizer_id  # Replace with actual tokenizer instance

    def _load_model(self, path: Optional[str]) -> Optional[torch.nn.Module]:
        if path is None:
            return None
        if os.path.exists(path):
            # Replace with your actual model loading logic
            print(f"Loading model from: {path}")
            return torch.load(path).cuda()
        else:
            raise FileNotFoundError(f"Model path does not exist: {path}")
    
    @abstractmethod
    def _text_mapping(self, text: List[str]) -> torch.Tensor:
        """
            Input text, after processing return the embedding represents text
        """

    def text_mapping(self, text:List[str]) -> torch.Tensor:
        return self._text_mapping(text)

    @abstractmethod
    def _pre_project_mapping(self, kernel_features: torch.Tensor) -> torch.Tensor:
        """
            Input original kernel feature: n*c, return n*c' feature
        """
        return kernel_features

    def pre_project_mapping(self, kernel_features: torch.Tensor) -> torch.Tensor:
        return self._pre_project_mapping(kernel_features)

    @abstractmethod
    def _post_project_mapping(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
            Input feature_map, H*W*C, after processing return H*W*C'
        """
        return feature_map

    def post_project_mapping(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self._post_project_mapping(feature_map)
    



"""
    We are using featup MASKCLIP model for feature lifting
"""
class feature_lift_mapper_config(base_feature_mapper_config):
    text_tokenizer: str = "featup"  # I think they are the same, feat up model
    text_encoder: str = 'maskclip'  # 
    gaussian_feature_decoder: Optional[str] = None  # Path to pre-project-mapping model
    feature_map_decoder: Optional[str] = None  # Path to post-project-mapping model


class feature_lift_mapper(base_feature_mapper):
    def __init__(self, config: feature_lift_mapper_config) -> None:
        super().__init__(config)
    

    def _load_tokenizer(self, tokenizer_id: str):
            return tokenize

    def _load_model(self, path: Optional[str]) -> Optional[torch.nn.Module]:
        if path == None:
            return None
        else:
            return torch.hub.load("mhamilton723/FeatUp", path, use_norm=False).cuda().model.model
    
    def _text_mapping(self, texts: List[str]) -> torch.Tensor:   
        """already normalized text feature"""
        text_features = self.text_encoder.encode_text(self.tokenizer(texts).cuda()).float()
        text_features = F.normalize(text_features, dim=0) # mc, the positive word is the last one
        return text_features

    def _pre_project_mapping(self, kernel_features: torch.Tensor) -> torch.Tensor:

        kernel_features = F.normalize(kernel_features.float(), dim=1)
        return kernel_features
    
    def _post_project_mapping(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map