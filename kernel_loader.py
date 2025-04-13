import torch
from plyfile import PlyData
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Literal
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
import os
import torch.nn.functional as F



@dataclass 
class base_kernel_loader_config:
    kernel_location: str # required
    feature_location: Optional[str] = None
    feature_id_location: Optional[str] = None
    kernel_type: Literal["ply", "ckpt"] = "ckpt"
    feature_type: Literal["pt", "ftz"] = "pt" # ftz is our proposed memory efficient compressing method

 

@dataclass
class Kernel:
    geometry: Dict[str, torch.Tensor]
    color: Optional[torch.Tensor] = None
    feature: Optional[torch.Tensor] = None
    feature_id: Optional[torch.Tensor] = None
    meta: Optional[Dict] = None # Others, for future development


class base_kernel_loader:

    """
        It should be a ply file, a feature + feature.pt file
        Or it should be a ply file
        Or ckpt file + feature.pt file
        or ckpt file
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def _load(self,**kwargs: Optional[Dict])->Kernel:
        """
            It should load the following:
            Geometries: if it is Gaussian, we load X, Alpha, Quat, Scales, if other kernel, use other kernel
            Color: fd_c or something else
            Features: The feature you want to embeds
            Feature ID(optional): if we have, we use feature ID to select a subsect from geometry paird with feature
        """

    def load(self,**kwargs: Optional[Dict]):
        """
            We think the thing we need to load is in config, if it is not in config, pass something else, I don't know
        """

        return self._load(**kwargs)


class general_gaussian_loader(base_kernel_loader):
    config: base_kernel_loader_config
    def __init__(self, config) -> None:
        super().__init__(config)
        
    
    def _load(self, **kwargs: Optional[Dict]) -> Kernel:
        
        model_location = self.config.kernel_location
        assert os.path.exists(model_location), f"input kernel location does not exist {model_location}"

        geometry = None # we need to assign it to kernel
        colors = None
        feature = None # Sometime, we might not load it, just for visualization
        feature_id = None


        ## TO DO @ Kenneth, how to load feature here for openGaussian!
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
            means = torch.tensor(xyz, dtype=torch.float, device="cuda")
            quats = F.normalize(torch.tensor(rots, dtype=torch.float, device="cuda"), p=2, dim=-1)
            scales = torch.tensor(scales, dtype=torch.float, device="cuda")
            opacities = torch.tensor(opacities, dtype=torch.float, device="cuda")
            sh0 = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            shN = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            geometry = {"means": means, "quats": quats, "scales": scales, "opacities": opacities}
            colors = torch.cat((sh0, shN), dim=1)
            torch.cuda.empty_cache()
        
        
        elif model_location.endswith('ckpt'):
            model_config = SplatfactoModelConfig()
            model: SplatfactoModel = model_config.setup(scene_box = None, num_train_data = 1)
            state_dict: dict = torch.load(model_location)['pipeline']
            state_dict = {".".join(key.split('.')[1:]): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            means: torch.Tensor = model.means.cuda()
            quats: torch.Tensor = model.quats.cuda()
            opacities: torch.Tensor = model.opacities.cuda()
            scales: torch.Tensor = model.scales.cuda()
            colors: torch.Tensor = torch.cat((model.features_dc[:, None, :], model.features_rest), dim=1).cuda()
            geometry = {"means": means, "quats": quats, "scales": scales, "opacities": opacities}

        else:
            print(f"currently we only accept ply and ckpt file, but get: {model_location}")
            raise NotImplementedError


        if self.config.feature_location != None:
            if self.config.feature_type == "ftz":
                """
                @ Yihan Fang: TO DO
                Load the ftz file correctly
                """
                print("CONTACT YIHAN FANG")
                assert self.config.feature_location.endswith("ftz"), f"feature type must be ftz as user required, but get feature path: {self.config.feature_location}"
                raise NotImplementedError

            elif self.config.feature_type == 'pt':
                assert self.config.feature_location.endswith("pt"), f"feature type must be pt as user required, but get feature path: {self.config.feature_location}"
                feature = torch.load(self.config.feature_location).cuda()

            else:
                print(f"currently we only accept pt and ftz file, but get: {model_location}")
                raise NotImplementedError

            if len(feature) < len(geometry["means"]):
                print("detect less feature than geometry, assume finish the feature downsampling")
                assert os.path.exists(self.config.feature_id_location), f"feature ID location not correct: {self.config.feature_id_location}"
                feature_id = torch.load(self.config.feature_id_location).cuda()            

        else:
            print("!!!Detect No feature input!!!")
            print("We assume we only need to render RGB !!!!!!!!")

        

        kernel = Kernel(
            geometry= geometry,
            color = colors,
            feature = feature,
            feature_id = feature_id,
        )
        
        return kernel

        
