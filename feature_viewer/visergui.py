import time
import os
import numpy as np
import torch
import viser
from collections import deque
from nerfstudio.cameras.cameras import Cameras
from .renderer import Renderer

R = 3
# Define image height and width.
H = 480 * R
W = 640 * R

# Define camera intrinsic parameters.
K = np.array([
    [525 * R, 0, 320 * R],  # fx, 0, cx
    [0, 525 * R, 240 * R],  # 0, fy, cy
    [0, 0,   1]     # 0,  0,  1
])



def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

# Get camera-to-world transformation matrix.
def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

# Get world-to-camera transformation matrix.
def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

class ViserViewer:
    def __init__(self, device: torch.device, viewer_port: int):
        self.device = device
        self.port = viewer_port
        self.server = viser.ViserServer(port=self.port)
        self.renderer: Renderer = None
        self.export_count = 0
        self.render_times = deque(maxlen=30)
        self.need_update = True
        self.last_update = 0.0
        self.debounce = 0.1

        self._init_gui()
        self._bind_events()

        # Re-render when camera moves
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

    def _init_gui(self):
        specs = [
            {"name": "Prompt",               "kind": "text",     "initial_value": "Text Prompt"},
            {"name": "Submit",               "kind": "button"},
            {"name": "Display Mode",         "kind": "dropdown", "choices": ("RGB","Attention","Segmentation","Feature_PCA","Weight_Filtered","Mask"), "initial_value": "RGB"},
            {"name": "Split View",           "kind": "checkbox", "initial_value": False},
            {"name": "Ratio",                "kind": "slider",   "min": 0.1, "max": 0.9, "step": 0.05, "initial_value": 0.5},
            {"name": "Left Mode",            "kind": "dropdown", "choices": ("RGB","Attention","Segmentation","Feature_PCA","Weight_Filtered","Mask"), "initial_value": "RGB"},
            {"name": "Right Mode",           "kind": "dropdown", "choices": ("RGB","Attention","Segmentation","Feature_PCA","Weight_Filtered","Mask"), "initial_value": "Attention"},
            {"name": "Positive Description", "kind": "text",     "initial_value": ""},
            {"name": "Background Words",     "kind": "text",     "initial_value": ""},
            {"name": "Segmentation Submit",  "kind": "button"},
            {"name": "Export Segmentation",  "kind": "button"},
            {"name": "Weight Value",        "kind": "slider",   "min": 0.0, "max": 1.0, "step": 0.01, "initial_value": 0.5},
            {"name": "Gaussian Scale",       "kind": "slider",   "min": 0.1, "max": 1.0, "step": 0.1, "initial_value": 1.0},
            {"name": "Export Camera",        "kind": "button"},
            {"name": "FPS",                  "kind": "text",     "initial_value": "-1", "disabled": True},
        ]
        self.controls = {}
        for spec in specs:
            kind = spec.pop('kind')
            name = spec.pop('name')
            if kind == 'text':
                ctrl = self.server.add_gui_text(name, **spec)
            elif kind == 'button':
                ctrl = self.server.add_gui_button(name)
            elif kind == 'dropdown':
                choices = spec.pop('choices')
                ctrl = self.server.add_gui_dropdown(name, choices, **spec)
            elif kind == 'checkbox':
                ctrl = self.server.add_gui_checkbox(name, **spec)
            elif kind == 'slider':
                ctrl = self.server.add_gui_slider(name, **spec)
            else:
                raise ValueError(f"Unknown control kind: {kind}")
            self.controls[name] = ctrl

    def _bind_events(self):
        c = self.controls
        c['Submit'].on_click(self._on_submit)
        for key in ('Display Mode','Split View','Ratio','Left Mode','Right Mode'):
            c[key].on_update(self._flag_update)
        c['Segmentation Submit'].on_click(self._on_segmentation)
        c['Export Segmentation'].on_click(self._on_export_seg)
        c['Weight Value'].on_update(self._on_weight)
        c['Gaussian Scale'].on_update(self._on_gaussian)
        c['Export Camera'].on_click(self._on_export_camera)

    def set_renderer(self, renderer: Renderer):
        self.renderer = renderer

    def _flag_update(self, _=None):
        self.need_update = True

    def _on_submit(self, _=None):
        txt = self.controls['Prompt'].value
        if self.renderer and hasattr(self.renderer, 'attention_score'):
            self.renderer.attention_score(txt)
        self._flag_update()

    def _on_segmentation(self, _=None):
        pos = self.controls['Positive Description'].value
        bg  = self.controls['Background Words'].value
        self.renderer.segmentation(pos, bg, True)
        # ← new line: only enter Segmentation mode now that we have a mask
        self.controls['Display Mode'].value = 'Segmentation'
        self._flag_update()

    def _on_export_seg(self, _=None):
        mask = getattr(self.renderer, 'segmentation_mask', None)
        if mask is not None:
            os.makedirs('./exported_gaussian_id', exist_ok=True)
            torch.save(mask, f'./exported_gaussian_id/{self.export_count:03d}.pt')
            self.export_count += 1
        self._flag_update()

    def _on_weight(self, _=None):
        r = self.controls['Weight Value'].value
        self.renderer.weight_filtering(r)
        self._flag_update()

    def _on_gaussian(self, _=None):
        s = self.controls['Gaussian Scale'].value
        self.renderer.set_gaussian_scale(s)
        self._flag_update()

    def _on_export_camera(self, _=None):
        for client in self.server.get_clients().values():
            pose = get_c2w(client.camera)
            fn = f'camera_{self.export_count:03d}.npy'
            np.save(fn, pose)
            self.export_count += 1
        self._flag_update()

    @torch.no_grad()
    def update(self):
        now = time.time()
        if not self.need_update or (now - self.last_update < self.debounce):
            return
        self.last_update = now
        start = now
        c = self.controls
        # Visibility
        split = c['Split View'].value
        # Render
        # ─── GUARD AGAINST EMPTY SEGMENTATION ───
        mode = c['Display Mode'].value
        if mode == 'Segmentation' and self.renderer.segmentation_mask is None:
            return

        if c['Split View'].value:
            lm = c['Left Mode'].value
            rm = c['Right Mode'].value
            if ('Segmentation' in (lm, rm)
                and self.renderer.segmentation_mask is None):
                return

        for client in self.server.get_clients().values():
            c2w = get_c2w(client.camera)
            w2c = np.linalg.inv(c2w)
            if split:
                left  = self.renderer.render(torch.tensor(w2c), K, c['Left Mode'].value, H,W)
                right = self.renderer.render(torch.tensor(w2c), K, c['Right Mode'].value,H,W)
                w = left.shape[1]
                cut = int(w * c['Ratio'].value)
                img = torch.concatenate([left[:,:cut], right[:,cut:]], axis=1)
            else:
                img = self.renderer.render(torch.tensor(w2c), K, c['Display Mode'].value,H,W)
            img = self._prepare_image(img)
            client.set_background_image(img, format="jpeg")
        # FPS
        self.render_times.append(time.time()-start)
        fps = 1.0/np.mean(self.render_times) if self.render_times else 0
        c['FPS'].value = f"{fps:.2f}"
        self.need_update = False

    def _prepare_image(self, img, cut=None, start=0):
        # Convert torch or numpy to uint8 array
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # Crop if requested
        if cut is not None:
            img = img[:, :cut]
        if start:
            img = img[:, start:]
        # Scale float images to 0-255
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
