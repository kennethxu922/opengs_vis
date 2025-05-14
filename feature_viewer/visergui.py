import time
import os
import numpy as np
import torch
import viser
from collections import deque
from nerfstudio.cameras.cameras import Cameras
from .renderer import Renderer
import io
from viser import IconName



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

import time
import os
import numpy as np
import torch
import viser
import cv2
from collections import deque
from nerfstudio.cameras.cameras import Cameras
from .renderer import Renderer

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
        self.error_start = None
        self.last_img = None

        # Base resolution
        self.H = self.base_H = 480
        self.W = self.base_W = 640
        self.K = self.base_K = np.array([[525,0,320],[0,525,240],[0,0,1]],dtype=np.float32)


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
            {"name": "Resolution Scale",    "kind": "slider",   "min": 1, "max": 8, "step": 1, "initial_value": 3},
            {"name": "Prompt",              "kind": "text",     "initial_value": "Text Prompt"},
            {"name": "Submit",              "kind": "button"},
            {"name": "Display Mode",        "kind": "dropdown", "choices": ("RGB","Attention","Segmentation","Feature_PCA","Weight_Filtered","Mask"),"initial_value":"RGB"},
            {"name": "Split View",          "kind": "checkbox", "initial_value": False},
            {"name": "Ratio",               "kind": "slider",   "min": 0.1, "max": 0.9, "step": 0.05, "initial_value": 0.5},
            {"name": "Left Mode",           "kind": "dropdown", "choices": ("RGB","Attention","Segmentation","Feature_PCA","Weight_Filtered","Mask"),"initial_value":"RGB"},
            {"name": "Right Mode",          "kind": "dropdown", "choices": ("RGB","Attention","Segmentation","Feature_PCA","Weight_Filtered","Mask"),"initial_value":"Attention"},
            {"name": "Positive Description","kind": "text",     "initial_value": ""},
            {"name": "Background Words",    "kind": "text",     "initial_value": ""},
            {"name": "Segmentation Submit", "kind": "button"},
            {"name": "Export Segmentation", "kind": "button"},
            {"name": "Weight Value",        "kind": "slider",   "min": 0.0, "max": 1.0, "step": 0.01, "initial_value": 0.5},
            {"name": "Gaussian Scale",      "kind": "slider",   "min": 0.1, "max": 10.0, "step": 0.1, "initial_value": 1.0},
            {"name": "Export Camera",       "kind": "button"},
            {"name": "FPS",                 "kind": "text",     "initial_value": "-1", "disabled": True},
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
        self.upload_button = self.server.gui.add_upload_button(
            "Load Camera PT", icon=viser.Icon.UPLOAD
        )

    def _bind_events(self):
        c = self.controls
        # Resolution change
        c['Resolution Scale'].on_update(self._on_resolution)
        # Core handlers
        c['Submit'].on_click(self._on_submit)
        for key in ('Display Mode','Split View','Ratio','Left Mode','Right Mode'):
            c[key].on_update(self._flag_update)
        c['Segmentation Submit'].on_click(self._on_segmentation)
        c['Export Segmentation'].on_click(self._on_export_seg)
        c['Weight Value'].on_update(self._on_weight)
        c['Gaussian Scale'].on_update(self._on_gaussian)
        c['Export Camera'].on_click(self._on_export_camera)
        self.upload_button.on_upload(self._on_camera_upload)



    def set_renderer(self, renderer: Renderer):
        self.renderer = renderer
        self.renderer.weight_filtering()
        

    def _on_resolution(self, _=None):
        # update H,W,K based on slider
        R = int(self.controls['Resolution Scale'].value)
        self.H = self.base_H * R
        self.W = self.base_W * R
        K = self.base_K.copy()
        self.K = K*R
        self.K[-1,-1] = 1
        self._flag_update()

    def _flag_update(self, _=None):
        self.need_update = True

    def _on_submit(self, _=None):
        txt = self.controls['Prompt'].value
        if self.renderer and hasattr(self.renderer, 'attention_score'):
            self.renderer.attention_score([txt])
        self._flag_update()

    def _on_segmentation(self, _=None):
        pos = self.controls['Positive Description'].value
        bg  = self.controls['Background Words'].value
        self.renderer.segmentation(pos, bg, True)
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
        clients = list(self.server.get_clients().values())
        if not clients:
            return
        for client in clients:
            cam = client.camera
            data = {
                'wxyz': cam.wxyz,
                'position': cam.position,
                'aspect': cam.aspect,
                'fov': cam.fov,
                'look_at': cam.look_at,
                'up_direction': cam.up_direction,
            }
            buf = io.BytesIO()
            np.savez(buf,
                     wxyz=data['wxyz'],
                     position=data['position'],
                     aspect=data['aspect'],
                     fov=data['fov'],
                     look_at=data['look_at'],
                     up_direction=data['up_direction'])
            buf.seek(0)
            self.server.send_file_download(
                filename=f"camera_{self.export_count:03d}.npz",
                content=buf.read(),
            )
            self.export_count += 1
        self._flag_update()
        
    def _on_camera_upload(self, _=None):
        file = self.upload_button.value
        try:
            buf = io.BytesIO(file.content)
            data = np.load(buf)
        except Exception as e:
            print(f"Failed loading camera file {file.name}: {e}")
            return
        for client in self.server.get_clients().values():
            print("camera set ")
            client.camera.position = data['position']
            client.camera.wxyz = data['wxyz']
        self._flag_update()
        


    @torch.no_grad()
    def update(self):
        now = time.time()
        if not self.need_update or (now - self.last_update < self.debounce):
            return
        self.last_update = now
        start = now
        c = self.controls
        split = c['Split View'].value

        # GUARD against segmentation
        mode = c['Display Mode'].value
        if mode == 'Segmentation' and self.renderer.segmentation_mask is None:
            return
        if split:
            lm, rm = c['Left Mode'].value, c['Right Mode'].value
            if ('Segmentation' in (lm, rm) and self.renderer.segmentation_mask is None):
                return

        # Render with error-handling
        try:
            for client in self.server.get_clients().values():
                c2w = get_c2w(client.camera)
                w2c = np.linalg.inv(c2w)
                if split:
                    left  = self.renderer.render(torch.tensor(w2c), self.K, c['Left Mode'].value, self.H, self.W)
                    right = self.renderer.render(torch.tensor(w2c), self.K, c['Right Mode'].value, self.H, self.W)
                    cut = int(left.shape[1] * c['Ratio'].value)
                    img = np.concatenate([left[:,:cut], right[:,cut:]], axis=1)
                else:
                    img = self.renderer.render(torch.tensor(w2c), self.K, mode, self.H, self.W)

                img = self._prepare_image(img)
                client.set_background_image(img, format="jpeg")
                self.last_img = img
            # reset error timer on success
            self.error_start = None
        except Exception as e:
            # first error: record start
            if self.error_start is None:
                self.error_start = time.time()
            # if within 3s, show loading overlay
            if time.time() - self.error_start < 3:
                if self.last_img is not None:
                    overlay = self._make_loading_overlay(self.last_img)
                    for client in self.server.get_clients().values():
                        client.set_background_image(overlay, format="jpeg")
                return
            # else rethrow after timeout
            raise

        # FPS and clear flag
        self.render_times.append(time.time()-start)
        fps = 1.0/np.mean(self.render_times) if self.render_times else 0
        c['FPS'].value = f"{fps:.2f}"
        self.need_update = False

    def _prepare_image(self, img):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def _make_loading_overlay(self, img):
        overlay = img.copy()
        h, w = overlay.shape[:2]
        alpha = 0.5
        # semi-transparent black bar
        cv2.rectangle(overlay, (0,h//2-20), (w,h//2+20), (0,0,0), -1)
        # blend
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, overlay)
        # text
        cv2.putText(overlay, "Loading...", (w//2 - 60, h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return overlay
