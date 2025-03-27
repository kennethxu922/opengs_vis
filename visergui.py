from threading import Thread
import torch
import numpy as np
import time
import viser
import cv2
from collections import deque
from nerfstudio.cameras.cameras import Cameras
import torch


H = 480
W = 640 

K = np.array([
    [525, 0, 320], #  fx, 0, cx
    [0, 525, 240], #  0, fy, cy
    [0, 0,   1]    #  0,  0,  1
])


class RenderThread(Thread):
    pass

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


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c



class ViserViewer:
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)

        self.text_input = self.server.add_gui_text("Prompt", initial_value="Text Prompy")
        self.submit_button = self.server.add_gui_button("Submit")
        # Add to __init__ after split mode controls
        self.single_display_mode = self.server.add_gui_dropdown(
            "Display Mode", ("RGB", "Attention"), initial_value="RGB"
        )
        # Display mode controls
        self.split_mode = self.server.add_gui_checkbox("Split View", initial_value=False)
        self.ratio = self.server.add_gui_slider(
            "Ratio", min=0.1, max=0.9, step=0.1, initial_value=0.5
        )
        self.display_mode_left = self.server.add_gui_dropdown(
            "Left Mode", ("RGB", "Attention"), initial_value="RGB"
        )
        self.display_mode_right = self.server.add_gui_dropdown(
            "Right Mode", ("RGB", "Attention"), initial_value="Attention"
        )


        # Initialize parameters
        self.need_update = True
        self.current_modes = ("RGB", "Attention")
        self.split_enabled = False


        @self.split_mode.on_update
        def _(event):
            self.split_enabled = self.split_mode.value
            # Disable split-related controls when split view is off
            self.ratio.disabled = not self.split_enabled
            self.display_mode_left.disabled = not self.split_enabled
            self.display_mode_right.disabled = not self.split_enabled
            # Enable single mode when split view is off
            self.single_display_mode.disabled = self.split_enabled
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        # Text input event handling
        @self.submit_button.on_click
        def _(_):
            text = self.text_input.value
            if hasattr(self.renderer, 'attention_score'):
                self.renderer.attention_score(text)
                self.need_update = True  # Trigger render update if needed
            else:
                print("Renderer does not support text updates!")

        # Performance monitoring
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)
        # Debounce timer to prevent rapid updates
        self.last_update_time = time.time()
        self.debounce_interval = 0.1  # 100ms

        # Setup event listeners
        self._setup_event_listeners()


        

    def _setup_event_listeners(self):
        # Add single display mode to listeners
        for gui_element in [
            self.split_mode,
            self.ratio,
            self.display_mode_left,
            self.display_mode_right,
            self.single_display_mode,  # Add this
            self.text_input
        ]:
            @gui_element.on_update
            def _(_):
                self._request_update()

            # Submit button
            @self.submit_button.on_click
            def _(_):
                self._request_update(force=True)  # Immediate update for text input


    def _request_update(self, force=False):
        """Trigger update with debouncing"""
        now = time.time()
        if force or (now - self.last_update_time > self.debounce_interval):
            self.last_update_time = now
            self.update()


    def set_renderer(self, renderer):
        """
            It needs the following argument: 
            K intrinsic metrics
            and w2c camera location given by 
        """
        self.renderer = renderer

    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            update_success = False
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = get_w2c(camera)

                
                try:
                    
                    # Get display configuration
                    if self.split_enabled:
                        left_mode = self.display_mode_left.value
                        right_mode = self.display_mode_right.value
                        split_pos = int(W * self.ratio.value)
                        
                        # Render left and right views
                        left_img = self.render_view(w2c, K, left_mode, H, W)
                        right_img = self.render_view(w2c, K, right_mode, H, W)
                        # Combine images
                        composite = np.concatenate([
                            left_img[:, :split_pos],
                            right_img[:, split_pos:]
                        ], axis=1)
                    else:
                        mode = self.single_display_mode.value
                        composite = self.render_view(w2c, K, mode, H, W)

                    # Send to client
                    client.set_background_image(
                        composite, 
                        format="jpeg",
                    )
                    update_success = True
                except Exception as e:
                    print(f"Rendering error: {e}")
                    continue

            # Update performance metrics
            interval = time.time() - start
            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.2f}"
            self.need_update = not update_success

    def render_view(self, w2c: np.ndarray, k, mode, H, W):
        """
            We need to actually input one camera model from nerfstudio.camera
        """

        w2c = torch.tensor(w2c)
        
        img: torch.Tensor = self.renderer.render(
            w2c,
            k,
            mode,  # Assuming renderer supports mode parameter
            H,W
        )
        
        # Post-processing based on mode
        img = img.cpu().detach().numpy()
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)