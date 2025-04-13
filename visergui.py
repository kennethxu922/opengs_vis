from genericpath import exists
from threading import Thread
import torch
import numpy as np
import time
import viser
import cv2
from collections import deque
from nerfstudio.cameras.cameras import Cameras
import torch
from renderer import renderer
import tempfile
import os
import shutil
import gradio as gr

# Define image height and width.
H = 480
W = 640 

# Define camera intrinsic parameters.
K = np.array([
    [525, 0, 320],  # fx, 0, cx
    [0, 525, 240],  # 0, fy, cy
    [0, 0,   1]     # 0,  0,  1
])


# Empty RenderThread for further extension.
class RenderThread(Thread):
    pass

# Convert quaternion vector to rotation matrix.
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
    def __init__(self, device, viewer_port):
        """
        Initialize the ViserViewer with rendering device and viewer port.
        Sets up the GUI elements including mode selection, segmentation controls,
        and event listeners.
        """
        self.device = device
        self.port = viewer_port
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.export_number = 0
        # -----------------------------
        # GUI Controls for General Mode
        # -----------------------------
        self.text_input = self.server.add_gui_text("Prompt", initial_value="Text Prompy")
        self.submit_button = self.server.add_gui_button("Submit")
        # Dropdown for display mode: now includes "RGB", "Attention", and "Segmentation".
        self.single_display_mode = self.server.add_gui_dropdown(
            "Display Mode", ("RGB", "Attention", "Segmentation", "Feature_PCA"), initial_value="RGB"
        )
        # Checkbox for split view.
        self.split_mode = self.server.add_gui_checkbox("Split View", initial_value=False)
        self.ratio = self.server.add_gui_slider(
            "Ratio", min=0.1, max=0.9, step=0.1, initial_value=0.5
        )
        # Dropdowns for left/right modes in split view.
        self.display_mode_left = self.server.add_gui_dropdown(
            "Left Mode", ("RGB", "Attention", "Segmentation", "Feature_PCA"), initial_value="RGB"
        )
        self.display_mode_right = self.server.add_gui_dropdown(
            "Right Mode", ("RGB", "Attention", "Segmentation", "Feature_PCA"), initial_value="Attention"
        )

        # -----------------------------
        # GUI Controls for Segmentation Mode
        # These controls will only be visible when segmentation is active.
        # -----------------------------
        self.seg_positive = self.server.add_gui_text("Positive Description", initial_value="")
        self.seg_background = self.server.add_gui_text("Background Words", initial_value="")
        self.seg_submit_button = self.server.add_gui_button("Segmentation Submit")
        self.seg_export_button = self.server.add_gui_button("Export Segmentation")
        # Initially hide segmentation controls.
        self.seg_positive.visible = False
        self.seg_background.visible = False
        self.seg_submit_button.visible = False
        self.seg_export_button.visible = False

        # Initialize parameters for updates and segmentation result.
        self.need_update = True
        self.current_modes = ("RGB", "Attention")
        self.split_enabled = False

        # -----------------------------
        # Event Listener for Split Mode Toggle
        # -----------------------------
        @self.split_mode.on_update
        def _(event):
            # Toggle split view and update related controls.
            self.split_enabled = self.split_mode.value
            self.ratio.disabled = not self.split_enabled
            self.display_mode_left.disabled = not self.split_enabled
            self.display_mode_right.disabled = not self.split_enabled
            self.single_display_mode.disabled = self.split_enabled
            self.need_update = True
            self.update_segmentation_controls_visibility()

        # -----------------------------
        # Event Listener for Client Camera Updates
        # -----------------------------
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        # -----------------------------
        # Event Listener for Main Submit Button
        # (For non-segmentation related text input)
        # -----------------------------
        @self.submit_button.on_click
        def _(_):
            text = self.text_input.value
            if hasattr(self.renderer, 'attention_score'):
                self.renderer.attention_score(text)
                self.need_update = True  # Trigger render update if needed
            else:
                print("Renderer does not support text updates!")

        # -----------------------------
        # Event Listeners for Mode Dropdowns to update segmentation controls
        # -----------------------------
        @self.single_display_mode.on_update
        def _(_):
            self.update_segmentation_controls_visibility()
            self._request_update()

        @self.display_mode_left.on_update
        def _(_):
            self.update_segmentation_controls_visibility()
            self._request_update()

        @self.display_mode_right.on_update
        def _(_):
            self.update_segmentation_controls_visibility()
            self._request_update()

        # -----------------------------
        # Event Listeners for Segmentation Controls
        # -----------------------------
        @self.seg_submit_button.on_click
        def _(_):
            self.segmentation_calculation()

        @self.seg_export_button.on_click
        def _(_):
            self.export_segmentation()

        # -----------------------------
        # Additional GUI Controls (FPS display)
        # -----------------------------
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)
        # Debounce timer to prevent rapid updates.
        self.last_update_time = time.time()
        self.debounce_interval = 0.1  # 100ms

        # Setup other event listeners.
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """
        Set up event listeners for multiple GUI elements.
        This helps trigger updates when any control value changes.
        """
        for gui_element in [
            self.split_mode,
            self.ratio,
            self.display_mode_left,
            self.display_mode_right,
            self.single_display_mode,
            self.text_input
        ]:
            @gui_element.on_update
            def _(_):
                self._request_update()
            # Also, force an immediate update when submit button is clicked.
            @self.submit_button.on_click
            def _(_):
                self._request_update(force=True)

    def update_segmentation_controls_visibility(self):
        """
        Update the visibility of segmentation-specific GUI controls.
        Controls are shown if segmentation mode is active in the current view.
        In split view, if either left or right mode is segmentation, controls will show.
        In single view, controls show if the selected mode is segmentation.
        """
        is_segmentation = False
        if self.split_enabled:
            if self.display_mode_left.value == "Segmentation" or self.display_mode_right.value == "Segmentation":
                is_segmentation = True
        else:
            if self.single_display_mode.value == "Segmentation":
                is_segmentation = True

        # Update visibility (assuming viser supports the 'visible' attribute).
        self.seg_positive.visible = is_segmentation
        self.seg_background.visible = is_segmentation
        self.seg_submit_button.visible = is_segmentation
        self.seg_export_button.visible = is_segmentation

    def _request_update(self, force=False):
        """
        Request a render update with debouncing.
        This ensures that rendering is not triggered too frequently.
        """
        now = time.time()
        if force or (now - self.last_update_time > self.debounce_interval):
            self.last_update_time = now
            self.update()

    def set_renderer(self, renderer:renderer):
        """
        Set the renderer instance to be used.
        The renderer must support the required methods (render, attention_score, segmentation_calculation, etc.).
        """
        self.renderer = renderer

    @torch.no_grad()
    def update(self):
        """
        Update function to render images from each connected client.
        Chooses the appropriate rendering mode (RGB, Attention, or Segmentation) based on the current GUI selection.
        """
        if self.need_update:
            start = time.time()
            update_success = False
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = get_w2c(camera)
                try:
                    # Determine the current mode and render accordingly.
                    if self.split_enabled:
                        left_mode = self.display_mode_left.value
                        right_mode = self.display_mode_right.value
                        split_pos = int(W * self.ratio.value)
                        left_img = self.render_view(w2c, K, left_mode, H, W)
                        right_img = self.render_view(w2c, K, right_mode, H, W)
                        composite = np.concatenate([
                            left_img[:, :split_pos],
                            right_img[:, split_pos:]
                        ], axis=1)
                    else:
                        mode = self.single_display_mode.value
                        composite = self.render_view(w2c, K, mode, H, W)

                    # Send the composite image to the client.
                    client.set_background_image(
                        composite, 
                        format="jpeg",
                    )
                    update_success = True
                except Exception as e:
                    print(f"Rendering error: {e}")
                    continue

            # Update FPS display based on render times.
            interval = time.time() - start
            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.2f}"
            self.need_update = not update_success

    def render_view(self, w2c: np.ndarray, k, mode, H, W):
        """
        Render view function that calls the renderer based on the mode.
        If segmentation mode is active and a segmentation result exists (from a prior calculation),
        it will be used. Otherwise, it falls back to the renderer's render method.
        """
        w2c = torch.tensor(w2c)
        # Call renderer with the current mode.
        img: torch.Tensor = self.renderer.render(
            w2c,
            k,
            mode,  # Renderer should support the mode parameter ("RGB", "Attention", or "Segmentation")
            H, W
        )
        img = img.cpu().detach().numpy()
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        # In case of segmentation mode without a precomputed result, the renderer may perform segmentation rendering.
        return img

    def segmentation_calculation(self):
        """
        Perform segmentation calculation using the provided positive description and background words.
        Calls the renderer's segmentation_calculation method (if available) and stores the result.
        Triggers an update to display the segmentation output.
        """
        pos_text = self.seg_positive.value
        bg_text = self.seg_background.value
        self.renderer.segmentation(pos_text, bg_text)

    def export_segmentation(self):
        """
        Export the current segmentation result to a downloadable file.
        If a segmentation result exists, export (e.g., save) it to a temporary file,
        then pass it to the frontend for download.
        Otherwise, prompt the user to perform a segmentation calculation first.
        """
        if self.renderer.segmentation_mask is not None:
            os.makedirs("./exported_gaussian_id", exist_ok=True)
            torch.save(self.renderer.segmentation_mask, os.path.join("./exported_gaussian_id", str(self.export_number)))
            self.export_number += 1
        else:
            print("No segmentation result to export. Please perform segmentation calculation first.")
