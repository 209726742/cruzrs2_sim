from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False


class HeadStereoVisualizer:
    """OpenCV-based camera viewer with producer-consumer threading.

    Producer thread: simulation/render callback pushes latest camera frames.
    Consumer thread: dedicated display thread renders a 2x2 figure.
    """

    CAMERA_ORDER = ("head_left", "head_right", "wrist_left", "wrist_right")

    def __init__(
        self,
        enabled: bool = False,
        window_name: str = "walker_s2_cameras",
        scale: float = 1.0,
        every_n: int = 1,
        window_x: int = 40,
        window_y: int = 40,
        show_labels: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._window_name = str(window_name)
        self._scale = max(float(scale), 0.1)
        self._every_n = max(int(every_n), 1)
        self._window_x = int(window_x)
        self._window_y = int(window_y)
        self._show_labels = bool(show_labels)

        self._started = False
        self._frame_idx = 0
        self._warned_unavailable = False
        self._warned_runtime = False
        self._display_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[dict[str, np.ndarray]] = queue.Queue(maxsize=2)

    def is_enabled(self) -> bool:
        return self._enabled and OPENCV_AVAILABLE

    def start(self) -> None:
        if not self._enabled:
            return
        if not OPENCV_AVAILABLE:
            if not self._warned_unavailable:
                logger.warning("HeadStereoVisualizer disabled: opencv-python is unavailable")
                self._warned_unavailable = True
            return
        if self._started:
            return

        try:
            self._stop_event.clear()
            self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self._display_thread.start()
            self._started = True
        except Exception as exc:
            if not self._warned_runtime:
                logger.warning("HeadStereoVisualizer start failed: %s", exc)
                self._warned_runtime = True
            self._started = False

    def update_cameras(self, camera_frames: dict[str, np.ndarray]) -> None:
        """Producer API: enqueue latest frames for display thread."""
        if not self._enabled or not OPENCV_AVAILABLE:
            return
        if not self._started:
            self.start()
        if not self._started:
            return

        self._frame_idx += 1
        if (self._frame_idx % self._every_n) != 0:
            return

        payload = {
            name: frame.copy()
            for name, frame in camera_frames.items()
            if frame is not None
        }
        if not payload:
            return

        # Keep latest data only to avoid rendering lag when simulation runs fast.
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            pass

    def update_frames(self, left_rgb: Optional[np.ndarray], right_rgb: Optional[np.ndarray]) -> None:
        # Backward compatibility wrapper.
        camera_frames: dict[str, np.ndarray] = {}
        if left_rgb is not None:
            camera_frames["head_left"] = left_rgb
        if right_rgb is not None:
            camera_frames["head_right"] = right_rgb
        self.update_cameras(camera_frames)

    def stop(self) -> None:
        if not self._started:
            return

        self._stop_event.set()
        if self._display_thread is not None:
            self._display_thread.join(timeout=2.0)
            self._display_thread = None

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._started = False

    def _display_loop(self) -> None:
        if not OPENCV_AVAILABLE:
            return

        last_payload: dict[str, np.ndarray] = {}
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self._window_name, self._window_x, self._window_y)
        except Exception as exc:
            if not self._warned_runtime:
                logger.warning("HeadStereoVisualizer display setup failed: %s", exc)
                self._warned_runtime = True
            return

        try:
            while not self._stop_event.is_set():
                try:
                    payload = self._frame_queue.get(timeout=0.05)
                    last_payload.update(payload)
                except queue.Empty:
                    pass

                canvas = self._compose_canvas(last_payload)
                if canvas is None:
                    continue

                cv2.imshow(self._window_name, canvas)
                cv2.waitKey(1)
        except Exception as exc:
            if not self._warned_runtime:
                logger.warning("HeadStereoVisualizer display loop failed: %s", exc)
                self._warned_runtime = True
        finally:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
            try:
                cv2.waitKey(1)
            except Exception:
                pass

    def _compose_canvas(self, payload: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if not payload:
            return None

        base_frame = next(iter(payload.values()))
        h, w = int(base_frame.shape[0]), int(base_frame.shape[1])
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        tiles: list[np.ndarray] = []
        for cam_name in self.CAMERA_ORDER:
            rgb = payload.get(cam_name)
            if rgb is None:
                tile = blank.copy()
            else:
                tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if tile.shape[0] != h or tile.shape[1] != w:
                    tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA)

            if self._show_labels:
                cv2.putText(
                    tile,
                    cam_name,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            tiles.append(tile)

        top_row = np.hstack([tiles[0], tiles[1]])
        bottom_row = np.hstack([tiles[2], tiles[3]])
        canvas = np.vstack([top_row, bottom_row])

        if self._scale != 1.0:
            canvas = cv2.resize(
                canvas,
                None,
                fx=self._scale,
                fy=self._scale,
                interpolation=cv2.INTER_AREA,
            )
        return canvas
