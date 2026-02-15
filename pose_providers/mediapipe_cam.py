"""MediaPipe monocular-camera head pose provider."""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from ..control.pose import Pose6D
from ..control.pose_provider import PoseProvider
from ..math3d.quaternion import q_normalize, rotmat_to_q

logger = logging.getLogger(__name__)


def _ensure_task_model(task_model_path: str, task_model_url: str) -> Path:
    path = Path(task_model_path)
    if path.exists():
        return path

    if not task_model_url:
        raise RuntimeError(
            f"MediaPipe .task model missing: {path}. Set --mp-task-url or provide local --mp-task-model."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".download")
    logger.info("[POSE] downloading MediaPipe task model -> %s", path)
    try:
        urllib.request.urlretrieve(task_model_url, tmp_path)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Failed to download MediaPipe task model from {task_model_url}: {exc}"
        ) from exc
    os.replace(tmp_path, path)
    return path


class MediaPipeCamPoseProvider(PoseProvider):
    """
    Webcam-based head orientation provider.

    Uses MediaPipe Face Mesh landmarks and solvePnP to estimate a head pose
    rotation, then exposes it as a quaternion [w, x, y, z].

    Native frame is camera-centric; acoustic front/back semantics are aligned
    downstream by the acoustic frame provider.
    """

    # 6 landmark indices: nose, chin, eye corners, mouth corners.
    _LM_IDS = (1, 152, 33, 263, 61, 291)
    _MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -63.6, -12.5),  # chin
            (-43.3, 32.7, -26.0),  # left eye outer corner
            (43.3, 32.7, -26.0),  # right eye outer corner
            (-28.9, -28.9, -24.1),  # left mouth corner
            (28.9, -28.9, -24.1),  # right mouth corner
        ],
        dtype=np.float64,
    )

    # Convert from OpenCV camera basis (x right, y down, z forward)
    # to app basis (x right, y up, z forward).
    _CV_TO_APP = np.diag(np.array([1.0, -1.0, 1.0], dtype=np.float64))
    # Camera-facing "front" should map to acoustic +Z, so native front/back is flipped.
    default_acoustic_frame_provider_name = "flip-front"

    def __init__(
        self,
        title: str,
        source_world: np.ndarray,
        sr: int,
        camera_index: int = 0,
        camera_width: int = 1280,
        camera_height: int = 720,
        mirror: bool = True,
        smoothing: float = 0.35,
        translation_scale: float = 0.001,
        position_smoothing: float = 0.25,
        task_model_path: str = "assets/models/face_landmarker.task",
        task_model_url: str = "",
    ):
        self.window_name = title
        self.source_world = np.asarray(source_world, dtype=np.float64)
        self.sr = int(sr)
        self.camera_index = int(camera_index)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.mirror = bool(mirror)
        self.smoothing = float(max(0.01, min(1.0, smoothing)))
        self.translation_scale = float(translation_scale)
        self.position_smoothing = float(max(0.01, min(1.0, position_smoothing)))
        self.task_model_path = str(task_model_path)
        self.task_model_url = str(task_model_url)

        self._status_text = ""
        self._closed = False
        self._on_tick = None
        self._has_face = False

        self._q_current = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._p_current = np.zeros(3, dtype=np.float64)
        self._q_prev = None
        self._p_prev = None
        self._R_ref_cv = None
        self._t_ref_app = None

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.camera_index}")

        if self.camera_width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        model_path = _ensure_task_model(self.task_model_path, self.task_model_url)

        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "mediapipe.tasks is unavailable. Please upgrade mediapipe (>=0.10)."
            ) from exc
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._video_ts_ms = 0
        self._t0 = time.monotonic()

        logger.info(
            "[POSE] provider=mediapipe (camera=%s, mirror=%s, smoothing=%.2f, "
            "pos_smoothing=%.2f, translation_scale=%.5f, task=%s)",
            self.camera_index,
            self.mirror,
            self.smoothing,
            self.position_smoothing,
            self.translation_scale,
            model_path,
        )

    def _camera_matrix(self, w: int, h: int) -> np.ndarray:
        f = float(max(w, h))
        return np.array(
            [[f, 0.0, w * 0.5], [0.0, f, h * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _estimate_pose(self, face_lm, w: int, h: int) -> tuple[np.ndarray, np.ndarray] | None:
        # MediaPipe may provide landmarks as either a sequence directly or
        # a proto-like object that exposes a `.landmark` field.
        points = face_lm.landmark if hasattr(face_lm, "landmark") else face_lm
        image_points = []
        for idx in self._LM_IDS:
            p = points[idx]
            image_points.append((float(p.x) * w, float(p.y) * h))
        image_points = np.array(image_points, dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS,
            image_points,
            self._camera_matrix(w, h),
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        R_cv, _ = cv2.Rodrigues(rvec)
        if self._R_ref_cv is None:
            self._R_ref_cv = R_cv.copy()

        # Relative orientation against startup pose.
        R_rel_cv = self._R_ref_cv.T @ R_cv
        R_rel = self._CV_TO_APP @ R_rel_cv @ self._CV_TO_APP
        q = rotmat_to_q(R_rel)

        if self._q_prev is not None and float(np.dot(self._q_prev, q)) < 0.0:
            q = -q
        if self._q_prev is not None and self.smoothing < 1.0:
            a = self.smoothing
            q = q_normalize((1.0 - a) * self._q_prev + a * q)

        t_app = self._CV_TO_APP @ np.asarray(tvec, dtype=np.float64).reshape(3)
        if self._t_ref_app is None:
            self._t_ref_app = t_app.copy()
        p = (t_app - self._t_ref_app) * self.translation_scale
        if self._p_prev is not None and self.position_smoothing < 1.0:
            b = self.position_smoothing
            p = (1.0 - b) * self._p_prev + b * p

        self._q_prev = q
        self._p_prev = p
        return q, p

    def get_pose(self) -> Pose6D:
        return Pose6D(
            position=self._p_current.copy(),
            quaternion=self._q_current.copy(),
        )

    def get_quaternion(self) -> np.ndarray:
        return self._q_current.copy()

    def signfix_enabled(self) -> bool:
        # Already handled in this provider.
        return False

    def get_camera_world_origin(self) -> np.ndarray | None:
        if self._t_ref_app is None:
            return None
        return (-self._t_ref_app * self.translation_scale).astype(np.float64, copy=True)

    def has_tracking(self) -> bool:
        return bool(self._has_face)

    def set_status(self, text: str) -> None:
        self._status_text = text

    def _draw_overlay(self, frame: np.ndarray) -> None:
        lines = [
            (
                f"Source xyz: ({self.source_world[0]:.2f}, "
                f"{self.source_world[1]:.2f}, {self.source_world[2]:.2f}) m | SOFA sr={self.sr}"
            ),
            "Keys: [q/ESC] quit, [r] recenter head pose",
            f"Face tracked: {self._has_face}",
            (
                f"Head xyz: ({self._p_current[0]:.2f}, "
                f"{self._p_current[1]:.2f}, {self._p_current[2]:.2f}) m"
            ),
        ]
        if not self._has_face:
            lines.append("No face detected. Keep face centered and well lit.")
        if self._status_text:
            lines.extend(self._status_text.splitlines()[:4])

        y = 24
        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (80, 255, 80),
                1,
                cv2.LINE_AA,
            )
            y += 22

    def run(self, on_tick):
        self._on_tick = on_tick
        while not self._closed:
            ok, frame = self.cap.read()
            if not ok:
                if self._on_tick is not None:
                    self._on_tick()
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                time.sleep(0.01)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - self._t0) * 1000.0)
            self._video_ts_ms = max(self._video_ts_ms + 1, ts_ms)
            result = self._landmarker.detect_for_video(mp_image, self._video_ts_ms)

            self._has_face = False
            if result.face_landmarks:
                pose = self._estimate_pose(result.face_landmarks[0], w, h)
                if pose is not None:
                    self._q_current, self._p_current = pose
                    self._has_face = True

            if self._on_tick is not None:
                self._on_tick()

            self._draw_overlay(frame)
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                self._R_ref_cv = None
                self._t_ref_app = None
                self._q_prev = None
                self._p_prev = None

        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._landmarker.close()
        except (AttributeError, RuntimeError):
            pass
        try:
            self.cap.release()
        except (AttributeError, cv2.error):
            pass
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
