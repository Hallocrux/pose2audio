"""Kinect V2 6DoF pose provider via external bridge (e.g. C# process).

This provider intentionally avoids direct Kinect hardware SDK bindings.
It consumes 6DoF pose packets from localhost UDP JSON so a future C# bridge
can own device access and tracking logic.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Optional, Tuple

import numpy as np

from ..control.pose import Pose6D
from ..control.pose_provider import PoseProvider
from ..math3d.quaternion import q_conj, q_mul, q_normalize

logger = logging.getLogger(__name__)


def _parse_pose_payload(payload: dict) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
    tracked = bool(payload.get("tracked", True))
    position = payload.get("position_m", payload.get("position"))
    quaternion = payload.get("quaternion_wxyz", payload.get("quaternion"))
    if position is None or quaternion is None:
        return None

    p = np.asarray(position, dtype=np.float64).reshape(-1)
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if p.size != 3 or q.size != 4:
        return None
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return None

    return p, q_normalize(q), tracked


def _parse_pose_packet(data: bytes) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_pose_payload(payload)


class _UdpPoseReceiver:
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)

    def recv_latest(self) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break
            parsed = _parse_pose_packet(data)
            if parsed is not None:
                latest = parsed
        return latest

    def close(self) -> None:
        self.sock.close()


class KinectV2PoseProvider(PoseProvider):
    """6DoF provider fed by external Kinect bridge messages over UDP.

    Expected JSON packet schema:
    {
      "tracked": true,
      "position_m": [x, y, z],
      "quaternion_wxyz": [w, x, y, z]
    }

    Native frame is sensor-space style (+z away from camera). Acoustic semantic
    mapping is handled downstream via acoustic frame provider.
    """

    # Sensor-space front/back is opposite to the acoustic front convention.
    default_acoustic_frame_provider_name = "flip-front"

    def __init__(
        self,
        title: str,
        source_world: np.ndarray,
        sr: int,
        smoothing: float = 0.35,
        position_scale: float = 1.0,
        position_smoothing: float = 0.35,
        flip_x: bool = True,
        poll_ms: int = 8,
        bridge_host: str = "127.0.0.1",
        bridge_port: int = 24567,
    ):
        self.title = title
        self.source_world = np.asarray(source_world, dtype=np.float64)
        self.sr = int(sr)
        self.smoothing = float(max(0.01, min(1.0, smoothing)))
        self.position_scale = float(position_scale)
        self.position_smoothing = float(max(0.01, min(1.0, position_smoothing)))
        self.flip_x = bool(flip_x)
        self.poll_s = max(0.001, float(poll_ms) / 1000.0)
        self.bridge_host = str(bridge_host)
        self.bridge_port = int(bridge_port)

        self._receiver = _UdpPoseReceiver(self.bridge_host, self.bridge_port)

        self._closed = False
        self._on_tick = None
        self._status_text = ""
        self._has_tracking = False

        self._q_current = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._p_current = np.zeros(3, dtype=np.float64)
        self._q_prev = None
        self._p_prev = None
        self._q_ref = None
        self._p_ref = None
        self._last_warn_t = 0.0
        self._last_recv_t = 0.0
        self._recv_count = 0

        logger.info(
            "[POSE] provider=kinectv2-bridge (host=%s, port=%s, smoothing=%.2f, "
            "pos_smoothing=%.2f, scale=%.3f, flip_x=%s, poll_ms=%.1f)",
            self.bridge_host,
            self.bridge_port,
            self.smoothing,
            self.position_smoothing,
            self.position_scale,
            self.flip_x,
            self.poll_s * 1000.0,
        )

    def get_pose(self) -> Pose6D:
        return Pose6D(
            position=self._p_current.copy(),
            quaternion=self._q_current.copy(),
        )

    def get_quaternion(self) -> np.ndarray:
        return self._q_current.copy()

    def signfix_enabled(self) -> bool:
        return False

    def get_camera_world_origin(self) -> np.ndarray | None:
        if self._p_ref is None:
            return None
        return (-self._p_ref).astype(np.float64, copy=True)

    def has_tracking(self) -> bool:
        return bool(self._has_tracking)

    def set_status(self, text: str) -> None:
        self._status_text = text

    def _update_pose(self, p_raw: np.ndarray, q_raw: np.ndarray) -> None:
        p = p_raw.astype(np.float64, copy=True)
        if self.flip_x:
            p[0] = -p[0]
        p *= self.position_scale
        q = q_normalize(q_raw.astype(np.float64, copy=False))

        if self._q_ref is None:
            self._q_ref = q.copy()
        if self._p_ref is None:
            self._p_ref = p.copy()

        q_rel = q_mul(q_conj(self._q_ref), q)
        p_rel = p - self._p_ref

        if self._q_prev is not None and float(np.dot(self._q_prev, q_rel)) < 0.0:
            q_rel = -q_rel
        if self._q_prev is not None and self.smoothing < 1.0:
            a = self.smoothing
            q_rel = q_normalize((1.0 - a) * self._q_prev + a * q_rel)

        if self._p_prev is not None and self.position_smoothing < 1.0:
            b = self.position_smoothing
            p_rel = (1.0 - b) * self._p_prev + b * p_rel

        self._q_prev = q_rel
        self._p_prev = p_rel
        self._q_current = q_rel
        self._p_current = p_rel
        self._has_tracking = True

    def _poll_once(self) -> None:
        sample = self._receiver.recv_latest()
        if sample is None:
            now = time.time()
            # Only warn if we have not received any packet recently.
            if (now - self._last_recv_t) > 2.0 and (now - self._last_warn_t) > 2.0:
                logger.info(
                    "[POSE] waiting for Kinect bridge packets on %s:%s",
                    self.bridge_host,
                    self.bridge_port,
                )
                self._last_warn_t = now
            return

        p, q, tracked = sample
        self._last_recv_t = time.time()
        self._recv_count += 1
        if self._recv_count == 1:
            logger.info(
                "[POSE] first Kinect bridge packet received on %s:%s",
                self.bridge_host,
                self.bridge_port,
            )
        if not tracked:
            self._has_tracking = False
            return

        self._update_pose(p, q)

    def run(self, on_tick):
        self._on_tick = on_tick
        while not self._closed:
            self._poll_once()
            if self._on_tick is not None:
                self._on_tick()
            time.sleep(self.poll_s)
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._receiver.close()
        except OSError:
            pass
