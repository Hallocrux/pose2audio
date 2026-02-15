"""Display providers for rendering runtime spatial state."""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..math3d.quaternion import q_rotate_vec
from .pose import Pose6D
from .pose_provider import PoseProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DisplayFrame:
    """Runtime frame data shared by all display providers."""

    pose: Pose6D
    source_world: np.ndarray
    camera_world: Optional[np.ndarray]
    s_head: np.ndarray
    rel_az: float
    rel_el: float
    acoustic_frame_provider: str
    # World-space direction corresponding to acoustic +Z (front).
    head_forward_world: np.ndarray
    selection_idx: int
    dataset_az: float
    dataset_el: float
    underruns: int
    did_signfix: bool


class DisplayProvider:
    """Base display provider interface."""

    def update(self, frame: DisplayFrame) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _status_lines(frame: DisplayFrame) -> list[str]:
    q = frame.pose.quaternion
    p = frame.pose.position
    s = frame.s_head
    src = frame.source_world
    return [
        f"relative az/el = ({frame.rel_az:7.2f} deg, {frame.rel_el:7.2f} deg)",
        (
            f"selected idx    = {frame.selection_idx}  "
            f"(dataset az={frame.dataset_az:.1f} deg, "
            f"el={frame.dataset_el:.1f} deg)"
        ),
        f"underruns       = {frame.underruns}",
        f"acoustic frame  = {frame.acoustic_frame_provider}",
        f"head xyz (m)    = [{p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f}]",
        f"s_head xyz      = [{s[0]: .3f}, {s[1]: .3f}, {s[2]: .3f}]",
        f"source xyz (m)  = [{src[0]: .3f}, {src[1]: .3f}, {src[2]: .3f}]",
        (
            f"q=[w,x,y,z]     = [{q[0]: .4f}, {q[1]: .4f}, "
            f"{q[2]: .4f}, {q[3]: .4f}]  signfix={frame.did_signfix}"
        ),
    ]


class _CliStatsSink:
    def __init__(self, mode: str):
        self.mode = "live" if mode == "live" else "scroll"
        self._is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._live_enabled = self.mode == "live" and self._is_tty
        self._line_count = 0

    def emit(self, lines: list[str], scroll_line: str) -> None:
        if not self._live_enabled:
            logger.info(scroll_line)
            return

        out = sys.stderr
        if self._line_count > 0:
            out.write(f"\x1b[{self._line_count}F")

        max_lines = max(self._line_count, len(lines))
        for i in range(max_lines):
            line = lines[i] if i < len(lines) else ""
            out.write("\x1b[2K")
            out.write(line)
            out.write("\n")
        out.flush()
        self._line_count = len(lines)


class TuiDisplayProvider(DisplayProvider):
    """Terminal + pose UI text display provider."""

    def __init__(self, pose_provider: PoseProvider, cli_output: str = "live"):
        self.pose_provider = pose_provider
        self.cli_sink = _CliStatsSink(cli_output)

    def update(self, frame: DisplayFrame) -> None:
        p = frame.pose.position
        s = frame.s_head
        src = frame.source_world

        self.pose_provider.set_status("\n".join(_status_lines(frame)))

        self.cli_sink.emit(
            lines=[
                "VisualSound Live Scene",
                f"head xyz      = ({p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f})",
                f"source xyz    = ({src[0]: .3f}, {src[1]: .3f}, {src[2]: .3f})",
                f"s_head xyz    = ({s[0]: .3f}, {s[1]: .3f}, {s[2]: .3f})",
                f"hrtf az/el    = ({frame.rel_az: .2f}, {frame.rel_el: .2f}) deg",
                f"selected idx  = {frame.selection_idx}",
                f"underruns     = {frame.underruns}",
            ],
            scroll_line=(
                "[POSE] head_xyz=(%.3f, %.3f, %.3f) source_xyz=(%.3f, %.3f, %.3f) "
                "hrtf_az_el=(%.2f, %.2f) idx=%s"
                % (
                    p[0],
                    p[1],
                    p[2],
                    src[0],
                    src[1],
                    src[2],
                    frame.rel_az,
                    frame.rel_el,
                    frame.selection_idx,
                )
            ),
        )


class Matplotlib3DAnimationDisplayProvider(DisplayProvider):
    """Simple 3D animated scene display using matplotlib."""

    def __init__(self, title: str = "VisualSound 3D Scene"):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "display-provider=3d requires matplotlib. Install with: uv add matplotlib"
            ) from exc

        self.plt = plt
        self.plt.ion()
        self.fig = self.plt.figure(title)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("x (right)")
        self.ax.set_ylabel("y (up)")
        self.ax.set_zlabel("z (forward)")

        self.cam_pt = self.ax.scatter([0.0], [0.0], [0.0], c="tab:blue", s=60, label="camera")
        self.head_pt = self.ax.scatter([0.0], [0.0], [0.0], c="tab:green", s=60, label="head")
        self.src_pt = self.ax.scatter([0.0], [0.0], [1.0], c="tab:red", s=60, label="source")
        (self.hs_line,) = self.ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c="gray", lw=1.2)
        (self.head_fwd_line,) = self.ax.plot(
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.3], c="tab:orange", lw=2.0, label="head forward"
        )
        (self.head_right_line,) = self.ax.plot(
            [0.0, 0.3], [0.0, 0.0], [0.0, 0.0], c="tab:red", lw=1.6, label="head right"
        )
        (self.head_up_line,) = self.ax.plot(
            [0.0, 0.0], [0.0, 0.3], [0.0, 0.0], c="tab:cyan", lw=1.6, label="head up"
        )
        self.ax.legend(loc="upper left")
        self._enabled = True
        self._axis_span = 2.0

    def _set_point(self, artist, x: float, y: float, z: float) -> None:
        artist._offsets3d = ([x], [y], [z])  # type: ignore[attr-defined]

    def _set_line(self, line_artist, p0: np.ndarray, p1: np.ndarray) -> None:
        line_artist.set_data_3d(
            [float(p0[0]), float(p1[0])],
            [float(p0[1]), float(p1[1])],
            [float(p0[2]), float(p1[2])],
        )

    def update(self, frame: DisplayFrame) -> None:
        if not self._enabled:
            return
        if not self.plt.fignum_exists(self.fig.number):
            self._enabled = False
            return

        p = np.asarray(frame.pose.position, dtype=np.float64).reshape(3)
        s = np.asarray(frame.source_world, dtype=np.float64).reshape(3)
        if frame.camera_world is None:
            c = np.zeros(3, dtype=np.float64)
        else:
            c = np.asarray(frame.camera_world, dtype=np.float64).reshape(3)

        self._set_point(self.cam_pt, c[0], c[1], c[2])
        self._set_point(self.head_pt, p[0], p[1], p[2])
        self._set_point(self.src_pt, s[0], s[1], s[2])
        self.hs_line.set_data_3d([p[0], s[0]], [p[1], s[1]], [p[2], s[2]])

        q = np.asarray(frame.pose.quaternion, dtype=np.float64).reshape(4)
        if not np.isfinite(q).all():
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        fwd = np.asarray(frame.head_forward_world, dtype=np.float64).reshape(3)
        fwd_n = float(np.linalg.norm(fwd))
        if fwd_n < 1e-12:
            fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            fwd = fwd / fwd_n
        right = q_rotate_vec(q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        up = q_rotate_vec(q, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis_len = max(0.08, self._axis_span * 0.18)
        self._set_line(self.head_fwd_line, p, p + fwd * axis_len)
        self._set_line(self.head_right_line, p, p + right * axis_len * 0.75)
        self._set_line(self.head_up_line, p, p + up * axis_len * 0.75)

        pts = np.vstack([c, p, s])
        center = np.mean(pts, axis=0)
        span = float(np.max(np.ptp(pts, axis=0)))
        self._axis_span = max(0.8, self._axis_span * 0.9 + max(1.2, span * 1.4) * 0.1)
        half = self._axis_span * 0.5
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_title(
            f"az/el=({frame.rel_az:.1f},{frame.rel_el:.1f})  "
            f"idx={frame.selection_idx}  underruns={frame.underruns}"
        )

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self) -> None:
        if not getattr(self, "_enabled", False):
            return
        self._enabled = False
        try:
            self.plt.close(self.fig)
        except Exception:
            pass


class _Open3DOrbitCamera:
    """Custom orbit camera controls for Open3D VisualizerWithKeyCallback."""

    GLFW_RELEASE = 0
    GLFW_PRESS = 1
    GLFW_REPEAT = 2

    GLFW_MOD_SHIFT = 0x0001
    GLFW_MOD_CONTROL = 0x0002

    GLFW_MOUSE_BUTTON_LEFT = 0

    GLFW_KEY_F = 70
    GLFW_KEY_R = 82
    GLFW_KEY_LEFT_SHIFT = 340
    GLFW_KEY_LEFT_CONTROL = 341
    GLFW_KEY_RIGHT_SHIFT = 344
    GLFW_KEY_RIGHT_CONTROL = 345

    def __init__(self):
        self.target = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        self.radius = 2.2
        self.yaw = 0.0
        self.pitch = 0.12

        self.ctrl_down = False
        self.shift_down = False
        self.left_dragging = False
        self.last_mouse_xy: tuple[float, float] | None = None

        self.rotate_speed = 0.006
        self.pan_speed = 0.12
        self.zoom_speed = 0.14

        self.follow_scene = True

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        sy = math.sin(self.yaw)
        cy = math.cos(self.yaw)
        offset = np.array([cp * sy, sp, cp * cy], dtype=np.float64)
        cam_pos = self.target + self.radius * offset

        front = self.target - cam_pos
        front /= float(np.linalg.norm(front)) + 1e-12

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(front, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= rn
        cam_up = np.cross(right, front)
        cam_up /= float(np.linalg.norm(cam_up)) + 1e-12
        return front, right, cam_up

    def apply(self, view_control) -> None:
        front, _, cam_up = self._camera_basis()
        view_control.set_front(front.tolist())
        view_control.set_up(cam_up.tolist())
        view_control.set_lookat(self.target.tolist())
        zoom = float(np.clip(1.15 / max(self.radius, 1e-6), 0.12, 1.0))
        view_control.set_zoom(zoom)

    def update_scene(self, center: np.ndarray, span: float) -> None:
        if not self.follow_scene:
            return
        a = 0.14
        self.target = (1.0 - a) * self.target + a * center
        desired_radius = max(0.8, span * 2.2)
        self.radius = (1.0 - a) * self.radius + a * desired_radius

    def on_key_action(self, key: int, action: int, mods: int) -> bool:
        is_down = action != self.GLFW_RELEASE
        if key in (self.GLFW_KEY_LEFT_CONTROL, self.GLFW_KEY_RIGHT_CONTROL):
            self.ctrl_down = is_down
            return True
        if key in (self.GLFW_KEY_LEFT_SHIFT, self.GLFW_KEY_RIGHT_SHIFT):
            self.shift_down = is_down
            return True
        if action == self.GLFW_PRESS and key in (self.GLFW_KEY_F, self.GLFW_KEY_R):
            self.follow_scene = True
            return True
        if action == self.GLFW_RELEASE:
            self.ctrl_down = bool(mods & self.GLFW_MOD_CONTROL)
            self.shift_down = bool(mods & self.GLFW_MOD_SHIFT)
            return True
        return False

    def on_mouse_button(self, button: int, action: int, mods: int) -> bool:
        self.ctrl_down = bool(mods & self.GLFW_MOD_CONTROL) or self.ctrl_down
        self.shift_down = bool(mods & self.GLFW_MOD_SHIFT) or self.shift_down
        if button != self.GLFW_MOUSE_BUTTON_LEFT:
            return False
        if action == self.GLFW_PRESS:
            self.left_dragging = True
            return True
        if action == self.GLFW_RELEASE:
            self.left_dragging = False
            self.last_mouse_xy = None
            return True
        return False

    def on_mouse_move(self, x: float, y: float) -> bool:
        current = (float(x), float(y))
        if self.last_mouse_xy is None:
            self.last_mouse_xy = current
            return False
        if not self.left_dragging:
            self.last_mouse_xy = current
            return False

        dx = current[0] - self.last_mouse_xy[0]
        dy = current[1] - self.last_mouse_xy[1]
        self.last_mouse_xy = current
        self.follow_scene = False
        self.yaw -= dx * self.rotate_speed
        self.pitch = float(np.clip(self.pitch - dy * self.rotate_speed, -1.45, 1.45))
        return True

    def on_mouse_scroll(self, y_offset: float) -> bool:
        dy = float(y_offset)
        if abs(dy) < 1e-12:
            return False
        front, right, cam_up = self._camera_basis()
        self.follow_scene = False
        if self.ctrl_down:
            self.radius *= math.exp(-dy * self.zoom_speed)
            self.radius = float(np.clip(self.radius, 0.15, 80.0))
            return True

        pan_step = self.radius * self.pan_speed * dy * 0.05
        if self.shift_down:
            self.target += right * pan_step
        else:
            self.target += cam_up * pan_step
        return True


class Open3DAnimationDisplayProvider(DisplayProvider):
    """High-performance realtime 3D display using Open3D Visualizer."""

    def __init__(
        self,
        pose_provider: PoseProvider,
        title: str = "VisualSound Open3D Scene",
    ):
        try:
            import open3d as o3d
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "display-provider=open3d requires open3d. Install with: uv add open3d"
            ) from exc

        self.pose_provider = pose_provider
        self.o3d = o3d
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(window_name=title, width=1120, height=760):
            raise RuntimeError("failed to create Open3D visualizer window")

        self._enabled = True
        self._axis_span = 2.0
        self._camera = _Open3DOrbitCamera()
        self._key_callbacks: list = []

        # camera/head/source + head orientation endpoints
        self._points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.2],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
        self._point_colors = np.array(
            [
                [0.2, 0.5, 1.0],  # camera (blue)
                [0.25, 0.9, 0.35],  # head (green)
                [1.0, 0.3, 0.2],  # source (red)
                [1.0, 0.65, 0.1],  # head forward endpoint
                [1.0, 0.25, 0.25],  # head right endpoint
                [0.2, 0.95, 1.0],  # head up endpoint
            ],
            dtype=np.float64,
        )
        self._line_indices = np.array(
            [
                [0, 1],  # camera->head
                [1, 2],  # head->source
                [1, 3],  # head forward
                [1, 4],  # head right
                [1, 5],  # head up
            ],
            dtype=np.int32,
        )
        self._line_colors = np.array(
            [
                [0.7, 0.7, 0.7],
                [0.9, 0.55, 0.55],
                [1.0, 0.65, 0.1],
                [1.0, 0.25, 0.25],
                [0.2, 0.95, 1.0],
            ],
            dtype=np.float64,
        )

        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self._points.copy())
        self.point_cloud.colors = o3d.utility.Vector3dVector(self._point_colors)
        self._pcd_points = np.asarray(self.point_cloud.points)

        self.line_set = o3d.geometry.LineSet()
        self.line_set.points = o3d.utility.Vector3dVector(self._points.copy())
        self.line_set.lines = o3d.utility.Vector2iVector(self._line_indices)
        self.line_set.colors = o3d.utility.Vector3dVector(self._line_colors)
        self._line_points = np.asarray(self.line_set.points)

        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.22)
        self.ground_grid = self._make_ground_grid(size=2.0, step=0.25)

        self.vis.add_geometry(self.axes)
        self.vis.add_geometry(self.ground_grid)
        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.line_set)

        ro = self.vis.get_render_option()
        ro.point_size = 14.0
        if hasattr(ro, "line_width"):
            ro.line_width = 2.0
        ro.background_color = np.array([0.03, 0.04, 0.06], dtype=np.float64)
        if hasattr(ro, "show_coordinate_frame"):
            ro.show_coordinate_frame = False
        if hasattr(ro, "light_on"):
            ro.light_on = True

        self._register_interaction_callbacks()
        self._apply_camera()

    def _make_ground_grid(self, size: float = 2.0, step: float = 0.25):
        n = int(max(1, round(size / step)))
        xs = np.arange(-n, n + 1, dtype=np.float64) * step
        points = []
        lines = []
        colors = []
        idx = 0
        for x in xs:
            points.append([x, 0.0, -size])
            points.append([x, 0.0, size])
            lines.append([idx, idx + 1])
            colors.append([0.25, 0.25, 0.30])
            idx += 2
        for z in xs:
            points.append([-size, 0.0, z])
            points.append([size, 0.0, z])
            lines.append([idx, idx + 1])
            colors.append([0.25, 0.25, 0.30])
            idx += 2

        grid = self.o3d.geometry.LineSet()
        grid.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        grid.lines = self.o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        grid.colors = self.o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        return grid

    def _register_interaction_callbacks(self) -> None:
        def _on_mouse_button(vis, button, action, mods):
            changed = self._camera.on_mouse_button(int(button), int(action), int(mods))
            if changed:
                self._apply_camera()
            return False

        def _on_mouse_move(vis, x, y):
            changed = self._camera.on_mouse_move(float(x), float(y))
            if changed:
                self._apply_camera()
            return False

        def _on_mouse_scroll(vis, x_off, y_off):
            changed = self._camera.on_mouse_scroll(float(y_off))
            if changed:
                self._apply_camera()
            return False

        self.vis.register_mouse_button_callback(_on_mouse_button)
        self.vis.register_mouse_move_callback(_on_mouse_move)
        self.vis.register_mouse_scroll_callback(_on_mouse_scroll)

        def _bind_key(key: int):
            def _cb(vis, action, mods):
                changed = self._camera.on_key_action(int(key), int(action), int(mods))
                if changed:
                    self._apply_camera()
                return False

            self._key_callbacks.append(_cb)
            self.vis.register_key_action_callback(int(key), _cb)

        _bind_key(_Open3DOrbitCamera.GLFW_KEY_LEFT_SHIFT)
        _bind_key(_Open3DOrbitCamera.GLFW_KEY_RIGHT_SHIFT)
        _bind_key(_Open3DOrbitCamera.GLFW_KEY_LEFT_CONTROL)
        _bind_key(_Open3DOrbitCamera.GLFW_KEY_RIGHT_CONTROL)
        _bind_key(_Open3DOrbitCamera.GLFW_KEY_F)
        _bind_key(_Open3DOrbitCamera.GLFW_KEY_R)

    def _apply_camera(self) -> None:
        vc = self.vis.get_view_control()
        self._camera.apply(vc)

    def update(self, frame: DisplayFrame) -> None:
        if not self._enabled:
            return
        if not self.vis.poll_events():
            self._enabled = False
            return

        p = np.asarray(frame.pose.position, dtype=np.float64).reshape(3)
        s = np.asarray(frame.source_world, dtype=np.float64).reshape(3)
        c = (
            np.zeros(3, dtype=np.float64)
            if frame.camera_world is None
            else np.asarray(frame.camera_world, dtype=np.float64).reshape(3)
        )

        self._points[0] = c
        self._points[1] = p
        self._points[2] = s
        q = np.asarray(frame.pose.quaternion, dtype=np.float64).reshape(4)
        if not np.isfinite(q).all():
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        fwd = np.asarray(frame.head_forward_world, dtype=np.float64).reshape(3)
        fwd_n = float(np.linalg.norm(fwd))
        if fwd_n < 1e-12:
            fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            fwd = fwd / fwd_n
        right = q_rotate_vec(q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        up = q_rotate_vec(q, np.array([0.0, 1.0, 0.0], dtype=np.float64))

        pts = np.vstack([c, p, s])
        center = np.mean(pts, axis=0)
        span = float(np.max(np.ptp(pts, axis=0)))
        self._axis_span = max(0.8, self._axis_span * 0.9 + max(1.2, span * 1.4) * 0.1)
        axis_len = max(0.08, self._axis_span * 0.18)
        self._points[3] = p + fwd * axis_len
        self._points[4] = p + right * axis_len * 0.75
        self._points[5] = p + up * axis_len * 0.75

        self._pcd_points[:] = self._points
        self._line_points[:] = self._points
        self.vis.update_geometry(self.point_cloud)
        self.vis.update_geometry(self.line_set)

        self._camera.update_scene(center=center, span=self._axis_span)
        self._apply_camera()

        self.pose_provider.set_status(
            "\n".join(
                _status_lines(frame)
                + [
                    "Open3D controls: L-drag=orbit  Wheel=vertical pan  Shift+Wheel=horizontal pan  Ctrl+Wheel=zoom",
                    "Open3D key: [F] follow scene center",
                ]
            )
        )
        self.vis.update_renderer()

    def close(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        try:
            self.vis.destroy_window()
        except Exception:
            pass
