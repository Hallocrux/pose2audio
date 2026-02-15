"""ToyCV-style Tk sliders for head pose."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from ..control.pose import Pose6D
from ..control.pose_provider import PoseProvider
from ..math3d.quaternion import euler_yaw_pitch_roll_to_q


class ToyCvTkPoseProvider(PoseProvider):
    """Debug provider in acoustic axes (x right, y up, z front)."""

    # Toy sliders are authored directly in app/acoustic axes.
    default_acoustic_frame_provider_name = "identity"

    def __init__(self, title: str, source_world: np.ndarray, sr: int):
        self.root = tk.Tk()
        self.root.title(title)

        self._var_yaw = tk.DoubleVar(value=0.0)
        self._var_pitch = tk.DoubleVar(value=0.0)
        self._var_roll = tk.DoubleVar(value=0.0)
        self._var_x = tk.DoubleVar(value=0.0)
        self._var_y = tk.DoubleVar(value=0.0)
        self._var_z = tk.DoubleVar(value=0.0)
        self._var_signfix = tk.IntVar(value=1)

        self._build_ui(source_world, sr)

        self._on_tick = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _build_ui(self, source_world: np.ndarray, sr: int) -> None:
        def add_slider(label: str, var: tk.DoubleVar, lo: int, hi: int) -> None:
            tk.Label(self.root, text=label).pack(anchor="w", padx=10, pady=2)
            tk.Scale(
                self.root,
                from_=lo,
                to=hi,
                orient="horizontal",
                resolution=1,
                length=520,
                variable=var,
            ).pack(padx=10, pady=2)

        def add_pos_slider(label: str, var: tk.DoubleVar, lo: float, hi: float) -> None:
            tk.Label(self.root, text=label).pack(anchor="w", padx=10, pady=2)
            tk.Scale(
                self.root,
                from_=lo,
                to=hi,
                orient="horizontal",
                resolution=0.01,
                length=520,
                variable=var,
            ).pack(padx=10, pady=2)

        add_slider("Head yaw (deg)   [-180..180]", self._var_yaw, -180, 180)
        add_slider("Head pitch (deg) [-89..89]", self._var_pitch, -89, 89)
        add_slider("Head roll (deg)  [-180..180]", self._var_roll, -180, 180)
        add_pos_slider("Head x (m)   [-1.5..1.5]", self._var_x, -1.5, 1.5)
        add_pos_slider("Head y (m)   [-1.5..1.5]", self._var_y, -1.5, 1.5)
        add_pos_slider("Head z (m)   [-1.5..1.5]", self._var_z, -1.5, 1.5)
        tk.Checkbutton(
            self.root,
            text="Sign-fix quaternion continuity (dot<0 => q=-q)",
            variable=self._var_signfix,
        ).pack(anchor="w", padx=10, pady=6)

        info = tk.Label(
            self.root,
            text=(
                "World source fixed at "
                f"[{source_world[0]:.2f}, {source_world[1]:.2f}, {source_world[2]:.2f}] m  |  SOFA sr={sr}"
            ),
        )
        info.pack(padx=10, pady=4)

        self._stats = tk.Label(self.root, text="", justify="left", font=("Consolas", 10))
        self._stats.pack(padx=10, pady=8)

    def _handle_close(self) -> None:
        self._closed = True
        self.root.destroy()

    def get_quaternion(self) -> np.ndarray:
        yaw = float(self._var_yaw.get())
        pitch = float(self._var_pitch.get())
        roll = float(self._var_roll.get())
        return euler_yaw_pitch_roll_to_q(yaw, pitch, roll)

    def get_pose(self) -> Pose6D:
        return Pose6D(
            position=np.array(
                [
                    float(self._var_x.get()),
                    float(self._var_y.get()),
                    float(self._var_z.get()),
                ],
                dtype=np.float64,
            ),
            quaternion=self.get_quaternion(),
        )

    def signfix_enabled(self) -> bool:
        return int(self._var_signfix.get()) == 1

    def set_status(self, text: str) -> None:
        if not self._closed:
            self._stats.config(text=text)

    def run(self, on_tick):
        self._on_tick = on_tick
        self.root.after(33, self._tick)
        self.root.mainloop()

    def _tick(self) -> None:
        if self._closed:
            return
        if self._on_tick is not None:
            self._on_tick()
        self.root.after(33, self._tick)

    def close(self) -> None:
        if not self._closed:
            self._handle_close()
