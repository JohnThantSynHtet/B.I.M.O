"""Display recorded poses in a separate model; GUI input cannot change results."""

from pathlib import Path
import time

import mujoco
from sesame_ml.model import load_model, make_data


def set_frame(model, data, row: dict) -> None:
    data.qpos[:] = [row[f"qpos_{i}"] for i in range(model.nq)]
    mujoco.mj_forward(model, data)


def camera(model, data):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = data.xpos[model.body("base").id]
    cam.distance, cam.azimuth, cam.elevation = 0.45, 135, -25
    return cam


def render_last(rows: list[dict], path: Path) -> None:
    from PIL import Image
    model = load_model()
    data = make_data(model)
    set_frame(model, data, rows[-1])
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera(model, data))
        Image.fromarray(renderer.render()).save(path)


def replay(rows: list[dict]) -> None:
    import mujoco.viewer
    model = load_model()
    data = make_data(model)
    set_frame(model, data, rows[0])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = camera(model, data).lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.45, 135, -25
        start = time.monotonic()
        # Fixed subsampling affects display only. Include the terminal pose.
        for index in sorted(set(range(0, len(rows), 10)) | {len(rows) - 1}):
            if not viewer.is_running():
                break
            row = rows[index]
            time.sleep(max(0, start + row["time_s"] - time.monotonic()))
            with viewer.lock():
                set_frame(model, data, row)
            viewer.sync()
