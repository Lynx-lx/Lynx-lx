"""由相机内参得到粗糙的地面分辨率估计（需使用者提供真实标定）。"""

from __future__ import annotations

from viscale.io.camera import CameraConfig


def meters_per_pixel_at_depth(camera: CameraConfig | None, depth_m: float) -> float | None:
    """
    小孔成像近似：mpp ≈ depth / fx （水平方向）。

    模板内参不得用于真实测量；camera 为 None 或 is_template 时返回 None。
    """
    if camera is None or camera.is_template:
        return None
    if camera.fx <= 1e-6 or depth_m <= 0:
        return None
    return float(depth_m / camera.fx)
