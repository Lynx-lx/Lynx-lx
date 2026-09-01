"""读取相机 YAML：兼容模板字段；文件缺失时返回提示，不抛崩溃。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_CAMERA = ROOT / "config" / "camera_sensor" / "camera.yaml"
DEFAULT_TEMPLATE = ROOT / "config" / "camera_sensor" / "camera_template.yaml"


@dataclass
class CameraConfig:
    name: str
    width: int
    height: int
    fps: float
    K: list[list[float]]
    D: list[float]
    roi_enabled: bool
    roi: tuple[int, int, int, int]
    T_cam_to_world: list[list[float]]
    is_template: bool
    source_path: Path
    warnings: list[str] = field(default_factory=list)

    @property
    def fx(self) -> float:
        return float(self.K[0][0])


def _as_float_mat(value, rows: int, cols: int, label: str, warnings: list[str]) -> list[list[float]]:
    try:
        mat = [[float(x) for x in row] for row in value]
        if len(mat) != rows or any(len(r) != cols for r in mat):
            raise ValueError("shape")
        return mat
    except (TypeError, ValueError):
        warnings.append(f"{label} 格式无效，已使用单位阵/默认占位")
        if rows == 3 and cols == 3:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ident = [[0.0] * cols for _ in range(rows)]
        for i in range(min(rows, cols)):
            ident[i][i] = 1.0
        return ident


def load_camera_config(path: str | Path | None = None) -> tuple[CameraConfig | None, str]:
    """
    加载相机配置。

    默认查找 ``config/camera_sensor/camera.yaml``（本地真实标定，不入库）。
    文件不存在时不抛异常，返回 (None, 友好中文说明)。
    """
    target = Path(path) if path else DEFAULT_LOCAL_CAMERA
    if not target.is_file():
        hint = (
            f"未找到相机配置文件: {target}\n"
            f"这是正常情况：公开仓库不包含真实标定。\n"
            f"请复制模板后自行填写：\n"
            f"  {DEFAULT_TEMPLATE}\n"
            f"  -> {DEFAULT_LOCAL_CAMERA}\n"
            "当前将跳过基于内参的尺度换算，检测与风险模块仍可运行。"
        )
        return None, hint

    try:
        import yaml
    except ImportError:
        return None, "未安装 PyYAML，请执行 pip install -r requirements.txt 后再读取相机配置。"

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        return None, f"无法读取相机配置 {target}: {exc}"
    except yaml.YAMLError as exc:
        return None, f"相机 YAML 解析失败 {target}: {exc}\n请对照 camera_template.yaml 检查字段。"

    if not isinstance(raw, dict):
        return None, f"相机配置根节点必须是字典: {target}"

    warnings: list[str] = []
    meta = raw.get("meta") or {}
    cam = raw.get("camera") or {}
    intra = raw.get("intrinsics") or {}
    roi = raw.get("roi") or {}
    extra = raw.get("extrinsics") or {}

    is_template = bool(meta.get("is_template", False))
    if is_template:
        warnings.append("当前文件标记为模板（is_template: true），内参仅为占位，不可当作现场标定。")

    K = _as_float_mat(intra.get("K"), 3, 3, "intrinsics.K", warnings)
    try:
        D = [float(x) for x in (intra.get("D") or [0, 0, 0, 0, 0])]
    except (TypeError, ValueError):
        D = [0.0, 0.0, 0.0, 0.0, 0.0]
        warnings.append("intrinsics.D 无效，已置零")

    T = extra.get("T_cam_to_world")
    Tm = _as_float_mat(T, 4, 4, "extrinsics.T_cam_to_world", warnings) if T is not None else [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    cfg = CameraConfig(
        name=str(cam.get("name", "unnamed")),
        width=int(cam.get("width", 0) or 0),
        height=int(cam.get("height", 0) or 0),
        fps=float(cam.get("fps", 0) or 0),
        K=K,
        D=D,
        roi_enabled=bool(roi.get("enabled", False)),
        roi=(
            int(roi.get("x", 0) or 0),
            int(roi.get("y", 0) or 0),
            int(roi.get("width", 0) or 0),
            int(roi.get("height", 0) or 0),
        ),
        T_cam_to_world=Tm,
        is_template=is_template,
        source_path=target,
        warnings=warnings,
    )
    msg = f"已加载相机配置: {target} （{cfg.name}）"
    if warnings:
        msg += "\n" + "\n".join(warnings)
    return cfg, msg
