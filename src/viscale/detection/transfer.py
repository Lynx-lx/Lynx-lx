"""Copy compatible tensors from official YOLOv5s COCO weights into YOLOv5sLite."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from viscale.detection.yolov5s_lite import YOLOv5sLite


class _YoloStub(nn.Module):
    """Placeholder so official yolov5s.pt (pickled models.yolo.Model) can unpickle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setstate__(self, state) -> None:
        nn.Module.__init__(self)
        if isinstance(state, dict):
            try:
                super().__setstate__(state)
            except Exception:
                self.__dict__.update(state)
        else:
            super().__setstate__(state)


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("models") or module.startswith("utils"):
            return _YoloStub
        return super().find_class(module, name)


class _StubMod(ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        setattr(self, name, _YoloStub)
        return _YoloStub


_STUB_NAMES = (
    "models",
    "models.yolo",
    "models.common",
    "models.experimental",
    "utils",
    "utils.general",
    "utils.torch_utils",
    "utils.downloads",
)


def _install_yolo_stubs() -> None:
    import sys

    for name in _STUB_NAMES:
        if name not in sys.modules or not isinstance(sys.modules[name], _StubMod):
            sys.modules[name] = _StubMod(name)


def _remove_yolo_stubs() -> None:
    import sys

    for name in _STUB_NAMES:
        mod = sys.modules.get(name)
        if isinstance(mod, _StubMod):
            del sys.modules[name]


def torch_load(path: Path, map_location="cpu"):
    _install_yolo_stubs()
    try:
        try:
            return torch.load(str(path), map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location=map_location)
        except ModuleNotFoundError:
            with open(path, "rb") as fh:
                data = io.BytesIO(fh.read())
            return _StubUnpickler(data).load()
    finally:
        _remove_yolo_stubs()


def unwrap_state_dict(raw) -> dict:
    obj = raw
    if isinstance(obj, dict):
        if "ema" in obj and obj["ema"] is not None:
            obj = obj["ema"]
        elif "model" in obj:
            obj = obj["model"]
        elif "state_dict" in obj:
            obj = obj["state_dict"]
    if hasattr(obj, "float"):
        try:
            obj = obj.float()
        except Exception:
            pass
    tensors = _collect_tensors(obj)
    if tensors:
        out = {}
        for k, v in tensors.items():
            key = str(k)
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("model."):
                key = key[len("model.") :]
            out[key] = v.detach().cpu().clone()
        return out
    raise TypeError("checkpoint has no tensors")


def _collect_tensors(obj, prefix: str = "") -> dict:
    out: dict = {}
    if torch.is_tensor(obj):
        if prefix:
            out[prefix] = obj
        return out
    if isinstance(obj, dict):
        for key, val in obj.items():
            if not isinstance(key, str):
                continue
            if torch.is_tensor(val):
                name = f"{prefix}.{key}" if prefix else key
                out[name] = val
            elif key in ("state_dict", "model", "ema") or isinstance(val, (dict, nn.Module)):
                child = prefix if key in ("state_dict",) else (f"{prefix}.{key}" if prefix else key)
                if key == "state_dict":
                    child = prefix
                out.update(_collect_tensors(val, child))
        return out
    if isinstance(obj, nn.Module):
        try:
            sd = nn.Module.state_dict(obj)
            for key, val in sd.items():
                if torch.is_tensor(val):
                    name = f"{prefix}.{key}" if prefix else key
                    out[name] = val
            if out:
                return out
        except Exception:
            pass
        out.update(_collect_tensors(vars(obj), prefix))
        return out
    return out


def is_detect_head_key(name: str) -> bool:
    n = name.replace("\\", "/")
    if n.startswith("detect.") or ".detect." in n:
        return True
    if n.startswith("model.24.") or n.startswith("24."):
        return True
    return False


def transfer_backbone(src: dict, dst: dict) -> tuple[dict, int]:
    """Shape-greedy copy of non-head tensors. Detect head stays as in dst."""
    src_backbone = [(k, v) for k, v in src.items() if not is_detect_head_key(k)]
    merged = {k: v.clone() for k, v in dst.items()}
    used: set[str] = set()
    n_ok = 0
    for dk, dv in dst.items():
        if is_detect_head_key(dk):
            continue
        hit = None
        for sk, sv in src_backbone:
            if sk in used:
                continue
            if tuple(sv.shape) == tuple(dv.shape):
                hit = sk
                break
        if hit is None:
            continue
        merged[dk] = src[hit].clone()
        used.add(hit)
        n_ok += 1
    return merged, n_ok


def load_coco_backbone(model: YOLOv5sLite, coco_path: Path) -> int:
    """Initialize non-head weights from official yolov5s.pt. Returns copied tensor count."""
    if not coco_path.is_file() or coco_path.stat().st_size <= 0:
        return 0
    raw = torch_load(coco_path, "cpu")
    src = unwrap_state_dict(raw)
    dst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    merged, n_ok = transfer_backbone(src, dst)
    model.load_state_dict(merged, strict=True)
    return n_ok
