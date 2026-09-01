# Lynx-lx · 电力安防视觉尺度测量图像检测

面向变电站 / 线路巡检等**电力安防**场景的视觉检测与风险评估工程：在图像上做目标检测，结合尺度信息做隐患评估，并配套万级标注数据的自动化校验与划分。

---

## 仓库声明（请先阅读）

**本仓库为后期独立复现实现，并非原始大创竞赛源码，原始学校项目不予开源。**

- 原始竞赛 / 校内项目源码已不可用，本仓库全部代码为从零重写的公开复现，可推送到个人 GitHub。
- 仓库内不含学校内部数据、竞赛提交包、未公开标注集或原始权重。
- 检测结构参考公开文献中的 YOLOv5s、ECA / SE / CBAM 等思路，属于独立实现，不拷贝任何学校仓库或 Ultralytics 发行源码。
- 若与简历中「视觉尺度测量下的图像检测系统」对应：简历描述的是本复现工程的能力范围，而不是已开源的校内原始仓库。

---

## 与简历描述的对应关系

| 简历要点 | 本仓库实现 |
|----------|------------|
| 电力安防场景图像检测 | `src/viscale/detection/` 轻量化改进 YOLOv5s（P2 小目标分支 + 注意力） |
| 视觉尺度测量 | 风险评估支持 `meters_per_pixel`；测量包目录已预留 `src/viscale/measurement/` |
| 自定义违规评估、ELC 阈值迭代、风险分级 | `src/viscale/risk/` |
| 万级图像标注与数据集自动化校验 | `dataset/` 校验、划分、统计脚本（流式处理，不入库真实万级数据） |
| 本地演示 | 根目录 `app.py`（Gradio，上传图片 → 画框 → 风险等级） |

当前**不附带训练好的权重**；演示可随机初始化打通流程，权重路径为预留配置。

---

## 环境

Python 3.10+（开发时使用 3.12）。建议虚拟环境：

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU 可选：按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版 `torch`，再安装其余依赖。

---

## 快速演示

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，上传本地图片后点击「检测并评估风险」。

```bash
python app.py --weights models/checkpoints/yolov5s_lite.pt --device cpu --attn eca
```

环境变量：`VISCALE_WEIGHTS`、`VISCALE_DEVICE`、`VISCALE_ATTN`、`VISCALE_IMGSZ`。权重文件不存在时自动跳过加载。

命令行推理：

```bash
python scripts/demo_yolov5s_lite.py --image your.jpg --weights models/checkpoints/yolov5s_lite.pt
```

---

## 检测模块

- 包路径：`src/viscale/detection/`
- YOLOv5s 宽度 / 深度；FPN+PAN；**P2/stride-4** 小目标头；C3 后接 **ECA**（可选 `se` / `cbam` / `none`）
- 默认类别：`person, helmet, insulator, bird_nest, smoke, fire, vehicle, foreign_object`

运行时请将 `src` 加入 `PYTHONPATH`，或使用已设置路径的 `app.py` / `scripts/`。

---

## 风险评估

- 包路径：`src/viscale/risk/`
- 可注册违规函数（未戴帽、鸟巢靠近绝缘子、烟火、异物、车辆等）
- **ELC（期望损失准则）** 迭代风险分数分界
- 四级：低 / 一般 / 较大 / 重大

```python
from viscale.risk import DetectionRecord, RiskAssessor

report = RiskAssessor().assess(detections, meters_per_pixel=0.02, image_wh=(640, 640))
print(report.label_zh, report.score)
```

---

## 数据工程（YOLO 标签）

默认布局：`<root>/images/` 与 `<root>/labels/`（同名 `.txt`，`class_id xc yc w h` 归一化）。脚本只处理逻辑，**本仓库不提供万级真实图片**。

```bash
python dataset/validate_annotations.py --root data --max-report 30
python dataset/split_dataset.py --root data --ratios 0.7,0.2,0.1 --out-dir dataset/splits
python dataset/analyze_dataset.py --root data
```

---

## 目录结构

```
.
├── app.py                   # Gradio 本地演示
├── requirements.txt
├── LICENSE                  # MIT
├── configs/                 # 配置占位
├── data/                    # 本地数据（大文件不入库）
├── dataset/                 # 标注校验 / 划分 / 统计
├── models/                  # 权重目录（*.pt 不入库）
├── outputs/                 # 运行输出（不入库）
├── scripts/                 # 命令行 demo
├── src/viscale/
│   ├── detection/           # 改进 YOLOv5s
│   ├── risk/                # 违规评估 + ELC + 分级
│   └── measurement/         # 尺度测量（接口占位）
└── tests/
```

---

## 许可

源码以 [MIT License](LICENSE) 发布。引用公开论文思想时请自行遵循原论文引用规范；本仓库不包含第三方检测框架的发行副本。
