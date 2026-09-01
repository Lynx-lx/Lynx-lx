# 电力安防视觉尺度测量图像检测系统

面向变电站 / 输电线路巡检的**计算机视觉**工程：轻量化目标检测、视觉尺度相关的隐患评估、万级标注数据的自动化校验。适合作为 AI 开发 / 算法岗作品集仓库。

**本仓库是代码实现。原始数据集、现场相机标定文件需在本地自行准备，请勿提交到 GitHub。**

---

## 仓库声明

**本仓库为后期独立复现实现，并非原始大创竞赛源码，原始学校项目不予开源。**

- 全部模块为从零重写，可公开推送到个人 GitHub。
- 不含校内原始数据、竞赛提交包、未公开权重或真实标定文件。
- 检测结构参考公开文献中的 YOLOv5s、ECA / SE / CBAM 等思路的独立实现，不拷贝任何学校仓库或 Ultralytics 发行源码。
- 简历中的「视觉尺度测量下的图像检测系统」对应本复现仓库的能力范围，而不是已开源的校内原始仓库。

---

## 项目简介

| 能力 | 说明 |
|------|------|
| 电力安防检测 | `src/viscale/detection/`：改进 YOLOv5s（P2 小目标分支 + 注意力） |
| 视觉尺度 | `src/viscale/measurement/`：由本地相机内参与工作距离估算米/像素；风险模块可使用该尺度 |
| 风险评估 | `src/viscale/risk/`：自定义违规函数、ELC 阈值迭代、四级分级 |
| 数据工程 | `dataset/`：标注越界/损坏图校验、train/val/test 划分、类别统计（万级流式） |
| 演示 | `app.py`：Gradio 上传图片 → 画框 → 风险等级 |

默认类别：`person, helmet, insulator, bird_nest, smoke, fire, vehicle, foreign_object`。仓库**不附带训练权重**；无 `.pt` 时随机初始化仅用于打通流程。

---

## 技术栈

- Python 3.10+（开发环境 3.12）
- PyTorch、OpenCV、NumPy
- PyYAML（相机传感器模板）
- Gradio（本地演示，非重型 Web 框架）

---

## 数据集与传感器配置说明

### 数据集（`data/`）

`data/` 只保留目录结构（`.gitkeep`）。**原始数据集体积大，不提交仓库。**

本地请按 YOLO 约定放置：

```
data/
  images/     # 图像
  labels/     # 与图像同名的 .txt
              # 每行: class_id  xc  yc  w  h  （相对宽高归一化到 0~1）
  raw/ processed/ annotations/ samples/   # 可选
```

示例：把巡检图放到 `data/images/`，标注放到 `data/labels/0001.txt` 等。然后：

```bash
python dataset/validate_annotations.py --root data
python dataset/split_dataset.py --root data --ratios 0.7,0.2,0.1 --out-dir dataset/splits
python dataset/analyze_dataset.py --root data
```

### 相机传感器（`config/camera_sensor/`）

**真实标定参数不随仓库提交。** 目录中提交的是：

- `.gitkeep`：占位
- `camera_template.yaml`：**模板**（示例占位数值 + 注释），不是现场标定

本地步骤：

```bash
# Windows
copy config\camera_sensor\camera_template.yaml config\camera_sensor\camera.yaml
# Linux / macOS
cp config/camera_sensor/camera_template.yaml config/camera_sensor/camera.yaml
```

编辑 `camera.yaml`：将 `meta.is_template` 改为 `false`，填入你的相机名称、分辨率、帧率、内参 **K**、畸变 **D**、ROI；外参按文件内坐标系注释填写。`camera.yaml` 已被 `.gitignore` 忽略。

运行时默认读取 `config/camera_sensor/camera.yaml`。**文件不存在不会崩溃**，终端会提示复制模板；检测与风险仍可运行，只是不启用内参尺度。

```bash
python app.py --camera-config config/camera_sensor/camera.yaml --working-distance-m 8
```

---

## 环境与运行

本项目为 Python 工程，**无需 CMake / 编译**。

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU 可选：先按 [PyTorch](https://pytorch.org/get-started/locally/) 安装 CUDA 版 `torch`，再安装其余依赖。

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`。可选：`--weights models/checkpoints/yolov5s_lite.pt`、`--device cpu`、`--attn eca`。

```bash
python scripts/demo_yolov5s_lite.py --image path/to/local.jpg
```

---

## 项目结构

```
.
├── app.py
├── requirements.txt
├── config/camera_sensor/    # 传感器：仅模板 + gitkeep
├── configs/                 # 其它配置占位
├── data/                    # 仅 gitkeep；本地放图与标签
├── dataset/                 # 校验 / 划分 / 统计脚本
├── models/                  # 权重目录（*.pt 不入库）
├── src/viscale/
│   ├── detection/
│   ├── io/                  # 相机 YAML、数据目录约定
│   ├── measurement/
│   └── risk/
├── scripts/
└── tests/
```

---

## 许可

[MIT License](LICENSE)。引用公开论文时请自行规范引用。本仓库不含第三方检测框架发行副本，也不含原始竞赛数据与硬件标定文件。
