# Lynx-lx

统计学背景，独立从 0 到 1 做过完整的**多模态 Agent**、**仿真数据闭环**和**目标检测**项目。

---

## 技术栈

**大模型 / Agent**：ReAct、Function-Calling、上下文工程；LangGraph；金标数据构建、LLM-as-Judge、消融与 bad-case 迭代；了解 SFT / DPO 基础。

**计算机视觉**：YOLO 系列、OpenCV 预处理、mAP 评估、难例挖掘、CARLA仿真图像生成。

**数据**：Hive SQL、MySQL、S3；数据清洗、特征提取、标注与样本筛选。

**开发**：Python、PyTorch 基础、Linux / Mac、Git、Docker、CARLA、Gradio、Cursor。

---

## 重点项目

### 1. VLM 泊车故障分析 Agent

独立从 0 到 1。用 VLM + ReAct / Function-Calling、LangGraph 做泊车故障分析；CARLA 合成泊车场景图像，构建评测集；LLM-as-Judge、消融实验、bad-case 回灌；Gradio 网页 Demo。

- **技术**：多模态 Agent、仿真数据生成、结构化故障报告、本地评测闭环
- **仓库**：[vlm_parking_agent](https://github.com/Lynx-lx/vlm_parking_agent)

![Demo](assets/parking_agent.gif)

### 2. Sim2Real 机械臂抓取（VLA）

独立从 0 到 1。MuJoCo 示教采集 → 多模态数据 Pipeline → 轻量 ACT / Diffusion Policy → SmolVLA 语言规划 → ROS2 闭环，预留 SO-100 与域随机化。

- **技术**：模仿学习、VLA、仿真到真机接口、ROS2、数据集构建与评测
- **仓库**：[Sim2Real-VLA-Manipulation](https://github.com/Lynx-lx/Sim2Real-VLA-Manipulation)

![Demo](assets/sim2real_vla.gif)

### 3. YOLOv5 电力安防缺陷检测

独立复现目标检测全流程：数据构建、OpenCV 预处理、难例挖掘、mAP 评估与可视化 Demo。

- **技术**：2D 感知、数据集与难例、检测评估
- **仓库**：[power-security-detection-system](https://github.com/Lynx-lx/power-security-detection-system)

![Demo](assets/yolo_power.gif)


---

## 学习笔记

### 算法基础框架

用同一套 `Estimator` / `Transform` / `Pipeline` 接口整理机器学习、大模型、图像、规控、NLP、推荐、SLAM、语音、多模态的最小可运行笔记与示例。

- **仓库**：[Learning-Note](https://github.com/Lynx-lx/Learning-Note)

---

## 联系方式

- GitHub：https://github.com/Lynx-lx
- 邮箱：kec31833@gmail.com

