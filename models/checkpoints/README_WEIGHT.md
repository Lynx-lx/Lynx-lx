# 权重文件说明（本目录的 *.pt 均不提交 Git）

请将权重放在本地：`models/checkpoints/`。仓库只保留本说明与 `.gitkeep`。

## 1. `yolov5s_lite.pt`

本仓库 YOLOv5s‑P2 + 注意力、4 类检测头的微调权重。Web 推理**优先**加载此文件。

推荐本地生成（公开学术数据，非电网私有现场库）::

    python scripts/prepare_public_power_data.py
    python scripts/train_yolov5s_lite.py

- CPLID：绝缘子 / 缺陷 → `insulator`, `damaged_insulator`
- FOTL_Drone：鸟巢 / 风筝气球等 → `bird_nest`, `foreign_object`
- 主干初始化：官方 ultralytics `yolov5s.pt`（COCO）

不能等同于现场全量业务模型；检测头已经过公开标注监督。文件不提交 Git。

## 2. `yolov5s_lite_demo.pt`

由 `python scripts/pretrain_adapter.py` 在本地生成。

- 尽量加载你自行下载的官方 **YOLOv5s COCO** `yolov5s.pt` 主干中形状匹配的卷积参数
- **检测头随机初始化**，没有在电力数据上微调
- 仅保证网络结构完整、能跑通推理链路
- **不能**真实识别鸟巢、绝缘子、异物、绝缘子缺陷，**不能**用于安防业务

生成前请自行将官方 `yolov5s.pt` 放到本目录。脚本不会把任何权重写入 git。

## 3. `yolov5s.pt`

官方公开 COCO 预训练（使用者自行下载）。仅作为适配器输入，不提交仓库。

## 加载优先级（Web `app.py`）

1. `yolov5s_lite.pt` → 视为业务微调权重  
2. 否则 `yolov5s_lite_demo.pt` → 加载并打印「仅演示、头未微调」警告  
3. 两者都没有 → 模拟检测框（不返回空列表，不向前端泄露本地绝对路径）
