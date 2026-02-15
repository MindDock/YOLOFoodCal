# 自定义食物类别指南

## 概述

YOLOFoodCal 支持两种方式来识别自定义食物类别：

1. **使用 COCO 预训练模型** - 已包含常见食物（pizza, burger, hot dog 等）
2. **训练自定义模型** - 支持任意食物类别

---

## 方式一：使用 COCO 预训练模型（最简单）

YOLO26n-seg 已经在 COCO 数据集上训练，可以检测以下食物类别：

| Class ID | 食物名称 | 中文 |
|----------|----------|------|
| 52 | pizza | 披萨 |
| 53 | pizza | 披萨 |
| 54 | banana | 香蕉 |
| 55 | apple | 苹果 |
| 56 | sandwich | 三明治 |
| 57 | orange | 橙子 |
| 58 | broccoli | 西兰花 |
| 59 | carrot | 胡萝卜 |
| 60 | hot dog | 热狗 |
| 61 | pizza | 披萨 |
| 62 | donut | 甜甜圈 |
| 63 | cake | 蛋糕 |

**优点**：无需训练，直接使用
**缺点**：只能识别 COCO 中的食物类别

---

## 方式二：使用 Roboflow 公开数据集（推荐）

### 步骤 1：选择数据集

Roboflow 上有多个公开的食物检测数据集：

| 数据集 | 类别数 | 链接 |
|--------|--------|------|
| Food Detection (YOLO) | 12 类 | [ahmad-nabil/food-detection-for-yolo-training](https://universe.roboflow.com/ahmad-nabil/food-detection-for-yolo-training) |
| Food Image (YOLO) | 85 类 | [food-image-classification/food-imgae-yolo](https://universe.roboflow.com/food-image-classification/food-imgae-yolo) |
| Foods Project 2 | 多种 | [foods-project/foods-project-2](https://universe.roboflow.com/foods-project/foods-project-2) |

### 步骤 2：下载数据集

1. 访问 Roboflow 页面
2. 点击 "Download" 按钮
3. 选择格式：**YOLO ZIP** (包含 txt 标注)
4. 解压到项目目录

### 步骤 3：训练模型

```bash
# 使用 YOLO26 训练
yolo segment train data=food.yaml model=yolo26n-seg.pt epochs=100

# 或使用脚本
python scripts/train_food_model.py --dataset /path/to/dataset --classes "rice,noodles,bread" --epochs 50
```

### 步骤 4：导出模型

```bash
# 导出为 ONNX
python scripts/export_onnx.py --model runs/segment/train/weights/best.pt
```

---

## 方式三：自己标注数据（最灵活）

### 步骤 1：收集图片

收集你想要识别的食物图片，建议：
- 每种食物至少 50-100 张图片
- 多种角度、光照条件
- 包含不同大小和遮挡情况

### 步骤 2：标注数据

推荐工具：
1. **Roboflow Annotate** (在线，推荐)
2. **LabelImg** (本地开源)
3. **CVAT** (企业级)

标注格式：YOLO Detection Format
```
# 每张图片对应一个 .txt 文件
# 格式: <class_id> <x_center> <y_center> <width> <height>
# 所有值都是归一化的 (0-1)

0 0.5 0.5 0.3 0.4  # class 0 (米饭)
1 0.2 0.3 0.2 0.2  # class 1 (面条)
```

### 步骤 3：创建数据集配置

```yaml
# data/food.yaml
path: /path/to/dataset
train: train/images
val: valid/images

names:
  0: rice
  1: noodles
  2: bread
  3: apple
  4: banana
```

### 步骤 4：训练

```bash
yolo segment train data=food.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
```

---

## 更新营养数据库

训练好模型后，需要更新 `data/nutrition_table.json`：

```json
{
  "rice": {
    "name": "米饭",
    "name_en": "Rice", 
    "category": "staple",
    "calories": 130,
    "protein": 2.6,
    "carbs": 28.2,
    "fat": 0.3,
    "px_to_gram_factor": 0.012,
    "cooked": true,
    "density": 0.85
  }
}
```

---

## 类别映射

在 `src/estimator.py` 中配置类别映射：

```python
# 添加到 class_mapping
self.class_mapping = {
    'rice': 'rice',
    'noodles': 'noodles', 
    'bread': 'bread',
    # ... 更多映射
}
```

---

## 常见问题

### Q: 需要多少训练数据？
A: 每类至少 50-100 张，建议 200+ 张以获得较好效果

### Q: 训练需要多长时间？
A: YOLO26n-seg 在 GPU 上约 10-30 分钟，在 CPU 上可能需要数小时

### Q: 如何提高精度？
- 增加训练数据量
- 使用数据增强
- 使用更大的模型 (yolo26m-seg)
- 调整超参数

### Q: 可以识别中文食物吗？
A: 可以，只需在营养数据库中添加对应的中文名称
