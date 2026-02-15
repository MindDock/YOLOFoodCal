# YOLOFoodCal

Lightweight AI Food Detection & Calorie Estimation Demo

> 用于展示 YOLO 如何落地到真实场景的 Demo 项目

## Features

- YOLO26 食物检测 (支持实例分割)
- Segmentation Mask 面积估算
- 静态 JSON 营养表热量计算
- 单机运行，无需数据库
- 支持图片/摄像头/视频输入
- Streamlit Web UI
- ONNX 导出支持

## 快速开始

### 安装

Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Requirement already satisfied: ultralytics>=8.4.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 5)) (8.4.13)
Requirement already satisfied: torch>=2.0.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 6)) (2.9.1)
Requirement already satisfied: numpy>=1.24.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 7)) (2.2.6)
Requirement already satisfied: opencv-python>=4.9.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 8)) (4.9.0.80)
Requirement already satisfied: Pillow>=10.0.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 9)) (11.3.0)
Requirement already satisfied: streamlit>=1.30.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 12)) (1.54.0)
Requirement already satisfied: pandas>=2.0.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 15)) (2.3.3)
Requirement already satisfied: PyYAML>=6.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 16)) (6.0.3)
Requirement already satisfied: tqdm>=4.65.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 19)) (4.67.1)
Requirement already satisfied: colorama>=0.4.6 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 20)) (0.4.6)
Requirement already satisfied: pytest>=7.0.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 23)) (9.0.2)
Requirement already satisfied: pytest-cov>=4.0.0 in /opt/miniconda3/lib/python3.13/site-packages (from -r requirements.txt (line 24)) (7.0.0)
Requirement already satisfied: matplotlib>=3.3.0 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (3.10.8)
Requirement already satisfied: requests>=2.23.0 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (2.32.5)
Requirement already satisfied: scipy>=1.4.1 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (1.16.3)
Requirement already satisfied: torchvision>=0.9.0 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (0.24.1)
Requirement already satisfied: psutil>=5.8.0 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (7.2.0)
Requirement already satisfied: polars>=0.20.0 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (1.36.1)
Requirement already satisfied: ultralytics-thop>=2.0.18 in /opt/miniconda3/lib/python3.13/site-packages (from ultralytics>=8.4.0->-r requirements.txt (line 5)) (2.0.18)
Requirement already satisfied: filelock in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (3.20.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (4.15.0)
Requirement already satisfied: setuptools in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (80.9.0)
Requirement already satisfied: sympy>=1.13.3 in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (3.6.1)
Requirement already satisfied: jinja2 in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /opt/miniconda3/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 6)) (2025.10.0)
Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (6.0.0)
Requirement already satisfied: blinker<2,>=1.5.0 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (1.9.0)
Requirement already satisfied: cachetools<7,>=5.5 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (6.2.2)
Requirement already satisfied: click<9,>=7.0 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (8.1.8)
Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (3.1.46)
Requirement already satisfied: packaging>=20 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (25.0)
Requirement already satisfied: pydeck<1,>=0.8.0b4 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (0.9.1)
Requirement already satisfied: protobuf<7,>=3.20 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (5.29.5)
Requirement already satisfied: pyarrow>=7.0 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (23.0.0)
Requirement already satisfied: tenacity<10,>=8.1.0 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (8.2.3)
Requirement already satisfied: toml<2,>=0.10.1 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (0.10.2)
Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /opt/miniconda3/lib/python3.13/site-packages (from streamlit>=1.30.0->-r requirements.txt (line 12)) (6.5.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/miniconda3/lib/python3.13/site-packages (from pandas>=2.0.0->-r requirements.txt (line 15)) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/miniconda3/lib/python3.13/site-packages (from pandas>=2.0.0->-r requirements.txt (line 15)) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.7 in /opt/miniconda3/lib/python3.13/site-packages (from pandas>=2.0.0->-r requirements.txt (line 15)) (2025.2)
Requirement already satisfied: jsonschema>=3.0 in /opt/miniconda3/lib/python3.13/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 12)) (4.26.0)
Requirement already satisfied: narwhals>=1.27.1 in /opt/miniconda3/lib/python3.13/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 12)) (2.16.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/miniconda3/lib/python3.13/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit>=1.30.0->-r requirements.txt (line 12)) (4.0.12)
Requirement already satisfied: smmap<6,>=3.0.1 in /opt/miniconda3/lib/python3.13/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit>=1.30.0->-r requirements.txt (line 12)) (5.0.2)
Requirement already satisfied: charset_normalizer<4,>=2 in /opt/miniconda3/lib/python3.13/site-packages (from requests>=2.23.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/miniconda3/lib/python3.13/site-packages (from requests>=2.23.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/lib/python3.13/site-packages (from requests>=2.23.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/miniconda3/lib/python3.13/site-packages (from requests>=2.23.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (2025.8.3)
Requirement already satisfied: iniconfig>=1.0.1 in /opt/miniconda3/lib/python3.13/site-packages (from pytest>=7.0.0->-r requirements.txt (line 23)) (2.3.0)
Requirement already satisfied: pluggy<2,>=1.5 in /opt/miniconda3/lib/python3.13/site-packages (from pytest>=7.0.0->-r requirements.txt (line 23)) (1.5.0)
Requirement already satisfied: pygments>=2.7.2 in /opt/miniconda3/lib/python3.13/site-packages (from pytest>=7.0.0->-r requirements.txt (line 23)) (2.19.1)
Requirement already satisfied: coverage>=7.10.6 in /opt/miniconda3/lib/python3.13/site-packages (from coverage[toml]>=7.10.6->pytest-cov>=4.0.0->-r requirements.txt (line 24)) (7.12.0)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/miniconda3/lib/python3.13/site-packages (from jinja2->torch>=2.0.0->-r requirements.txt (line 6)) (3.0.3)
Requirement already satisfied: attrs>=22.2.0 in /opt/miniconda3/lib/python3.13/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 12)) (25.4.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/miniconda3/lib/python3.13/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 12)) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in /opt/miniconda3/lib/python3.13/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 12)) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in /opt/miniconda3/lib/python3.13/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit>=1.30.0->-r requirements.txt (line 12)) (0.30.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/miniconda3/lib/python3.13/site-packages (from matplotlib>=3.3.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /opt/miniconda3/lib/python3.13/site-packages (from matplotlib>=3.3.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /opt/miniconda3/lib/python3.13/site-packages (from matplotlib>=3.3.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (4.61.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /opt/miniconda3/lib/python3.13/site-packages (from matplotlib>=3.3.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (1.4.9)
Requirement already satisfied: pyparsing>=3 in /opt/miniconda3/lib/python3.13/site-packages (from matplotlib>=3.3.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (3.2.5)
Requirement already satisfied: polars-runtime-32==1.36.1 in /opt/miniconda3/lib/python3.13/site-packages (from polars>=0.20.0->ultralytics>=8.4.0->-r requirements.txt (line 5)) (1.36.1)
Requirement already satisfied: six>=1.5 in /opt/miniconda3/lib/python3.13/site-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->-r requirements.txt (line 15)) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/miniconda3/lib/python3.13/site-packages (from sympy>=1.13.3->torch>=2.0.0->-r requirements.txt (line 6)) (1.3.0)

### 使用 CLI



### 使用 Web UI



## 项目结构



## 技术栈

- **模型**: Ultralytics YOLO26 (YOLO26n-seg)
- **推理**: PyTorch / ONNX Runtime
- **Web UI**: Streamlit
- **图像处理**: OpenCV, NumPy, Pillow

## 性能基准

- YOLO26n-seg: ~20-30 FPS (CPU)
- 模型大小: ~6MB
- 内存占用: ~500MB

## 支持的食物类别

| 类别 | 数量 | 示例 |
|------|------|------|
| 主食 (Staple) | 5 | 米饭、面条、面包、馒头、饺子 |
| 肉类 (Meat) | 6 | 鸡腿、牛排、鱼片、鸡蛋、排骨、虾仁 |
| 蔬果 (Fruit/Veg) | 6 | 苹果、香蕉、橙子、西瓜、黄瓜、西红柿 |
| 小吃 (Snack) | 6 | 薯条、汉堡、披萨、蛋糕、饼干、巧克力 |
| 饮品 (Drink) | 6 | 可乐、咖啡、牛奶、啤酒、果汁、豆浆 |

## 自定义训练

如需训练自定义模型:


[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 1% ──────────── 48.0KB/6.4MB 142.2KB/s 0.1s<45.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 2% ──────────── 144.0KB/6.4MB 385.7KB/s 0.2s<16.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 7% ╸─────────── 464.0KB/6.4MB 2.4MB/s 0.3s<2.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 27% ━━━───────── 1.7/6.4MB 12.8MB/s 0.4s<0.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 59% ━━━━━━━───── 3.8/6.4MB 20.6MB/s 0.5s<0.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 93% ━━━━━━━━━━━─ 6.0/6.4MB 21.6MB/s 0.6s<0.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt to 'yolo26n-seg.pt': 100% ━━━━━━━━━━━━ 6.4MB 9.8MB/s 0.7s
New https://pypi.org/project/ultralytics/8.4.14 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.13 🚀 Python-3.13.5 torch-2.9.1 CPU (Apple M4 Pro)
[34m[1mengine/trainer: [0magnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=food.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26n-seg.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Volumes/MindDockSSD/projects/opensource/runs/segment/train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=segment, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None

## 扩展

- 添加新的食物类别: 编辑 
- 调整份量估算: 修改  中的转换系数

## License

MIT License

## 鸣谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- 食物营养数据参考 USDA FoodData Central
