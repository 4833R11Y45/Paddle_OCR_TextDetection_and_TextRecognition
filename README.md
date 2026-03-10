# PaddleOCR Module — Text Detection & Text Recognition

Standalone PaddleOCR wrappers for the project using **TextDetection** and **TextRecognition** only.  
**No PPStructure / PPStructureV3.**

## Folder Structure

```
app/paddle_ocr/
├── __init__.py          # Public API exports
├── config.py            # OCRConfig dataclass (all tunable parameters)
├── text_detector.py     # TextDetector — wraps PaddleOCR TextDetection
├── text_recognizer.py   # TextRecognizer — wraps PaddleOCR TextRecognition
├── ocr_engine.py        # PaddleOCREngine — end-to-end detect + recognize
├── preprocessing.py     # CLAHE, scaling, tile generation
├── postprocessing.py    # Dedup, filtering, fragment merging
├── utils.py             # TextBox dataclass, box math, serialization
├── visualization.py     # Debug drawing helpers
├── requirements.txt     # Dependencies (paddleocr only, no doc-parser)
└── README.md            # This file
```

## Installation

```bash
# Step 1 — PaddlePaddle GPU (custom index, must be first)
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Step 2 — PaddleOCR (no doc-parser extra = no PPStructure)
pip install paddleocr==3.3.2

# Step 3 — Other deps (most are already in the project root requirements.txt)
pip install -r app/paddle_ocr/requirements.txt
```

## Quick Start

### Basic OCR

```python
from app.paddle_ocr import PaddleOCREngine, OCRConfig
import cv2

config = OCRConfig(use_gpu=True)
engine = PaddleOCREngine(config)

image = cv2.imread("blueprint.png")
results = engine.run(image)

for box in results:
    print(f"{box.text}  (conf={box.confidence:.2f})  bbox={box.get_bbox()}")
```

### Tile-Based OCR (Large Images)

```python
# For images larger than 4000px — tiles with 50% overlap
results = engine.run_tiled(large_image, tile_size=1000, overlap=0.50)
```

### One-Liner Text Extraction

```python
all_text = engine.extract_text(image, use_tiling=True)
print(all_text)
```

### File-Based OCR (Auto Scaling)

```python
results = engine.run_file("/path/to/image.png", use_tiling=True)
```

### Batch Processing

```python
paths = ["page1.png", "page2.png", "page3.png"]
batch_results = engine.run_batch(paths)
for path, boxes in batch_results.items():
    print(f"{path}: {len(boxes)} text regions")
```

### Using Individual Components

```python
from .paddle_ocr import TextDetector, TextRecognizer, OCRConfig

config = OCRConfig(use_gpu=True)

# Detection only
detector = TextDetector(config)
polys_and_scores = detector.detect(image)

# Recognition only
recognizer = TextRecognizer(config)
crops = [image[y1:y2, x1:x2] for ...]  # your cropped regions
texts_and_scores = recognizer.recognize(crops)
```

### Preprocessing Utilities

```python
from .paddle_ocr.preprocessing import (
    apply_clahe,
    scale_image_if_needed,
    generate_tiles,
)

# CLAHE contrast enhancement
enhanced = apply_clahe(image, clip_limit=1.5)

# Scale for PaddleOCR's 4000px limit
scaled, factor = scale_image_if_needed(image, max_side_px=3800)

# Generate overlapping tiles
tiles = generate_tiles(image, tile_size=1000, overlap=0.50)
for tile_img, x_offset, y_offset in tiles:
    ...
```

### Post-Processing

```python
from .paddle_ocr.postprocessing import postprocess, deduplicate_by_location

# Full pipeline
clean_boxes = postprocess(raw_boxes, min_confidence=0.3, dedup_distance=50)

# Or individual steps
from .paddle_ocr.postprocessing import (
    filter_by_confidence,
    filter_by_size,
    merge_horizontal_fragments,
    deduplicate_by_location,
)
```

### Visualization

```python
from .paddle_ocr.visualization import draw_boxes, save_annotated

annotated = draw_boxes(image, results, show_text=True, show_confidence=True)
save_annotated(image, results, "debug_output.png")
```

### JSON Serialization

```python
dicts = PaddleOCREngine.results_to_dicts(results)
# [{"text": "10'-6\"", "bbox": [100, 200, 180, 230], "confidence": 0.95, ...}, ...]
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_gpu` | `True` | Use GPU acceleration |
| `det_model_name` | `PP-OCRv5_server_det` | Detection model |
| `rec_model_name` | `PP-OCRv5_server_rec` | Recognition model |
| `det_db_thresh` | `0.25` | DB binarization threshold |
| `det_db_box_thresh` | `0.5` | Box score threshold |
| `det_limit_side_len` | `20000` | Max side before internal resize |
| `rec_batch_size` | `8` | Recognition batch size |
| `rec_score_threshold` | `0.5` | Min recognition confidence |
| `min_confidence` | `0.20` | Min combined (det×rec) confidence |
| `tile_size` | `1000` | Tile side length (px) |
| `tile_overlap` | `0.50` | Tile overlap ratio |
| `max_side_px` | `3800` | Scale threshold (PaddleOCR limit is 4000) |
| `use_clahe` | `False` | CLAHE contrast enhancement |
| `clahe_clip_limit` | `1.5` | CLAHE clip limit |
| `dedup_distance` | `50.0` | Dedup center-distance threshold (px) |
| `min_box_width` | `15` | Minimum box width to keep |
| `min_box_height` | `8` | Minimum box height to keep |

## Models Used

- **Text Detection**: `PP-OCRv5_server_det` — DB-based text detector
- **Text Recognition**: `PP-OCRv5_server_rec` — SVTR-based text recognizer
- **No PPStructure** — no layout analysis, no table recognition, no formula recognition

## Compatibility

- PaddlePaddle GPU 3.2.0
- PaddleOCR 3.3.2
- Python 3.10
- CUDA 12.6
