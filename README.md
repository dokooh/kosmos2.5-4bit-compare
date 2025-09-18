# Kosmos-2.5 4-bit Quantization and OCR Inference

This repository contains scripts for quantizing the Kosmos-2.5 model using different 4-bit quantization strategies and performing OCR inference on images.

## Scripts Overview

### 1. `quantize_kosmos25_4bit.py`
Static 4-bit quantization script that:
- Tests multiple quantization strategies (NF4, FP4, with different compute dtypes)
- Saves quantized models to disk
- Monitors GPU/RAM memory usage
- Provides detailed debugging output
- Saves comprehensive results to JSON

### 2. `ocr_inference_kosmos25.py`
OCR inference script that:
- Loads locally saved quantized models
- Performs OCR/Markdown tasks on images
- Monitors memory footprint during inference
- Can test single or multiple models
- Provides detailed debugging output

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quantization

```bash
# Basic usage - quantize with all strategies
python quantize_kosmos25_4bit.py --output-dir ./quantized_models

# Specific strategies only
python quantize_kosmos25_4bit.py \
    --output-dir ./quantized_models \
    --strategies nf4 fp4 \
    --results-file my_results.json

# Different model
python quantize_kosmos25_4bit.py \
    --output-dir ./quantized_models \
    --model-name microsoft/kosmos-2.5 \
    --strategies nf4_bf16
```

### OCR Inference

```bash
# Single model inference
python ocr_inference_kosmos25.py \
    --model-path ./quantized_models/kosmos25_nf4 \
    --image-path sample_image.jpg \
    --task ocr

# Test all quantized models
python ocr_inference_kosmos25.py \
    --model-path ./quantized_models \
    --image-path sample_image.jpg \
    --task markdown \
    --test-all-models

# Different tasks
python ocr_inference_kosmos25.py \
    --model-path ./quantized_models/kosmos25_fp4 \
    --image-path document.png \
    --task text_detection \
    --output-file detection_results.json
```

## Quantization Strategies

The script supports the following 4-bit quantization strategies:

1. **nf4**: NF4 quantization with double quantization and float16 compute
2. **fp4**: FP4 quantization with double quantization and float16 compute  
3. **nf4_bf16**: NF4 quantization with double quantization and bfloat16 compute
4. **fp4_bf16**: FP4 quantization with double quantization and bfloat16 compute
5. **nf4_single**: NF4 quantization without double quantization

## Tasks Supported

- **ocr**: Standard OCR text extraction
- **markdown**: Convert image to markdown format
- **text_detection**: Detect text regions
- **caption**: Generate image captions

## Output

### Quantization Results
```json
{
  "timestamp": "2025-01-18 11:49:53",
  "model_name": "microsoft/kosmos-2.5",
  "strategies_tested": ["nf4", "fp4"],
  "successful_strategies": 2,
  "results": [
    {
      "strategy": "nf4",
      "success": true,
      "quantization_time_seconds": 45.2,
      "model_size_gb": 2.1,
      "memory_usage": {...}
    }
  ]
}
```

### Inference Results
```json
{
  "timestamp": "2025-01-18 11:49:53", 
  "task": "ocr",
  "models_tested": 2,
  "results": [
    {
      "model_name": "kosmos25_nf4",
      "success": true,
      "generated_text": "Extracted text...",
      "timing": {
        "total_time_seconds": 3.2,
        "generation_time_seconds": 2.1
      },
      "memory_usage": {...}
    }
  ]
}
```

## Debugging

Both scripts provide extensive debugging output with `[DEBUG]` prefixes to trace:
- Memory usage at each step
- Model loading progress
- Inference timing
- Error handling with full tracebacks

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- ~10GB disk space for quantized models

## Example Scripts

Run the provided example scripts:

```bash
# Quantize models
bash run_quantization_example.sh

# Run inference (need sample_image.jpg)
bash run_inference_example.sh
```