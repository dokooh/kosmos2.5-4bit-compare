#!/bin/bash

# Example script to run OCR inference
echo "[DEBUG] Starting OCR inference example..."

# Make sure you have an image file
if [ ! -f "sample_image.jpg" ]; then
    echo "[ERROR] Please provide a sample_image.jpg file for testing"
    exit 1
fi

# Run inference on a single model
python ocr_inference_kosmos25.py \
    --model-path ./quantized_models/kosmos25_nf4 \
    --image-path sample_image.jpg \
    --task ocr \
    --output-file single_model_results.json

# Run inference on all quantized models
python ocr_inference_kosmos25.py \
    --model-path ./quantized_models \
    --image-path sample_image.jpg \
    --task ocr \
    --test-all-models \
    --output-file all_models_results.json

echo "[DEBUG] Inference completed. Results saved to JSON files"