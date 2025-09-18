#!/bin/bash

# Example script to run quantization
echo "[DEBUG] Starting quantization example..."

# Create output directory
mkdir -p ./quantized_models

# Run quantization with all strategies
python quantize_kosmos25_4bit.py \
    --output-dir ./quantized_models \
    --model-name microsoft/kosmos-2.5 \
    --strategies nf4 fp4 nf4_bf16 \
    --results-file quantization_results.json

echo "[DEBUG] Quantization completed. Results saved to quantization_results.json"
echo "[DEBUG] Quantized models saved to ./quantized_models/"