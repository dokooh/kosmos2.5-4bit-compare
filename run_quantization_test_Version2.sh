#!/bin/bash

# Comprehensive test script for Kosmos2.5 4-bit quantization
# This script runs both quantization and inference testing

set -e  # Exit on any error

echo "🚀 Starting Kosmos2.5 4-bit Quantization Test Suite"
echo "=================================================="

# Configuration
MODEL_NAME="microsoft/kosmos-2.5"
OUTPUT_DIR="./kosmos25_quantized_models"
RESULTS_DIR="./inference_results"
SAMPLE_IMAGE="sample_image.jpg"  # You need to provide this

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

echo "📁 Created directories:"
echo "  - Models: $OUTPUT_DIR"
echo "  - Results: $RESULTS_DIR"

# Check if sample image exists
if [ ! -f "$SAMPLE_IMAGE" ]; then
    echo "⚠️  Warning: Sample image '$SAMPLE_IMAGE' not found!"
    echo "Please provide a sample image for OCR testing."
    echo "You can download a sample or use your own image."
    echo ""
    echo "Example commands to get a sample image:"
    echo "wget https://example.com/sample-text-image.jpg -O $SAMPLE_IMAGE"
    echo "or provide your own image file"
    exit 1
fi

echo "🖼️  Using sample image: $SAMPLE_IMAGE"

# Step 1: Quantize models with different strategies
echo ""
echo "🔧 Step 1: Quantizing models with different 4-bit strategies"
echo "============================================================"

python quantize_kosmos25_4bit.py \
    --model "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --strategies nf4 fp4 nf4_no_double fp4_no_double

if [ $? -eq 0 ]; then
    echo "✅ Quantization completed successfully"
else
    echo "❌ Quantization failed"
    exit 1
fi

# Step 2: Test inference on each quantized model
echo ""
echo "🧠 Step 2: Testing OCR inference on quantized models"
echo "===================================================="

# List of strategies to test (should match successful quantizations)
STRATEGIES=("nf4" "fp4" "nf4_no_double" "fp4_no_double")

for strategy in "${STRATEGIES[@]}"; do
    model_path="$OUTPUT_DIR/kosmos25_$strategy"
    result_file="$RESULTS_DIR/ocr_results_$strategy.json"
    
    echo ""
    echo "🔍 Testing strategy: $strategy"
    echo "Model path: $model_path"
    echo "Result file: $result_file"
    
    if [ -d "$model_path" ]; then
        python ocr_inference_kosmos25.py \
            --model-path "$model_path" \
            --image "$SAMPLE_IMAGE" \
            --output "$result_file" \
            --task OCR
        
        if [ $? -eq 0 ]; then
            echo "✅ $strategy inference completed"
        else
            echo "❌ $strategy inference failed"
        fi
    else
        echo "❌ Model directory not found: $model_path"
    fi
done

# Step 3: Test Markdown task as well
echo ""
echo "📝 Step 3: Testing Markdown extraction on quantized models"
echo "=========================================================="

for strategy in "${STRATEGIES[@]}"; do
    model_path="$OUTPUT_DIR/kosmos25_$strategy"
    result_file="$RESULTS_DIR/markdown_results_$strategy.json"
    
    echo ""
    echo "📝 Testing Markdown with strategy: $strategy"
    
    if [ -d "$model_path" ]; then
        python ocr_inference_kosmos25.py \
            --model-path "$model_path" \
            --image "$SAMPLE_IMAGE" \
            --output "$result_file" \
            --task MARKDOWN
        
        if [ $? -eq 0 ]; then
            echo "✅ $strategy Markdown extraction completed"
        else
            echo "❌ $strategy Markdown extraction failed"
        fi
    else
        echo "❌ Model directory not found: $model_path"
    fi
done

# Step 4: Generate summary report
echo ""
echo "📊 Step 4: Generating summary report"
echo "===================================="

python -c "
import json
import os
import glob
from datetime import datetime

print('\\n📈 QUANTIZATION AND INFERENCE SUMMARY')
print('=' * 50)

# Load quantization results
quant_results_file = '$OUTPUT_DIR/quantization_results.json'
if os.path.exists(quant_results_file):
    with open(quant_results_file, 'r') as f:
        quant_data = json.load(f)
    
    print(f'📦 Quantization Results:')
    print(f'  - Total strategies tested: {len(quant_data[\"strategies_tested\"])}')
    print(f'  - Successful: {len(quant_data[\"successful_quantizations\"])}')
    print(f'  - Failed: {len(quant_data[\"failed_quantizations\"])}')
    print(f'  - Successful strategies: {quant_data[\"successful_quantizations\"]}')
    if quant_data['failed_quantizations']:
        print(f'  - Failed strategies: {quant_data[\"failed_quantizations\"]}')

# Load inference results
print(f'\\n🧠 OCR Inference Results:')
ocr_files = glob.glob('$RESULTS_DIR/ocr_results_*.json')
for file in sorted(ocr_files):
    strategy = os.path.basename(file).replace('ocr_results_', '').replace('.json', '')
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        text_len = len(data['extracted_text'])
        inference_time = data['inference_time_seconds']
        print(f'  - {strategy}: {text_len} chars, {inference_time:.2f}s')
    except:
        print(f'  - {strategy}: ❌ Failed to load results')

print(f'\\n📝 Markdown Extraction Results:')
md_files = glob.glob('$RESULTS_DIR/markdown_results_*.json')
for file in sorted(md_files):
    strategy = os.path.basename(file).replace('markdown_results_', '').replace('.json', '')
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        text_len = len(data['extracted_text'])
        inference_time = data['inference_time_seconds']
        print(f'  - {strategy}: {text_len} chars, {inference_time:.2f}s')
    except:
        print(f'  - {strategy}: ❌ Failed to load results')

print(f'\\n📁 Output Locations:')
print(f'  - Quantized models: $OUTPUT_DIR')
print(f'  - Inference results: $RESULTS_DIR')
print(f'\\n✅ Test suite completed at {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
"

echo ""
echo "🎉 Kosmos2.5 4-bit Quantization Test Suite Completed!"
echo "====================================================="
echo "Check the output directories for detailed results:"
echo "  - Quantized models: $OUTPUT_DIR"
echo "  - Inference results: $RESULTS_DIR"