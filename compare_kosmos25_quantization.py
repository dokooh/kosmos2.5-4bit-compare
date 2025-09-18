#!/usr/bin/env python3
"""
Comprehensive comparison script for Kosmos-2.5 4-bit quantization strategies.
Tests multiple quantization approaches and generates detailed performance reports.
"""

import argparse
import os
import sys
import time
import torch
import gc
import psutil
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import (
    Kosmos2_5ForConditionalGeneration, 
    Kosmos2_5Processor,
    BitsAndBytesConfig
)

def get_memory_usage():
    """Get current GPU and RAM memory usage"""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    
    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
    return ram_mb, gpu_mb

def print_debug(message, level="INFO"):
    """Print debug message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

def create_quantization_configs():
    """Create all 4-bit quantization configurations"""
    return {
        'nf4': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        ),
        'nf4_double': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        'fp4': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        ),
        'fp4_double': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    }

def fix_model_dtypes(model):
    """Fix dtype mismatches in quantized models"""
    print_debug("Applying dtype fixes...")
    
    def fix_layer_dtypes(module, target_dtype=torch.float16):
        for name, param in module.named_parameters(recurse=False):
            if param.dtype != target_dtype and param.dtype not in [torch.int8, torch.uint8]:
                param.data = param.data.to(target_dtype)
        
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer.dtype != target_dtype and buffer.dtype not in [torch.int8, torch.uint8]:
                buffer.data = buffer.data.to(target_dtype)
        
        for child_module in module.children():
            fix_layer_dtypes(child_module, target_dtype)
    
    fix_layer_dtypes(model, torch.float16)
    model = model.to(torch.float16)
    return model

def test_quantization_strategy(model_name, strategy, config, image_path, task_type="markdown"):
    """Test a single quantization strategy"""
    print_debug(f"Testing quantization strategy: {strategy}")
    
    result = {
        'strategy': strategy,
        'success': False,
        'load_time': 0,
        'model_size_mb': 0,
        'ram_usage_mb': 0,
        'gpu_usage_mb': 0,
        'inference_time': 0,
        'generated_text': None,
        'error': None
    }
    
    try:
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Get initial memory
        ram_before, gpu_before = get_memory_usage()
        
        # Load model and processor
        print_debug(f"Loading model with {strategy} quantization...")
        load_start = time.time()
        
        processor = Kosmos2_5Processor.from_pretrained(model_name)
        
        if config is None:  # Original model
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:  # Quantized model
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            # Apply dtype fixes for quantized models
            model = fix_model_dtypes(model)
        
        model.eval()
        load_time = time.time() - load_start
        
        # Get memory after loading
        ram_after_load, gpu_after_load = get_memory_usage()
        
        # Estimate model size (approximate)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        print_debug(f"Model loaded in {load_time:.2f}s, size: {model_size_mb:.1f}MB")
        
        # Perform OCR inference
        print_debug("Performing OCR inference...")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare prompt
        prompt = "<md>" if task_type.lower() == "markdown" else "<ocr>"
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get memory before inference
        ram_before_inf, gpu_before_inf = get_memory_usage()
        
        # Perform inference
        inference_start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        inference_time = time.time() - inference_start
        
        # Get memory after inference
        ram_after_inf, gpu_after_inf = get_memory_usage()
        
        # Decode results
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if prompt in generated_text:
            generated_text = generated_text.split(prompt, 1)[1].strip()
        
        # Update results
        result.update({
            'success': True,
            'load_time': load_time,
            'model_size_mb': model_size_mb,
            'ram_usage_mb': ram_after_load - ram_before,
            'gpu_usage_mb': gpu_after_load - gpu_before,
            'inference_time': inference_time,
            'generated_text': generated_text
        })
        
        print_debug(f"✅ {strategy} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        print_debug(f"❌ {strategy} failed: {error_msg}", "ERROR")
        result['error'] = error_msg
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return result

def generate_report(results, output_dir):
    """Generate comprehensive comparison report"""
    print_debug("Generating comparison report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON results
    json_file = Path(output_dir) / "detailed_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'Strategy': result['strategy'],
            'Success': result['success'],
            'Load Time (s)': f"{result['load_time']:.2f}" if result['success'] else "N/A",
            'Model Size (MB)': f"{result['model_size_mb']:.1f}" if result['success'] else "N/A",
            'RAM Usage (MB)': f"{result['ram_usage_mb']:.1f}" if result['success'] else "N/A",
            'GPU Usage (MB)': f"{result['gpu_usage_mb']:.1f}" if result['success'] else "N/A",
            'Inference Time (s)': f"{result['inference_time']:.2f}" if result['success'] else "N/A",
            'Text Length': len(result['generated_text']) if result['generated_text'] else 0,
            'Error': result['error'] if result['error'] else "None"
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save CSV report
    csv_file = Path(output_dir) / "comparison_summary.csv"
    df.to_csv(csv_file, index=False)
    
    # Generate text report
    txt_file = Path(output_dir) / "comparison_report.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("KOSMOS-2.5 4-BIT QUANTIZATION COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Successful strategies
        successful = [r for r in results if r['success']]
        if successful:
            f.write("SUCCESSFUL STRATEGIES:\n")
            f.write("-" * 40 + "\n")
            for result in successful:
                f.write(f"• {result['strategy']}: ")
                f.write(f"Load: {result['load_time']:.2f}s, ")
                f.write(f"Inference: {result['inference_time']:.2f}s, ")
                f.write(f"Size: {result['model_size_mb']:.1f}MB\n")
            f.write("\n")
        
        # Failed strategies
        failed = [r for r in results if not r['success']]
        if failed:
            f.write("FAILED STRATEGIES:\n")
            f.write("-" * 40 + "\n")
            for result in failed:
                f.write(f"• {result['strategy']}: {result['error']}\n")
            f.write("\n")
        
        # Generated text samples
        f.write("GENERATED TEXT SAMPLES:\n")
        f.write("-" * 40 + "\n")
        for result in successful:
            if result['generated_text']:
                f.write(f"\n{result['strategy'].upper()}:\n")
                f.write("-" * 20 + "\n")
                preview = result['generated_text'][:500]
                if len(result['generated_text']) > 500:
                    preview += "..."
                f.write(preview + "\n")
    
    print_debug(f"Reports saved to {output_dir}")
    print_debug(f"- JSON: {json_file}")
    print_debug(f"- CSV: {csv_file}")
    print_debug(f"- Text: {txt_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare Kosmos-2.5 4-bit quantization strategies")
    parser.add_argument("--model", default="microsoft/kosmos-2.5",
                       help="Model name or path")
    parser.add_argument("--image_path", required=True,
                       help="Path to test image")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--task_type", choices=['ocr', 'markdown'], default='markdown',
                       help="Type of task to perform")
    parser.add_argument("--include_original", action='store_true',
                       help="Also test original (non-quantized) model")
    
    args = parser.parse_args()
    
    print_debug("=== Kosmos-2.5 4-bit Quantization Comparison ===")
    print_debug(f"Model: {args.model}")
    print_debug(f"Image: {args.image_path}")
    print_debug(f"Output: {args.output_dir}")
    print_debug(f"Task: {args.task_type}")
    print_debug(f"Include original: {args.include_original}")
    
    # Check inputs
    if not os.path.exists(args.image_path):
        print_debug(f"Image path does not exist: {args.image_path}", "ERROR")
        return 1
    
    # Check CUDA
    if torch.cuda.is_available():
        print_debug(f"CUDA: {torch.cuda.get_device_name()}")
    else:
        print_debug("CUDA not available")
    
    # Prepare test configurations
    configs = create_quantization_configs()
    
    test_cases = []
    if args.include_original:
        test_cases.append(('original', None))
    
    for strategy, config in configs.items():
        test_cases.append((strategy, config))
    
    print_debug(f"Will test {len(test_cases)} configurations")
    
    # Run tests
    results = []
    for i, (strategy, config) in enumerate(test_cases, 1):
        print_debug(f"\n{'='*60}")
        print_debug(f"TEST {i}/{len(test_cases)}: {strategy.upper()}")
        print_debug(f"{'='*60}")
        
        result = test_quantization_strategy(
            args.model, strategy, config, args.image_path, args.task_type
        )
        results.append(result)
        
        # Print immediate result
        if result['success']:
            print_debug(f"✅ {strategy}: Load {result['load_time']:.2f}s, "
                       f"Inference {result['inference_time']:.2f}s, "
                       f"Size {result['model_size_mb']:.1f}MB")
        else:
            print_debug(f"❌ {strategy}: {result['error']}")
    
    # Generate report
    print_debug(f"\n{'='*60}")
    print_debug("GENERATING REPORT")
    print_debug(f"{'='*60}")
    
    generate_report(results, args.output_dir)
    
    # Print final summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print_debug(f"\n{'='*60}")
    print_debug("FINAL SUMMARY")
    print_debug(f"{'='*60}")
    print_debug(f"Total tests: {len(results)}")
    print_debug(f"Successful: {len(successful)}")
    print_debug(f"Failed: {len(failed)}")
    
    if successful:
        print_debug("\nSuccessful strategies:")
        for result in successful:
            print_debug(f"  ✅ {result['strategy']}")
    
    if failed:
        print_debug("\nFailed strategies:")
        for result in failed:
            print_debug(f"  ❌ {result['strategy']}: {result['error']}")
    
    print_debug(f"\nDetailed results saved in: {args.output_dir}")
    
    return 0 if successful else 1

if __name__ == "__main__":
    sys.exit(main())