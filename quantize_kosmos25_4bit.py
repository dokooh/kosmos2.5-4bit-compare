#!/usr/bin/env python3
"""
4-bit Quantization Script for Kosmos-2.5 Model
Tests multiple quantization strategies and saves quantized models to disk.
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import psutil
import GPUtil
from transformers import (
    AutoProcessor, 
    Kosmos2_5ForConditionalGeneration,
    BitsAndBytesConfig
)
from accelerate import infer_auto_device_map, dispatch_model

print(f"[DEBUG] Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] PyTorch version: {torch.__version__}")
print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[DEBUG] CUDA version: {torch.version.cuda}")
    print(f"[DEBUG] GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[DEBUG] GPU {i}: {torch.cuda.get_device_name(i)}")

def get_memory_usage() -> Dict[str, float]:
    """Get current GPU and RAM memory usage"""
    print("[DEBUG] Getting memory usage...")
    
    memory_info = {
        'ram_used_gb': psutil.virtual_memory().used / (1024**3),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'ram_percent': psutil.virtual_memory().percent
    }
    
    try:
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                memory_info.update({
                    'gpu_used_mb': gpu.memoryUsed,
                    'gpu_total_mb': gpu.memoryTotal,
                    'gpu_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                })
                print(f"[DEBUG] GPU memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({memory_info['gpu_percent']:.1f}%)")
        
        # Also try torch memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            memory_info.update({
                'torch_allocated_mb': allocated,
                'torch_cached_mb': cached
            })
            print(f"[DEBUG] Torch GPU memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
            
    except Exception as e:
        print(f"[DEBUG] Error getting GPU memory: {e}")
        
    print(f"[DEBUG] RAM usage: {memory_info['ram_used_gb']:.1f}GB / {memory_info['ram_total_gb']:.1f}GB ({memory_info['ram_percent']:.1f}%)")
    return memory_info

def create_quantization_config(strategy: str) -> Optional[BitsAndBytesConfig]:
    """Create quantization configuration for different strategies"""
    print(f"[DEBUG] Creating quantization config for strategy: {strategy}")
    
    configs = {
        "nf4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        "fp4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        "nf4_bf16": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ),
        "fp4_bf16": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ),
        "nf4_single": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )
    }
    
    config = configs.get(strategy)
    if config:
        print(f"[DEBUG] Created config - Type: {config.bnb_4bit_quant_type}, Double quant: {config.bnb_4bit_use_double_quant}, Compute dtype: {config.bnb_4bit_compute_dtype}")
    else:
        print(f"[DEBUG] Unknown strategy: {strategy}")
    
    return config

def quantize_and_save_model(strategy: str, output_dir: str, model_name: str = "microsoft/kosmos-2.5") -> Dict[str, Any]:
    """Quantize model with given strategy and save to disk"""
    print(f"\n[DEBUG] ========== Starting quantization: {strategy} ==========")
    start_time = time.time()
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[DEBUG] Cleared GPU cache")
        
        # Get initial memory usage
        print("[DEBUG] Getting initial memory usage...")
        initial_memory = get_memory_usage()
        
        # Create quantization config
        print("[DEBUG] Creating quantization configuration...")
        quant_config = create_quantization_config(strategy)
        if not quant_config:
            raise ValueError(f"Unknown quantization strategy: {strategy}")
        
        # Load processor
        print("[DEBUG] Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        print("[DEBUG] Processor loaded successfully")
        
        # Load model with quantization
        print(f"[DEBUG] Loading model with {strategy} quantization...")
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("[DEBUG] Model loaded successfully")
        
        # Get memory usage after loading
        print("[DEBUG] Getting memory usage after model loading...")
        loaded_memory = get_memory_usage()
        
        # Create output directory
        strategy_output_dir = Path(output_dir) / f"kosmos25_{strategy}"
        strategy_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Created output directory: {strategy_output_dir}")
        
        # Save model and processor
        print("[DEBUG] Saving quantized model...")
        model.save_pretrained(strategy_output_dir)
        print("[DEBUG] Model saved successfully")
        
        print("[DEBUG] Saving processor...")
        processor.save_pretrained(strategy_output_dir)
        print("[DEBUG] Processor saved successfully")
        
        # Calculate model size on disk
        print("[DEBUG] Calculating model size on disk...")
        model_size_bytes = sum(f.stat().st_size for f in strategy_output_dir.rglob('*') if f.is_file())
        model_size_gb = model_size_bytes / (1024**3)
        print(f"[DEBUG] Model size on disk: {model_size_gb:.2f} GB")
        
        # Get final memory usage
        print("[DEBUG] Getting final memory usage...")
        final_memory = get_memory_usage()
        
        # Calculate memory differences
        ram_diff = loaded_memory['ram_used_gb'] - initial_memory['ram_used_gb']
        gpu_diff = 0
        if 'gpu_used_mb' in loaded_memory and 'gpu_used_mb' in initial_memory:
            gpu_diff = loaded_memory['gpu_used_mb'] - initial_memory['gpu_used_mb']
        
        end_time = time.time()
        quantization_time = end_time - start_time
        
        result = {
            'strategy': strategy,
            'success': True,
            'quantization_time_seconds': quantization_time,
            'model_size_gb': model_size_gb,
            'output_path': str(strategy_output_dir),
            'memory_usage': {
                'initial': initial_memory,
                'loaded': loaded_memory,
                'final': final_memory,
                'ram_increase_gb': ram_diff,
                'gpu_increase_mb': gpu_diff
            },
            'model_info': {
                'model_name': model_name,
                'quant_type': quant_config.bnb_4bit_quant_type,
                'double_quant': quant_config.bnb_4bit_use_double_quant,
                'compute_dtype': str(quant_config.bnb_4bit_compute_dtype)
            }
        }
        
        print(f"[DEBUG] Quantization completed successfully in {quantization_time:.2f} seconds")
        print(f"[DEBUG] RAM increase: {ram_diff:.2f} GB, GPU increase: {gpu_diff:.1f} MB")
        
        # Clear memory
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[DEBUG] Cleared model from memory")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Error during quantization: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        end_time = time.time()
        return {
            'strategy': strategy,
            'success': False,
            'error': str(e),
            'quantization_time_seconds': end_time - start_time,
            'traceback': traceback.format_exc()
        }

def main():
    parser = argparse.ArgumentParser(description='Quantize Kosmos-2.5 model with different 4-bit strategies')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory to save quantized models')
    parser.add_argument('--model-name', '-m', type=str, default='microsoft/kosmos-2.5',
                        help='HuggingFace model name (default: microsoft/kosmos-2.5)')
    parser.add_argument('--strategies', '-s', nargs='+', 
                        choices=['nf4', 'fp4', 'nf4_bf16', 'fp4_bf16', 'nf4_single'],
                        default=['nf4', 'fp4', 'nf4_bf16', 'fp4_bf16', 'nf4_single'],
                        help='Quantization strategies to test')
    parser.add_argument('--results-file', '-r', type=str, default='quantization_results.json',
                        help='File to save results (default: quantization_results.json)')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Arguments parsed:")
    print(f"[DEBUG] - Output directory: {args.output_dir}")
    print(f"[DEBUG] - Model name: {args.model_name}")
    print(f"[DEBUG] - Strategies: {args.strategies}")
    print(f"[DEBUG] - Results file: {args.results_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Created main output directory: {output_dir}")
    
    # Test all strategies
    all_results = []
    
    print(f"\n[DEBUG] ========== STARTING QUANTIZATION PROCESS ==========")
    print(f"[DEBUG] Testing {len(args.strategies)} strategies: {args.strategies}")
    
    for i, strategy in enumerate(args.strategies, 1):
        print(f"\n[DEBUG] ========== STRATEGY {i}/{len(args.strategies)}: {strategy} ==========")
        
        result = quantize_and_save_model(strategy, args.output_dir, args.model_name)
        all_results.append(result)
        
        if result['success']:
            print(f"[DEBUG] ✅ {strategy} completed successfully")
            print(f"[DEBUG] - Time: {result['quantization_time_seconds']:.2f}s")
            print(f"[DEBUG] - Size: {result['model_size_gb']:.2f}GB")
            print(f"[DEBUG] - Path: {result['output_path']}")
        else:
            print(f"[DEBUG] ❌ {strategy} failed: {result['error']}")
    
    # Save results to file
    results_file = Path(args.results_file)
    print(f"\n[DEBUG] Saving results to {results_file}")
    
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': args.model_name,
        'output_directory': args.output_dir,
        'strategies_tested': args.strategies,
        'total_strategies': len(args.strategies),
        'successful_strategies': len([r for r in all_results if r['success']]),
        'failed_strategies': len([r for r in all_results if not r['success']]),
        'results': all_results,
        'system_info': {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'pytorch_version': torch.__version__,
            'python_version': sys.version
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"[DEBUG] Results saved to {results_file}")
    
    # Print summary
    print(f"\n[DEBUG] ========== QUANTIZATION SUMMARY ==========")
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"[DEBUG] Total strategies tested: {len(all_results)}")
    print(f"[DEBUG] Successful: {len(successful)}")
    print(f"[DEBUG] Failed: {len(failed)}")
    
    if successful:
        print(f"\n[DEBUG] Successful quantizations:")
        for result in successful:
            print(f"[DEBUG] - {result['strategy']}: {result['model_size_gb']:.2f}GB in {result['quantization_time_seconds']:.1f}s")
    
    if failed:
        print(f"\n[DEBUG] Failed quantizations:")
        for result in failed:
            print(f"[DEBUG] - {result['strategy']}: {result['error']}")
    
    print(f"\n[DEBUG] Script completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()