#!/usr/bin/env python3
"""
Static 4-bit quantization script for Kosmos2.5 model
Saves quantized model to disk for later inference
"""

import torch
import argparse
import os
import sys
import gc
import psutil
from datetime import datetime
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
from transformers import BitsAndBytesConfig
import json

def get_memory_usage():
    """Get current CPU and GPU memory usage"""
    cpu_memory = psutil.virtual_memory()
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    
    return {
        'cpu_used_gb': cpu_memory.used / 1024**3,
        'cpu_available_gb': cpu_memory.available / 1024**3,
        'cpu_percent': cpu_memory.percent,
        'gpu_memory': gpu_memory
    }

def log_debug(message):
    """Debug logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}")
    sys.stdout.flush()

def log_error(message):
    """Error logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR {timestamp}] {message}")
    sys.stdout.flush()

def create_quantization_config(strategy_name):
    """Create quantization configuration for different strategies"""
    log_debug(f"Creating quantization config for strategy: {strategy_name}")
    
    configs = {
        "nf4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        "fp4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        "nf4_no_double": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
        "fp4_no_double": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
        "nf4_bf16": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
    }
    
    if strategy_name not in configs:
        raise ValueError(f"Unknown quantization strategy: {strategy_name}")
    
    log_debug(f"✅ Quantization config created for {strategy_name}")
    return configs[strategy_name]

def quantize_and_save_model(model_name, strategy_name, output_dir):
    """Quantize model and save to disk"""
    log_debug(f"Starting quantization of {model_name} with strategy {strategy_name}")
    
    # Create output directory
    strategy_output_dir = os.path.join(output_dir, f"kosmos25_{strategy_name}")
    os.makedirs(strategy_output_dir, exist_ok=True)
    log_debug(f"Created output directory: {strategy_output_dir}")
    
    # Memory before loading
    memory_before = get_memory_usage()
    log_debug(f"Memory before loading: CPU {memory_before['cpu_used_gb']:.2f}GB, GPU {memory_before['gpu_memory']['allocated'] if memory_before['gpu_memory'] else 0:.2f}GB")
    
    try:
        # Create quantization config
        quantization_config = create_quantization_config(strategy_name)
        
        # Load processor
        log_debug("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        log_debug("✅ Processor loaded successfully")
        
        # Load and quantize model
        log_debug(f"Loading and quantizing model with {strategy_name}...")
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        log_debug("✅ Model loaded and quantized successfully")
        
        # Memory after loading
        memory_after = get_memory_usage()
        log_debug(f"Memory after loading: CPU {memory_after['cpu_used_gb']:.2f}GB, GPU {memory_after['gpu_memory']['allocated'] if memory_after['gpu_memory'] else 0:.2f}GB")
        
        # Save quantized model
        log_debug("Saving quantized model...")
        model.save_pretrained(strategy_output_dir)
        processor.save_pretrained(strategy_output_dir)
        log_debug(f"✅ Model and processor saved to {strategy_output_dir}")
        
        # Save quantization info
        quant_info = {
            "strategy": strategy_name,
            "model_name": model_name,
            "quantization_config": {
                "load_in_4bit": quantization_config.load_in_4bit,
                "bnb_4bit_quant_type": quantization_config.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": str(quantization_config.bnb_4bit_compute_dtype),
                "bnb_4bit_use_double_quant": quantization_config.bnb_4bit_use_double_quant,
            },
            "memory_usage": {
                "before_loading": memory_before,
                "after_loading": memory_after,
                "memory_increase_gb": memory_after['cpu_used_gb'] - memory_before['cpu_used_gb']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(strategy_output_dir, "quantization_info.json"), "w") as f:
            json.dump(quant_info, f, indent=2, default=str)
        
        log_debug(f"✅ Quantization info saved")
        
        # Cleanup
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        log_debug("✅ Memory cleanup completed")
        
        return strategy_output_dir, quant_info
        
    except Exception as e:
        log_error(f"Failed to quantize with strategy {strategy_name}: {str(e)}")
        log_error(f"Exception type: {type(e).__name__}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Quantize Kosmos2.5 model with 4-bit strategies")
    parser.add_argument("--model", default="microsoft/kosmos-2.5", 
                       help="Model name or path (default: microsoft/kosmos-2.5)")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory to save quantized models")
    parser.add_argument("--strategies", nargs="+", 
                       choices=["nf4", "fp4", "nf4_no_double", "fp4_no_double", "nf4_bf16"],
                       default=["nf4", "fp4", "nf4_no_double", "fp4_no_double"],
                       help="Quantization strategies to test")
    
    args = parser.parse_args()
    
    log_debug(f"Starting quantization script")
    log_debug(f"Model: {args.model}")
    log_debug(f"Output directory: {args.output_dir}")
    log_debug(f"Strategies: {args.strategies}")
    log_debug(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_debug(f"CUDA device: {torch.cuda.get_device_name()}")
        log_debug(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    successful_quantizations = []
    failed_quantizations = []
    
    for strategy in args.strategies:
        log_debug(f"\n{'='*50}")
        log_debug(f"Processing strategy: {strategy}")
        log_debug(f"{'='*50}")
        
        output_path, quant_info = quantize_and_save_model(args.model, strategy, args.output_dir)
        
        if output_path and quant_info:
            results[strategy] = {
                "status": "success",
                "output_path": output_path,
                "info": quant_info
            }
            successful_quantizations.append(strategy)
            log_debug(f"✅ {strategy} quantization completed successfully")
        else:
            results[strategy] = {
                "status": "failed",
                "output_path": None,
                "info": None
            }
            failed_quantizations.append(strategy)
            log_debug(f"❌ {strategy} quantization failed")
    
    # Save overall results
    final_results = {
        "model_name": args.model,
        "strategies_tested": args.strategies,
        "successful_quantizations": successful_quantizations,
        "failed_quantizations": failed_quantizations,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = os.path.join(args.output_dir, "quantization_results.json")
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    log_debug(f"\n{'='*50}")
    log_debug(f"QUANTIZATION SUMMARY")
    log_debug(f"{'='*50}")
    log_debug(f"Total strategies tested: {len(args.strategies)}")
    log_debug(f"Successful: {len(successful_quantizations)} - {successful_quantizations}")
    log_debug(f"Failed: {len(failed_quantizations)} - {failed_quantizations}")
    log_debug(f"Results saved to: {results_file}")
    log_debug(f"Individual models saved in: {args.output_dir}")
    
    if failed_quantizations:
        log_error(f"Some quantizations failed. Check the logs above for details.")
        sys.exit(1)
    else:
        log_debug("✅ All quantizations completed successfully!")

if __name__ == "__main__":
    main()