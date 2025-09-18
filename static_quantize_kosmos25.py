#!/usr/bin/env python3
"""
Static 4-bit Quantization Script for Kosmos-2.5
Quantizes the model and saves it to disk for later inference.
"""

import argparse
import os
import sys
import time
import torch
import gc
import psutil
from pathlib import Path
from transformers import (
    Kosmos2_5ForConditionalGeneration, 
    Kosmos2_5Processor,
    BitsAndBytesConfig
)
import bitsandbytes as bnb

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

def create_quantization_config(strategy):
    """Create BitsAndBytesConfig for different 4-bit strategies"""
    configs = {
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
    return configs.get(strategy)

def quantize_and_save_model(model_name, strategy, output_dir):
    """Quantize model using specified strategy and save to disk"""
    print_debug(f"Starting quantization process for strategy: {strategy}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Get initial memory usage
    ram_before, gpu_before = get_memory_usage()
    print_debug(f"Initial memory - RAM: {ram_before:.1f}MB, GPU: {gpu_before:.1f}MB")
    
    try:
        # Create quantization config
        quantization_config = create_quantization_config(strategy)
        if quantization_config is None:
            raise ValueError(f"Unknown quantization strategy: {strategy}")
        
        print_debug(f"Created quantization config: {strategy}")
        print_debug(f"Config details: quant_type={quantization_config.bnb_4bit_quant_type}, "
                   f"double_quant={quantization_config.bnb_4bit_use_double_quant}, "
                   f"compute_dtype={quantization_config.bnb_4bit_compute_dtype}")
        
        # Load processor
        print_debug("Loading processor...")
        processor = Kosmos2_5Processor.from_pretrained(model_name)
        
        # Load and quantize model
        print_debug("Loading and quantizing model...")
        start_time = time.time()
        
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print_debug(f"Model loaded and quantized in {load_time:.2f} seconds")
        
        # Get memory usage after loading
        ram_after, gpu_after = get_memory_usage()
        print_debug(f"Memory after loading - RAM: {ram_after:.1f}MB (+{ram_after-ram_before:.1f}MB), "
                   f"GPU: {gpu_after:.1f}MB (+{gpu_after-gpu_before:.1f}MB)")
        
        # Create output directory
        strategy_output_dir = Path(output_dir) / f"kosmos25_{strategy}"
        strategy_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save quantized model
        print_debug(f"Saving quantized model to {strategy_output_dir}...")
        save_start = time.time()
        
        model.save_pretrained(
            strategy_output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        processor.save_pretrained(strategy_output_dir)
        
        save_time = time.time() - save_start
        print_debug(f"Model saved in {save_time:.2f} seconds")
        
        # Check saved model size
        model_size = sum(f.stat().st_size for f in strategy_output_dir.rglob('*') if f.is_file())
        model_size_mb = model_size / 1024 / 1024
        print_debug(f"Saved model size: {model_size_mb:.1f}MB")
        
        # Save quantization info
        info_file = strategy_output_dir / "quantization_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Quantization Strategy: {strategy}\n")
            f.write(f"Original Model: {model_name}\n")
            f.write(f"Quantization Type: {quantization_config.bnb_4bit_quant_type}\n")
            f.write(f"Double Quantization: {quantization_config.bnb_4bit_use_double_quant}\n")
            f.write(f"Compute Dtype: {quantization_config.bnb_4bit_compute_dtype}\n")
            f.write(f"Load Time: {load_time:.2f}s\n")
            f.write(f"Save Time: {save_time:.2f}s\n")
            f.write(f"Model Size: {model_size_mb:.1f}MB\n")
            f.write(f"RAM Usage: {ram_after-ram_before:.1f}MB\n")
            f.write(f"GPU Usage: {gpu_after-gpu_before:.1f}MB\n")
        
        print_debug(f"✅ Successfully quantized and saved model with strategy: {strategy}")
        return True
        
    except Exception as e:
        print_debug(f"❌ Error during quantization: {str(e)}", "ERROR")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Static 4-bit quantization for Kosmos-2.5")
    parser.add_argument("--model", default="microsoft/kosmos-2.5", 
                       help="Model name or path")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory to save quantized models")
    parser.add_argument("--strategy", 
                       choices=['nf4', 'nf4_double', 'fp4', 'fp4_double', 'all'],
                       default='all',
                       help="Quantization strategy to use")
    
    args = parser.parse_args()
    
    print_debug("=== Kosmos-2.5 Static 4-bit Quantization ===")
    print_debug(f"Model: {args.model}")
    print_debug(f"Output directory: {args.output_dir}")
    print_debug(f"Strategy: {args.strategy}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print_debug(f"CUDA available: {torch.cuda.get_device_name()}")
        print_debug(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print_debug("CUDA not available, using CPU")
    
    # Determine strategies to run
    if args.strategy == 'all':
        strategies = ['nf4', 'nf4_double', 'fp4', 'fp4_double']
    else:
        strategies = [args.strategy]
    
    print_debug(f"Will quantize using strategies: {strategies}")
    
    # Run quantization for each strategy
    results = {}
    for strategy in strategies:
        print_debug(f"\n{'='*50}")
        print_debug(f"Starting quantization with strategy: {strategy}")
        print_debug(f"{'='*50}")
        
        success = quantize_and_save_model(args.model, strategy, args.output_dir)
        results[strategy] = success
        
        if success:
            print_debug(f"✅ {strategy} quantization completed successfully")
        else:
            print_debug(f"❌ {strategy} quantization failed")
    
    # Print summary
    print_debug(f"\n{'='*50}")
    print_debug("QUANTIZATION SUMMARY")
    print_debug(f"{'='*50}")
    
    for strategy, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print_debug(f"{strategy:12} : {status}")
    
    successful_count = sum(results.values())
    print_debug(f"\nSuccessfully quantized {successful_count}/{len(strategies)} strategies")
    
    if successful_count > 0:
        print_debug(f"\nQuantized models saved in: {args.output_dir}")
        print_debug("Use the OCR inference script to test the quantized models")

if __name__ == "__main__":
    main()