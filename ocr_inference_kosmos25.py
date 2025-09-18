#!/usr/bin/env python3
"""
OCR Inference Script for Kosmos-2.5
Tests quantized models on OCR/Markdown tasks with proper dtype handling.
"""

import argparse
import os
import sys
import time
import torch
import gc
import psutil
import json
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

def fix_model_dtypes(model):
    """Fix dtype mismatches in the model - crucial for 4-bit quantized models"""
    print_debug("Fixing model dtypes to prevent bias dtype errors...")
    
    def fix_layer_dtypes(module, target_dtype=torch.float16):
        """Recursively fix dtypes in model layers"""
        for name, param in module.named_parameters(recurse=False):
            if param.dtype != target_dtype and param.dtype != torch.int8 and param.dtype != torch.uint8:
                print_debug(f"Converting {name} from {param.dtype} to {target_dtype}")
                param.data = param.data.to(target_dtype)
        
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer.dtype != target_dtype and buffer.dtype != torch.int8 and buffer.dtype != torch.uint8:
                print_debug(f"Converting buffer {name} from {buffer.dtype} to {target_dtype}")
                buffer.data = buffer.data.to(target_dtype)
        
        # Recursively process child modules
        for child_module in module.children():
            fix_layer_dtypes(child_module, target_dtype)
    
    # Fix dtypes in the entire model
    fix_layer_dtypes(model, torch.float16)
    
    # Ensure model is in the right dtype
    model = model.to(torch.float16)
    
    print_debug("Model dtype fixes applied")
    return model

def load_model_and_processor(model_path, is_quantized=False):
    """Load model and processor with proper error handling"""
    print_debug(f"Loading model from: {model_path}")
    print_debug(f"Is quantized: {is_quantized}")
    
    try:
        # Load processor
        print_debug("Loading processor...")
        processor = Kosmos2_5Processor.from_pretrained(model_path)
        
        # Load model with appropriate settings
        print_debug("Loading model...")
        if is_quantized:
            # For pre-quantized models, load without quantization config
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # For original model
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Apply dtype fixes to prevent bias errors
        model = fix_model_dtypes(model)
        
        # Set model to eval mode
        model.eval()
        
        print_debug("Model and processor loaded successfully")
        return model, processor
        
    except Exception as e:
        print_debug(f"Error loading model: {str(e)}", "ERROR")
        raise

def perform_ocr_inference(model, processor, image_path, task_type="markdown"):
    """Perform OCR inference on the provided image"""
    print_debug(f"Starting OCR inference on: {image_path}")
    print_debug(f"Task type: {task_type}")
    
    try:
        # Load and process image
        print_debug("Loading image...")
        image = Image.open(image_path).convert('RGB')
        print_debug(f"Image size: {image.size}")
        
        # Prepare prompts based on task type
        if task_type.lower() == "markdown":
            prompt = "<md>"
            print_debug("Using Markdown extraction prompt")
        else:  # OCR
            prompt = "<ocr>"
            print_debug("Using OCR prompt")
        
        # Process inputs
        print_debug("Processing inputs...")
        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        print_debug(f"Moving inputs to device: {device}")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get memory usage before inference
        ram_before, gpu_before = get_memory_usage()
        print_debug(f"Memory before inference - RAM: {ram_before:.1f}MB, GPU: {gpu_before:.1f}MB")
        
        # Perform inference
        print_debug("Starting model generation...")
        start_time = time.time()
        
        with torch.no_grad():
            # Use more conservative generation parameters to avoid errors
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Use greedy decoding for stability
                temperature=None,
                top_p=None,
                num_beams=1,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        inference_time = time.time() - start_time
        print_debug(f"Inference completed in {inference_time:.2f} seconds")
        
        # Get memory usage after inference
        ram_after, gpu_after = get_memory_usage()
        print_debug(f"Memory after inference - RAM: {ram_after:.1f}MB, GPU: {gpu_after:.1f}MB")
        
        # Decode results
        print_debug("Decoding generated text...")
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Extract the generated content (remove the prompt)
        if prompt in generated_text:
            result_text = generated_text.split(prompt, 1)[1].strip()
        else:
            result_text = generated_text.strip()
        
        print_debug("OCR inference completed successfully")
        
        return {
            'success': True,
            'generated_text': result_text,
            'inference_time': inference_time,
            'ram_usage': ram_after - ram_before,
            'gpu_usage': gpu_after - gpu_before,
            'error': None
        }
        
    except Exception as e:
        print_debug(f"Error during inference: {str(e)}", "ERROR")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}", "ERROR")
        
        return {
            'success': False,
            'generated_text': None,
            'inference_time': 0,
            'ram_usage': 0,
            'gpu_usage': 0,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="OCR inference with Kosmos-2.5")
    parser.add_argument("--model_path", required=True,
                       help="Path to model (original or quantized)")
    parser.add_argument("--image_path", required=True,
                       help="Path to input image")
    parser.add_argument("--output_file", required=True,
                       help="Path to save results")
    parser.add_argument("--task_type", choices=['ocr', 'markdown'], default='markdown',
                       help="Type of task to perform")
    parser.add_argument("--is_quantized", action='store_true',
                       help="Whether the model is pre-quantized")
    
    args = parser.parse_args()
    
    print_debug("=== Kosmos-2.5 OCR Inference ===")
    print_debug(f"Model path: {args.model_path}")
    print_debug(f"Image path: {args.image_path}")
    print_debug(f"Output file: {args.output_file}")
    print_debug(f"Task type: {args.task_type}")
    print_debug(f"Is quantized: {args.is_quantized}")
    
    # Check inputs
    if not os.path.exists(args.model_path):
        print_debug(f"Model path does not exist: {args.model_path}", "ERROR")
        return 1
    
    if not os.path.exists(args.image_path):
        print_debug(f"Image path does not exist: {args.image_path}", "ERROR")
        return 1
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print_debug(f"CUDA available: {torch.cuda.get_device_name()}")
        print_debug(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print_debug("CUDA not available, using CPU")
    
    try:
        # Load model and processor
        print_debug("\n" + "="*50)
        print_debug("LOADING MODEL")
        print_debug("="*50)
        
        model, processor = load_model_and_processor(args.model_path, args.is_quantized)
        
        # Perform inference
        print_debug("\n" + "="*50)
        print_debug("PERFORMING INFERENCE")
        print_debug("="*50)
        
        result = perform_ocr_inference(model, processor, args.image_path, args.task_type)
        
        # Save results
        print_debug("\n" + "="*50)
        print_debug("SAVING RESULTS")
        print_debug("="*50)
        
        output_data = {
            'model_path': args.model_path,
            'image_path': args.image_path,
            'task_type': args.task_type,
            'is_quantized': args.is_quantized,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'result': result
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print_debug(f"Results saved to: {args.output_file}")
        
        # Print summary
        print_debug("\n" + "="*50)
        print_debug("INFERENCE SUMMARY")
        print_debug("="*50)
        
        if result['success']:
            print_debug("✅ Inference successful")
            print_debug(f"Inference time: {result['inference_time']:.2f}s")
            print_debug(f"RAM usage: {result['ram_usage']:.1f}MB")
            print_debug(f"GPU usage: {result['gpu_usage']:.1f}MB")
            print_debug(f"Generated text length: {len(result['generated_text']) if result['generated_text'] else 0} characters")
            if result['generated_text']:
                preview = result['generated_text'][:200]
                if len(result['generated_text']) > 200:
                    preview += "..."
                print_debug(f"Text preview: {preview}")
        else:
            print_debug("❌ Inference failed")
            print_debug(f"Error: {result['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        print_debug(f"Fatal error: {str(e)}", "ERROR")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}", "ERROR")
        return 1
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    sys.exit(main())