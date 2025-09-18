#!/usr/bin/env python3
"""
OCR inference script for quantized Kosmos2.5 models
Takes model location and sample image as input
"""

import torch
import argparse
import os
import sys
import gc
import psutil
import time
import json
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
import traceback

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

def load_and_preprocess_image(image_path):
    """Load and preprocess image for OCR"""
    log_debug(f"Loading image from: {image_path}")
    
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        log_debug(f"✅ Image loaded successfully: {image.size} pixels, mode: {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            log_debug(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        log_error(f"Failed to load image: {str(e)}")
        raise

def load_model_and_processor(model_path):
    """Load model and processor from local path"""
    log_debug(f"Loading model and processor from: {model_path}")
    
    memory_before = get_memory_usage()
    log_debug(f"Memory before loading: CPU {memory_before['cpu_used_gb']:.2f}GB, GPU {memory_before['gpu_memory']['allocated'] if memory_before['gpu_memory'] else 0:.2f}GB")
    
    try:
        # Check if model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for required files
        required_files = ['config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                log_error(f"Required file missing: {file}")
        
        # Load processor
        log_debug("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        log_debug("✅ Processor loaded successfully")
        
        # Load model
        log_debug("Loading model...")
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        log_debug("✅ Model loaded successfully")
        
        memory_after = get_memory_usage()
        log_debug(f"Memory after loading: CPU {memory_after['cpu_used_gb']:.2f}GB, GPU {memory_after['gpu_memory']['allocated'] if memory_after['gpu_memory'] else 0:.2f}GB")
        
        memory_increase = memory_after['cpu_used_gb'] - memory_before['cpu_used_gb']
        log_debug(f"Memory increase: {memory_increase:.2f}GB")
        
        return model, processor, {
            'before_loading': memory_before,
            'after_loading': memory_after,
            'memory_increase_gb': memory_increase
        }
    
    except Exception as e:
        log_error(f"Failed to load model: {str(e)}")
        log_error(f"Exception type: {type(e).__name__}")
        log_error(f"Traceback: {traceback.format_exc()}")
        raise

def perform_ocr_inference(model, processor, image, task_type="OCR"):
    """Perform OCR inference on the image"""
    log_debug(f"Starting {task_type} inference...")
    
    try:
        # Prepare inputs
        log_debug("Preparing model inputs...")
        
        if task_type.upper() == "OCR":
            prompt = "<ocr>"
        elif task_type.upper() == "MARKDOWN":
            prompt = "<md>"
        else:
            prompt = "<ocr>"
            log_debug(f"Unknown task type {task_type}, defaulting to OCR")
        
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        log_debug(f"✅ Inputs prepared with prompt: '{prompt}'")
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        log_debug(f"Model device: {device}")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        log_debug("✅ Inputs moved to model device")
        
        # Memory before inference
        memory_before = get_memory_usage()
        log_debug(f"Memory before inference: GPU {memory_before['gpu_memory']['allocated'] if memory_before['gpu_memory'] else 0:.2f}GB")
        
        # Perform inference
        log_debug("Generating output...")
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        inference_time = time.time() - start_time
        log_debug(f"✅ Generation completed in {inference_time:.2f} seconds")
        
        # Memory after inference
        memory_after = get_memory_usage()
        log_debug(f"Memory after inference: GPU {memory_after['gpu_memory']['allocated'] if memory_after['gpu_memory'] else 0:.2f}GB")
        
        # Decode output
        log_debug("Decoding generated text...")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the output (remove the input prompt)
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        log_debug(f"✅ Text decoded, length: {len(generated_text)} characters")
        
        return {
            'text': generated_text,
            'inference_time': inference_time,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'task_type': task_type
        }
    
    except Exception as e:
        log_error(f"Inference failed: {str(e)}")
        log_error(f"Exception type: {type(e).__name__}")
        log_error(f"Traceback: {traceback.format_exc()}")
        raise

def save_results(results, output_file):
    """Save results to JSON file"""
    log_debug(f"Saving results to: {output_file}")
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log_debug("✅ Results saved successfully")
    
    except Exception as e:
        log_error(f"Failed to save results: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Perform OCR inference with quantized Kosmos2.5 model")
    parser.add_argument("--model-path", required=True,
                       help="Path to the quantized model directory")
    parser.add_argument("--image", required=True,
                       help="Path to the input image")
    parser.add_argument("--output", required=True,
                       help="Output file to save results (JSON)")
    parser.add_argument("--task", choices=["OCR", "MARKDOWN"], default="OCR",
                       help="Task type: OCR or MARKDOWN (default: OCR)")
    
    args = parser.parse_args()
    
    log_debug(f"Starting OCR inference script")
    log_debug(f"Model path: {args.model_path}")
    log_debug(f"Image path: {args.image}")
    log_debug(f"Output file: {args.output}")
    log_debug(f"Task type: {args.task}")
    log_debug(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_debug(f"CUDA device: {torch.cuda.get_device_name()}")
        log_debug(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    try:
        # Load image
        log_debug(f"\n{'='*50}")
        log_debug("STEP 1: Loading and preprocessing image")
        log_debug(f"{'='*50}")
        image = load_and_preprocess_image(args.image)
        
        # Load model and processor
        log_debug(f"\n{'='*50}")
        log_debug("STEP 2: Loading model and processor")
        log_debug(f"{'='*50}")
        model, processor, loading_memory = load_model_and_processor(args.model_path)
        
        # Perform inference
        log_debug(f"\n{'='*50}")
        log_debug("STEP 3: Performing OCR inference")
        log_debug(f"{'='*50}")
        inference_results = perform_ocr_inference(model, processor, image, args.task)
        
        # Prepare final results
        log_debug(f"\n{'='*50}")
        log_debug("STEP 4: Preparing final results")
        log_debug(f"{'='*50}")
        
        final_results = {
            'model_path': args.model_path,
            'image_path': args.image,
            'task_type': args.task,
            'loading_memory': loading_memory,
            'inference_results': inference_results,
            'extracted_text': inference_results['text'],
            'inference_time_seconds': inference_results['inference_time'],
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'python_version': sys.version,
                'torch_version': torch.__version__,
            }
        }
        
        # Save results
        log_debug(f"\n{'='*50}")
        log_debug("STEP 5: Saving results")
        log_debug(f"{'='*50}")
        save_results(final_results, args.output)
        
        # Print summary
        log_debug(f"\n{'='*50}")
        log_debug("INFERENCE SUMMARY")
        log_debug(f"{'='*50}")
        log_debug(f"✅ Inference completed successfully")
        log_debug(f"Text length: {len(inference_results['text'])} characters")
        log_debug(f"Inference time: {inference_results['inference_time']:.2f} seconds")
        log_debug(f"Results saved to: {args.output}")
        
        # Show extracted text (first 200 chars)
        text_preview = inference_results['text'][:200] + "..." if len(inference_results['text']) > 200 else inference_results['text']
        log_debug(f"Extracted text preview: {text_preview}")
        
        # Cleanup
        log_debug("\nCleaning up memory...")
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        log_debug("✅ Memory cleanup completed")
        
    except Exception as e:
        log_error(f"Script failed: {str(e)}")
        log_error(f"Exception type: {type(e).__name__}")
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()