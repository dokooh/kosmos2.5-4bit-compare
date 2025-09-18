#!/usr/bin/env python3
"""
OCR Inference Script for Kosmos-2.5 Model
Performs OCR/Markdown task on images using locally saved quantized models.
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import psutil
import GPUtil
from PIL import Image
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

print(f"[DEBUG] Inference script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] PyTorch version: {torch.__version__}")
print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[DEBUG] CUDA version: {torch.version.cuda}")
    print(f"[DEBUG] GPU count: {torch.cuda.device_count()}")

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

def load_and_preprocess_image(image_path: str) -> Optional[Image.Image]:
    """Load and preprocess image for OCR"""
    print(f"[DEBUG] Loading image from: {image_path}")
    
    try:
        image = Image.open(image_path)
        print(f"[DEBUG] Image loaded - Size: {image.size}, Mode: {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            print(f"[DEBUG] Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        print(f"[DEBUG] Final image - Size: {image.size}, Mode: {image.mode}")
        return image
        
    except Exception as e:
        print(f"[ERROR] Error loading image: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return None

def perform_ocr_inference(model_path: str, image_path: str, task: str = "ocr") -> Dict[str, Any]:
    """Perform OCR inference using the quantized model"""
    print(f"\n[DEBUG] ========== Starting OCR inference ==========")
    print(f"[DEBUG] Model path: {model_path}")
    print(f"[DEBUG] Image path: {image_path}")
    print(f"[DEBUG] Task: {task}")
    
    start_time = time.time()
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[DEBUG] Cleared GPU cache")
        
        # Get initial memory usage
        print("[DEBUG] Getting initial memory usage...")
        initial_memory = get_memory_usage()
        
        # Load image
        print("[DEBUG] Loading and preprocessing image...")
        image = load_and_preprocess_image(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        # Load processor
        print("[DEBUG] Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        print("[DEBUG] Processor loaded successfully")
        
        # Load model
        print("[DEBUG] Loading quantized model...")
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("[DEBUG] Model loaded successfully")
        
        # Get memory usage after loading
        print("[DEBUG] Getting memory usage after model loading...")
        loaded_memory = get_memory_usage()
        
        # Prepare prompt based on task
        task_prompts = {
            "ocr": "<ocr>",  # Standard OCR task
            "markdown": "<md>",  # Markdown conversion task
            "text_detection": "<detect>",  # Text detection
            "caption": "<cap>",  # Image captioning
        }
        
        prompt = task_prompts.get(task, "<ocr>")
        print(f"[DEBUG] Using prompt: {prompt}")
        
        # Process inputs
        print("[DEBUG] Processing inputs...")
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        print(f"[DEBUG] Input keys: {list(inputs.keys())}")
        print(f"[DEBUG] Input shapes: {{k: v.shape if hasattr(v, 'shape') else type(v) for k, v in inputs.items()}}")
        
        # Move inputs to appropriate device
        if torch.cuda.is_available():
            device = next(model.parameters()).device
            print(f"[DEBUG] Moving inputs to device: {device}")
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        print("[DEBUG] Starting generation...")
        generation_start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        generation_time = time.time() - generation_start
        print(f"[DEBUG] Generation completed in {generation_time:.2f} seconds")
        print(f"[DEBUG] Generated IDs shape: {generated_ids.shape}")
        
        # Decode output
        print("[DEBUG] Decoding generated text...")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the output (remove the input prompt)
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "", 1).strip()
            print("[DEBUG] Removed prompt from generated text")
        
        print(f"[DEBUG] Generated text length: {len(generated_text)} characters")
        print(f"[DEBUG] Generated text preview: {generated_text[:200]}...")
        
        # Get final memory usage
        print("[DEBUG] Getting final memory usage...")
        final_memory = get_memory_usage()
        
        # Calculate memory differences
        ram_diff = loaded_memory['ram_used_gb'] - initial_memory['ram_used_gb']
        gpu_diff = 0
        if 'gpu_used_mb' in loaded_memory and 'gpu_used_mb' in initial_memory:
            gpu_diff = loaded_memory['gpu_used_mb'] - initial_memory['gpu_used_mb']
        
        end_time = time.time()
        total_time = end_time - start_time
        
        result = {
            'success': True,
            'model_path': model_path,
            'image_path': image_path,
            'task': task,
            'prompt': prompt,
            'generated_text': generated_text,
            'text_length': len(generated_text),
            'timing': {
                'total_time_seconds': total_time,
                'generation_time_seconds': generation_time,
                'loading_time_seconds': total_time - generation_time
            },
            'memory_usage': {
                'initial': initial_memory,
                'loaded': loaded_memory,
                'final': final_memory,
                'ram_increase_gb': ram_diff,
                'gpu_increase_mb': gpu_diff
            },
            'image_info': {
                'size': image.size,
                'mode': image.mode
            }
        }
        
        print(f"[DEBUG] Inference completed successfully in {total_time:.2f} seconds")
        print(f"[DEBUG] Generation took {generation_time:.2f} seconds")
        print(f"[DEBUG] Generated {len(generated_text)} characters")
        
        # Clear memory
        del model, processor, inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[DEBUG] Cleared models from memory")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Error during inference: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        end_time = time.time()
        return {
            'success': False,
            'model_path': model_path,
            'image_path': image_path,
            'task': task,
            'error': str(e),
            'total_time_seconds': end_time - start_time,
            'traceback': traceback.format_exc()
        }

def find_quantized_models(base_dir: str) -> List[str]:
    """Find all quantized model directories"""
    print(f"[DEBUG] Searching for quantized models in: {base_dir}")
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[ERROR] Base directory does not exist: {base_dir}")
        return []
    
    model_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('kosmos25_'):
            # Check if it contains model files
            config_file = item / 'config.json'
            if config_file.exists():
                model_dirs.append(str(item))
                print(f"[DEBUG] Found quantized model: {item.name}")
    
    print(f"[DEBUG] Found {len(model_dirs)} quantized models")
    return sorted(model_dirs)

def main():
    parser = argparse.ArgumentParser(description='Perform OCR inference with quantized Kosmos-2.5 models')
    parser.add_argument('--model-path', '-m', type=str, required=True,
                        help='Path to the quantized model directory or base directory containing multiple models')
    parser.add_argument('--image-path', '-i', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output-file', '-o', type=str, default='ocr_results.json',
                        help='File to save results (default: ocr_results.json)')
    parser.add_argument('--task', '-t', type=str, choices=['ocr', 'markdown', 'text_detection', 'caption'],
                        default='ocr', help='Task to perform (default: ocr)')
    parser.add_argument('--test-all-models', action='store_true',
                        help='Test all quantized models found in the base directory')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Arguments parsed:")
    print(f"[DEBUG] - Model path: {args.model_path}")
    print(f"[DEBUG] - Image path: {args.image_path}")
    print(f"[DEBUG] - Output file: {args.output_file}")
    print(f"[DEBUG] - Task: {args.task}")
    print(f"[DEBUG] - Test all models: {args.test_all_models}")
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"[ERROR] Image file does not exist: {args.image_path}")
        sys.exit(1)
    
    # Determine models to test
    models_to_test = []
    
    if args.test_all_models:
        print("[DEBUG] Searching for all quantized models...")
        models_to_test = find_quantized_models(args.model_path)
        if not models_to_test:
            print(f"[ERROR] No quantized models found in: {args.model_path}")
            sys.exit(1)
    else:
        if not Path(args.model_path).exists():
            print(f"[ERROR] Model path does not exist: {args.model_path}")
            sys.exit(1)
        models_to_test = [args.model_path]
    
    print(f"[DEBUG] Will test {len(models_to_test)} model(s)")
    
    # Perform inference on all models
    all_results = []
    
    print(f"\n[DEBUG] ========== STARTING INFERENCE PROCESS ==========")
    
    for i, model_path in enumerate(models_to_test, 1):
        model_name = Path(model_path).name
        print(f"\n[DEBUG] ========== MODEL {i}/{len(models_to_test)}: {model_name} ==========")
        
        result = perform_ocr_inference(model_path, args.image_path, args.task)
        result['model_name'] = model_name
        all_results.append(result)
        
        if result['success']:
            print(f"[DEBUG] ✅ {model_name} inference completed successfully")
            print(f"[DEBUG] - Total time: {result['timing']['total_time_seconds']:.2f}s")
            print(f"[DEBUG] - Generation time: {result['timing']['generation_time_seconds']:.2f}s")
            print(f"[DEBUG] - Text length: {result['text_length']} characters")
        else:
            print(f"[DEBUG] ❌ {model_name} inference failed: {result['error']}")
    
    # Save results to file
    results_file = Path(args.output_file)
    print(f"\n[DEBUG] Saving results to {results_file}")
    
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'image_path': args.image_path,
        'task': args.task,
        'models_tested': len(models_to_test),
        'successful_inferences': len([r for r in all_results if r['success']]),
        'failed_inferences': len([r for r in all_results if not r['success']]),
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
    print(f"\n[DEBUG] ========== INFERENCE SUMMARY ==========")
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"[DEBUG] Total models tested: {len(all_results)}")
    print(f"[DEBUG] Successful: {len(successful)}")
    print(f"[DEBUG] Failed: {len(failed)}")
    
    if successful:
        print(f"\n[DEBUG] Successful inferences:")
        for result in successful:
            print(f"[DEBUG] - {result['model_name']}: {result['text_length']} chars in {result['timing']['total_time_seconds']:.1f}s")
    
    if failed:
        print(f"\n[DEBUG] Failed inferences:")
        for result in failed:
            print(f"[DEBUG] - {result['model_name']}: {result['error']}")
    
    print(f"\n[DEBUG] Inference script completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()