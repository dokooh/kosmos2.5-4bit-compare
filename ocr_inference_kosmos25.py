#!/usr/bin/env python3
"""
OCR Inference Script for Quantized Kosmos2.5 Models
Handles dtype compatibility and provides extensive debugging
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available, GPU monitoring will be limited")
from PIL import Image
from transformers import (
    Kosmos2_5ForConditionalGeneration,
    Kosmos2_5Processor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor GPU and RAM memory usage"""
    
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_percent': psutil.virtual_memory().percent,
        }
        
        if self.has_gpu:
            try:
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        memory_info.update({
                            'gpu_used_gb': gpu.memoryUsed / 1024,
                            'gpu_total_gb': gpu.memoryTotal / 1024,
                            'gpu_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                        })
                
                # Add PyTorch CUDA memory info (this doesn't require GPUtil)
                if torch.cuda.is_available():
                    memory_info.update({
                        'torch_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                        'torch_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
                    })
            except Exception as e:
                logger.warning(f"[DEBUG] ⚠️ Could not get GPU info: {e}")
                
        return memory_info

class Kosmos25Inferencer:
    """Handles OCR inference with quantized Kosmos2.5 models"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.processor = None
        self.monitor = MemoryMonitor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"[DEBUG] 🚀 Initializing Kosmos25Inferencer")
        logger.info(f"[DEBUG] 📁 Model path: {model_path}")
        logger.info(f"[DEBUG] 🖥️ Device: {self.device}")
        
    def load_model(self) -> bool:
        """Load model and processor with proper dtype handling"""
        try:
            logger.info(f"[DEBUG] 📥 Loading model from {self.model_path}")
            
            # Check if quantization metadata exists
            metadata_path = self.model_path / "quantization_metadata.json"
            is_quantized = False
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                is_quantized = True
                logger.info(f"[DEBUG] 📋 Found quantization metadata: {metadata.get('quantization_type', 'unknown')}")
            
            # Also check for quantization info file
            info_path = self.model_path / "quantization_info.txt"
            if info_path.exists():
                is_quantized = True
                logger.info(f"[DEBUG] 📋 Found quantization info file")
                
            logger.info(f"[DEBUG] 🔍 Model appears to be quantized: {is_quantized}")
            
            # Memory before loading
            mem_before = self.monitor.get_memory_info()
            logger.info(f"[DEBUG] 📊 Memory before loading: {mem_before}")
            
            # Load processor first
            logger.info(f"[DEBUG] 🔄 Loading processor...")
            self.processor = Kosmos2_5Processor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Load model with careful dtype handling
            logger.info(f"[DEBUG] 🔄 Loading model...")
            
            # Try different loading strategies based on model type
            if is_quantized:
                # For quantized models, avoid torch_dtype to prevent casting errors
                load_strategies = [
                    {
                        "name": "quantized_auto",
                        "kwargs": {
                            "device_map": "auto",
                            "trust_remote_code": True
                        }
                    },
                    {
                        "name": "quantized_explicit_device",
                        "kwargs": {
                            "device_map": {"": self.device},
                            "trust_remote_code": True
                        }
                    },
                    {
                        "name": "quantized_cpu_fallback",
                        "kwargs": {
                            "device_map": "cpu",
                            "trust_remote_code": True
                        }
                    }
                ]
            else:
                # For non-quantized models, try with explicit dtype
                load_strategies = [
                    {
                        "name": "standard_float16",
                        "kwargs": {
                            "device_map": "auto",
                            "trust_remote_code": True,
                            "torch_dtype": torch.float16
                        }
                    },
                    {
                        "name": "standard_auto_dtype",
                        "kwargs": {
                            "device_map": "auto",
                            "trust_remote_code": True
                        }
                    }
                ]
            
            for strategy in load_strategies:
                try:
                    logger.info(f"[DEBUG] 🔄 Trying loading strategy: {strategy['name']}")
                    self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                        str(self.model_path),
                        **strategy["kwargs"]
                    )
                    logger.info(f"[DEBUG] ✅ Model loaded successfully with strategy: {strategy['name']}")
                    break
                except Exception as e:
                    logger.warning(f"[DEBUG] ⚠️ Strategy {strategy['name']} failed: {e}")
                    continue
            
            if self.model is None:
                raise RuntimeError("All loading strategies failed")
            
            # Memory after loading
            mem_after = self.monitor.get_memory_info()
            logger.info(f"[DEBUG] 📊 Memory after loading: {mem_after}")
            
            # Model info
            logger.info(f"[DEBUG] 🔍 Model device: {next(self.model.parameters()).device}")
            logger.info(f"[DEBUG] 🔍 Model dtype: {next(self.model.parameters()).dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"[DEBUG] ❌ Failed to load model: {e}")
            logger.error(f"[DEBUG] 📋 Traceback: {traceback.format_exc()}")
            return False
    
    def prepare_image(self, image_path: str) -> Image.Image:
        """Load and prepare image for inference"""
        logger.info(f"[DEBUG] 🖼️ Loading image: {image_path}")
        
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                logger.info(f"[DEBUG] 🎨 Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            logger.info(f"[DEBUG] 📐 Image size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"[DEBUG] ❌ Failed to load image: {e}")
            raise
    
    def perform_ocr_inference(self, image: Image.Image, task: str = "ocr") -> Dict[str, Any]:
        """Perform OCR or Markdown inference"""
        logger.info(f"[DEBUG] 🎯 Starting {task} inference")
        start_time = time.time()
        
        try:
            # Memory before inference
            mem_before = self.monitor.get_memory_info()
            logger.info(f"[DEBUG] 📊 Memory before inference: {mem_before}")
            
            # Prepare inputs
            if task.lower() == "ocr":
                prompt = "<ocr>"
            elif task.lower() in ["markdown", "md"]:
                prompt = "<md>"
            else:
                prompt = "<ocr>"  # Default to OCR
                
            logger.info(f"[DEBUG] 📝 Using prompt: '{prompt}'")
            
            # Process inputs
            logger.info(f"[DEBUG] 🔄 Processing inputs...")
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Move to device
            logger.info(f"[DEBUG] 📤 Moving inputs to device: {self.device}")
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
                    logger.info(f"[DEBUG] 🔍 {key} shape: {inputs[key].shape}, dtype: {inputs[key].dtype}")
            
            # Generate
            logger.info(f"[DEBUG] 🚀 Starting generation...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode
            logger.info(f"[DEBUG] 🔄 Decoding output...")
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )[0]
            
            inference_time = time.time() - start_time
            
            # Memory after inference
            mem_after = self.monitor.get_memory_info()
            logger.info(f"[DEBUG] 📊 Memory after inference: {mem_after}")
            
            result = {
                "success": True,
                "task": task,
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time,
                "memory_usage": {
                    "before": mem_before,
                    "after": mem_after
                }
            }
            
            logger.info(f"[DEBUG] ✅ Inference completed in {inference_time:.2f}s")
            logger.info(f"[DEBUG] 📝 Generated text length: {len(generated_text)} characters")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "task": task,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "inference_time": time.time() - start_time
            }
            
            logger.error(f"[DEBUG] ❌ Inference failed: {e}")
            logger.error(f"[DEBUG] 📋 Traceback: {traceback.format_exc()}")
            
            return error_result
    
    def cleanup(self):
        """Clean up resources"""
        logger.info(f"[DEBUG] 🧹 Cleaning up resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[DEBUG] 🧹 CUDA cache cleared")

def main():
    parser = argparse.ArgumentParser(description="Perform OCR inference with quantized Kosmos2.5 model")
    parser.add_argument("model_path", help="Path to the quantized model directory")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--task", choices=["ocr", "markdown", "md"], default="ocr", 
                       help="Task type (ocr or markdown)")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    logger.info(f"[DEBUG] 🎬 Script started with args: {vars(args)}")
    
    # Validate inputs
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"[DEBUG] ❌ Model path does not exist: {model_path}")
        sys.exit(1)
        
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"[DEBUG] ❌ Image path does not exist: {image_path}")
        sys.exit(1)
    
    # Initialize inferencer
    inferencer = Kosmos25Inferencer(str(model_path))
    
    try:
        # Load model
        if not inferencer.load_model():
            logger.error(f"[DEBUG] ❌ Failed to load model")
            sys.exit(1)
        
        # Load image
        image = inferencer.prepare_image(str(image_path))
        
        # Perform inference
        result = inferencer.perform_ocr_inference(image, args.task)
        
        # Add metadata
        result.update({
            "model_path": str(model_path),
            "image_path": str(image_path),
            "timestamp": time.time()
        })
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"[DEBUG] 💾 Results saved to: {output_path}")
        else:
            print(json.dumps(result, indent=2))
        
        # Print summary
        if result["success"]:
            logger.info(f"[DEBUG] 🎉 Inference successful!")
            logger.info(f"[DEBUG] 📝 Generated text preview: {result['generated_text'][:100]}...")
        else:
            logger.error(f"[DEBUG] ❌ Inference failed: {result['error']}")
            
    finally:
        inferencer.cleanup()
        logger.info(f"[DEBUG] 🏁 Script completed")

if __name__ == "__main__":
    main()
