#!/usr/bin/env python3
"""
Kosmos-2.5 4-bit Quantization Testing Script

This script tests multiple 4-bit quantization strategies for the Hugging Face Kosmos-2.5 model,
evaluating OCR/Markdown tasks on locally provided images while monitoring memory usage.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import psutil
import torch
from PIL import Image
from transformers import (
    AutoProcessor, 
    Kosmos2_5ForConditionalGeneration,
    BitsAndBytesConfig
)
import gc

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU memory monitoring will be limited.")

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. Advanced GPU monitoring will be limited.")


class MemoryMonitor:
    """Monitor GPU and RAM memory usage."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "total": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        
        gpu_info = {"allocated": allocated, "cached": cached, "total": 0, "free": 0}
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info["total"] = gpu.memoryTotal
                    gpu_info["free"] = gpu.memoryFree
            except Exception as e:
                logging.warning(f"Error getting GPU info with GPUtil: {e}")
        
        return gpu_info
    
    def get_ram_usage(self) -> Dict[str, float]:
        """Get RAM usage in MB."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / 1024**2,
            "available": memory.available / 1024**2,
            "used": memory.used / 1024**2,
            "percent": memory.percent
        }
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class QuantizationTester:
    """Test different quantization strategies for Kosmos-2.5."""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2.5"):
        self.model_name = model_name
        self.monitor = MemoryMonitor()
        self.processor = None
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_processor(self):
        """Load the processor once to reuse across tests."""
        if self.processor is None:
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def get_quantization_configs(self) -> Dict[str, Optional[BitsAndBytesConfig]]:
        """Define different quantization configurations to test."""
        configs = {
            "baseline": None,  # No quantization
            "nf4_fp16": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            ),
            "nf4_bf16": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            ),
            "fp4_fp16": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            ),
            "fp4_bf16": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            ),
            "nf4_fp16_single": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )
        }
        return configs
    
    def load_model(self, config_name: str, quantization_config: Optional[BitsAndBytesConfig]) -> tuple:
        """Load model with specified quantization configuration."""
        self.logger.info(f"Loading model with {config_name} configuration...")
        
        start_time = time.time()
        
        try:
            if quantization_config is None:
                # Baseline model without quantization
                model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Quantized model
                model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            
            load_time = time.time() - start_time
            
            # Get memory usage after loading
            gpu_memory = self.monitor.get_gpu_memory()
            ram_usage = self.monitor.get_ram_usage()
            
            self.logger.info(f"Model loaded in {load_time:.2f}s")
            self.logger.info(f"GPU Memory - Allocated: {gpu_memory['allocated']:.1f}MB, Cached: {gpu_memory['cached']:.1f}MB")
            
            return model, load_time, gpu_memory, ram_usage
            
        except Exception as e:
            self.logger.error(f"Error loading model with {config_name}: {str(e)}")
            return None, 0, {}, {}
    
    def test_ocr_task(self, model, image_path: str) -> Dict[str, Any]:
        """Test OCR/Markdown generation on the provided image."""
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Prepare inputs
            prompt = "<ocr>"  # OCR task prompt for Kosmos-2.5
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Move inputs to the same device as model
            if hasattr(model, 'device'):
                device = model.device
            else:
                device = next(model.parameters()).device
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            inference_time = time.time() - start_time
            
            # Decode output
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract the generated content (remove the prompt)
            if prompt in generated_text:
                result_text = generated_text.replace(prompt, "").strip()
            else:
                result_text = generated_text.strip()
            
            return {
                "success": True,
                "inference_time": inference_time,
                "generated_text": result_text,
                "text_length": len(result_text),
                "error": None
            }
            
        except Exception as e:
            self.logger.error(f"Error during OCR task: {str(e)}")
            return {
                "success": False,
                "inference_time": 0,
                "generated_text": "",
                "text_length": 0,
                "error": str(e)
            }
    
    def run_test(self, config_name: str, quantization_config: Optional[BitsAndBytesConfig], 
                image_path: str) -> Dict[str, Any]:
        """Run a complete test for one quantization configuration."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Testing configuration: {config_name}")
        self.logger.info(f"{'='*50}")
        
        # Clear memory before loading
        self.monitor.clear_cache()
        
        # Record initial memory state
        initial_gpu = self.monitor.get_gpu_memory()
        initial_ram = self.monitor.get_ram_usage()
        
        # Load model
        model, load_time, post_load_gpu, post_load_ram = self.load_model(config_name, quantization_config)
        
        if model is None:
            return {
                "config_name": config_name,
                "success": False,
                "error": "Failed to load model",
                "load_time": 0,
                "inference_time": 0,
                "memory_usage": {}
            }
        
        # Test OCR task
        ocr_results = self.test_ocr_task(model, image_path)
        
        # Get final memory usage
        final_gpu = self.monitor.get_gpu_memory()
        final_ram = self.monitor.get_ram_usage()
        
        # Calculate model size (approximate)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # Compile results
        result = {
            "config_name": config_name,
            "success": ocr_results["success"],
            "timestamp": datetime.now().isoformat(),
            "model_size_mb": model_size_mb,
            "load_time": load_time,
            "inference_time": ocr_results["inference_time"],
            "generated_text_length": ocr_results["text_length"],
            "generated_text": ocr_results["generated_text"][:500] + "..." if len(ocr_results["generated_text"]) > 500 else ocr_results["generated_text"],
            "error": ocr_results["error"],
            "memory_usage": {
                "initial_gpu": initial_gpu,
                "post_load_gpu": post_load_gpu,
                "final_gpu": final_gpu,
                "initial_ram": initial_ram,
                "post_load_ram": post_load_ram,
                "final_ram": final_ram,
                "gpu_memory_increase": post_load_gpu["allocated"] - initial_gpu["allocated"],
                "ram_memory_increase": post_load_ram["used"] - initial_ram["used"]
            }
        }
        
        # Cleanup
        del model
        self.monitor.clear_cache()
        
        self.logger.info(f"Test completed for {config_name}")
        if ocr_results["success"]:
            self.logger.info(f"Inference time: {ocr_results['inference_time']:.2f}s")
            self.logger.info(f"Generated text length: {ocr_results['text_length']} characters")
        
        return result
    
    def run_all_tests(self, image_path: str) -> List[Dict[str, Any]]:
        """Run tests for all quantization configurations."""
        self.logger.info("Starting comprehensive quantization tests...")
        
        # Load processor once
        self.load_processor()
        
        # Get all configurations
        configs = self.get_quantization_configs()
        
        results = []
        for config_name, quantization_config in configs.items():
            try:
                result = self.run_test(config_name, quantization_config, image_path)
                results.append(result)
                
                # Add small delay between tests
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Critical error testing {config_name}: {str(e)}")
                results.append({
                    "config_name": config_name,
                    "success": False,
                    "error": f"Critical error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save test results to file."""
        try:
            # Prepare summary data
            summary = {
                "test_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "total_configurations": len(results),
                    "successful_tests": sum(1 for r in results if r.get("success", False)),
                    "cuda_available": torch.cuda.is_available(),
                    "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
                },
                "results": results,
                "performance_comparison": self._generate_performance_comparison(results)
            }
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {output_path}")
            
            # Print summary to console
            self._print_summary(results)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def _generate_performance_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance comparison metrics."""
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful tests to compare"}
        
        comparison = {
            "fastest_inference": min(successful_results, key=lambda x: x["inference_time"]),
            "slowest_inference": max(successful_results, key=lambda x: x["inference_time"]),
            "lowest_memory": min(successful_results, key=lambda x: x["memory_usage"]["gpu_memory_increase"]),
            "highest_memory": max(successful_results, key=lambda x: x["memory_usage"]["gpu_memory_increase"]),
            "smallest_model": min(successful_results, key=lambda x: x["model_size_mb"]),
            "largest_model": max(successful_results, key=lambda x: x["model_size_mb"])
        }
        
        return comparison
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of test results to console."""
        print(f"\n{'='*60}")
        print("KOSMOS-2.5 QUANTIZATION TEST SUMMARY")
        print(f"{'='*60}")
        
        successful_results = [r for r in results if r.get("success", False)]
        
        print(f"Total configurations tested: {len(results)}")
        print(f"Successful tests: {len(successful_results)}")
        print(f"Failed tests: {len(results) - len(successful_results)}")
        
        if successful_results:
            print(f"\n{'Configuration':<20} {'Load Time':<12} {'Inference':<12} {'GPU Mem':<12} {'Model Size':<12}")
            print(f"{'Name':<20} {'(seconds)':<12} {'(seconds)':<12} {'(MB)':<12} {'(MB)':<12}")
            print("-" * 68)
            
            for result in successful_results:
                print(f"{result['config_name']:<20} "
                      f"{result['load_time']:<12.2f} "
                      f"{result['inference_time']:<12.2f} "
                      f"{result['memory_usage']['gpu_memory_increase']:<12.1f} "
                      f"{result['model_size_mb']:<12.1f}")
        
        print(f"\n{'='*60}")


def main():
    """Main function to run the quantization tests."""
    parser = argparse.ArgumentParser(
        description="Test 4-bit quantization strategies for Kosmos-2.5 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kosmos_quantization_test.py -i image.jpg -o results.json
  python kosmos_quantization_test.py --input /path/to/image.png --output /path/to/results.json
  python kosmos_quantization_test.py -i image.jpg -o results.json --model microsoft/kosmos-2.5
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input image file for OCR testing"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output JSON file for saving results"
    )
    
    parser.add_argument(
        "--model",
        default="microsoft/kosmos-2.5",
        help="Model name or path (default: microsoft/kosmos-2.5)"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu) (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input image file '{args.input}' does not exist.")
        sys.exit(1)
    
    # Validate input is an image file
    try:
        with Image.open(args.input) as img:
            img.verify()
    except Exception as e:
        print(f"Error: '{args.input}' is not a valid image file: {str(e)}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available. Using CPU (this will be slow).")
    
    # Initialize tester and run tests
    tester = QuantizationTester(model_name=args.model)
    
    try:
        results = tester.run_all_tests(args.input)
        tester.save_results(results, args.output)
        
        print(f"\nTesting completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
