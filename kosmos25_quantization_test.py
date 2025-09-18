#!/usr/bin/env python3
"""
Kosmos 2.5 4-bit Quantization Testing Script
Tests various 4-bit quantization strategies on Kosmos 2.5 model for OCR/Markdown tasks
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoTokenizer, 
    Kosmos2_5ForConditionalGeneration,
    BitsAndBytesConfig
)
import gc

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kosmos2_5_quantization_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor GPU and RAM memory usage"""
    
    def __init__(self):
        logger.debug("Initializing MemoryMonitor")
        self.gpu_available = torch.cuda.is_available()
        logger.debug(f"GPU available: {self.gpu_available}")
        if self.gpu_available:
            logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.debug(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        logger.debug("Getting memory statistics")
        
        # RAM usage
        ram = psutil.virtual_memory()
        ram_stats = {
            'ram_total_gb': round(ram.total / (1024**3), 2),
            'ram_used_gb': round(ram.used / (1024**3), 2),
            'ram_available_gb': round(ram.available / (1024**3), 2),
            'ram_percent': ram.percent
        }
        
        # GPU usage
        gpu_stats = {}
        if self.gpu_available:
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    gpu_stats[f'gpu_{i}_allocated_gb'] = round(allocated, 2)
                    gpu_stats[f'gpu_{i}_reserved_gb'] = round(reserved, 2)
                    
                    # Get total memory
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_stats[f'gpu_{i}_total_gb'] = round(total_memory, 2)
                    gpu_stats[f'gpu_{i}_percent'] = round((allocated / total_memory) * 100, 2)
                    
                    logger.debug(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total_memory:.2f}GB")
            except Exception as e:
                logger.error(f"Error getting GPU memory stats: {e}")
                gpu_stats['gpu_error'] = str(e)
        
        logger.debug(f"RAM stats: {ram_stats}")
        logger.debug(f"GPU stats: {gpu_stats}")
        
        return {**ram_stats, **gpu_stats}

class Kosmos25QuantizationTester:
    """Test different 4-bit quantization strategies for Kosmos 2.5"""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2.5"):
        logger.debug(f"Initializing Kosmos25QuantizationTester with model: {model_name}")
        self.model_name = model_name
        self.memory_monitor = MemoryMonitor()
        self.results = []
        
        # Define quantization strategies
        self.quantization_strategies = {
            'nf4': {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': True
            },
            'fp4': {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'fp4',
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': True
            },
            'nf4_no_double_quant': {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': False
            },
            'fp4_no_double_quant': {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'fp4',
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': False
            }
        }
        
        logger.debug(f"Defined {len(self.quantization_strategies)} quantization strategies")
        for name, config in self.quantization_strategies.items():
            logger.debug(f"Strategy '{name}': {config}")
    
    def cleanup_model(self):
        """Clean up model and free memory"""
        logger.debug("Starting model cleanup")
        
        if hasattr(self, 'model') and self.model is not None:
            logger.debug("Deleting model")
            del self.model
            self.model = None
        
        if hasattr(self, 'processor') and self.processor is not None:
            logger.debug("Deleting processor")
            del self.processor
            self.processor = None
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            logger.debug("Deleting tokenizer")
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        logger.debug("Running garbage collection")
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            logger.debug("Clearing GPU cache")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.debug("Model cleanup completed")
    
    def load_model_with_quantization(self, strategy_name: str, config: Dict) -> Tuple[bool, str, Dict]:
        """Load model with specific quantization configuration"""
        logger.debug(f"Loading model with quantization strategy: {strategy_name}")
        logger.debug(f"Quantization config: {config}")
        
        try:
            # Get memory before loading
            memory_before = self.memory_monitor.get_memory_stats()
            logger.debug(f"Memory before loading: {memory_before}")
            
            start_time = time.time()
            
            # Create quantization config
            logger.debug("Creating BitsAndBytesConfig")
            quantization_config = BitsAndBytesConfig(**config)
            
            # Load processor
            logger.debug("Loading image processor")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Load tokenizer
            logger.debug("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with quantization
            logger.debug("Loading quantized model")
            self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            load_time = time.time() - start_time
            logger.debug(f"Model loading completed in {load_time:.2f} seconds")
            
            # Get memory after loading
            memory_after = self.memory_monitor.get_memory_stats()
            logger.debug(f"Memory after loading: {memory_after}")
            
            # Calculate memory usage
            memory_usage = {}
            for key in memory_before:
                if key in memory_after:
                    if 'gb' in key and key != 'ram_total_gb':
                        memory_usage[f"{key}_diff"] = round(memory_after[key] - memory_before[key], 2)
            
            logger.debug(f"Memory usage difference: {memory_usage}")
            
            return True, "Success", {
                'load_time_seconds': round(load_time, 2),
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_usage': memory_usage
            }
            
        except Exception as e:
            error_msg = f"Error loading model with {strategy_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, error_msg, {}
    
    def test_ocr_inference(self, image_path: str) -> Tuple[bool, str, Dict]:
        """Test OCR/Markdown inference on the loaded model"""
        logger.debug(f"Testing OCR inference with image: {image_path}")
        
        try:
            # Load and validate image
            logger.debug("Loading image")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = Image.open(image_path)
            logger.debug(f"Image loaded: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                logger.debug(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Get memory before inference
            memory_before = self.memory_monitor.get_memory_stats()
            logger.debug(f"Memory before inference: {memory_before}")
            
            start_time = time.time()
            
            # Prepare inputs
            logger.debug("Processing image inputs")
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move to GPU if available
            device = next(self.model.parameters()).device
            logger.debug(f"Model device: {device}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate OCR output
            logger.debug("Starting model inference")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            logger.debug("Decoding generated tokens")
            generated_text = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time:.2f} seconds")
            
            # Get memory after inference
            memory_after = self.memory_monitor.get_memory_stats()
            logger.debug(f"Memory after inference: {memory_after}")
            
            # Calculate memory usage during inference
            memory_usage = {}
            for key in memory_before:
                if key in memory_after:
                    if 'gb' in key and key != 'ram_total_gb':
                        memory_usage[f"{key}_diff"] = round(memory_after[key] - memory_before[key], 2)
            
            logger.debug(f"Generated text length: {len(generated_text)} characters")
            logger.debug(f"Generated text preview: {generated_text[:200]}...")
            
            return True, "Success", {
                'inference_time_seconds': round(inference_time, 2),
                'generated_text': generated_text,
                'generated_text_length': len(generated_text),
                'memory_before_inference': memory_before,
                'memory_after_inference': memory_after,
                'memory_usage_inference': memory_usage
            }
            
        except Exception as e:
            error_msg = f"Error during OCR inference: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, error_msg, {}
    
    def test_strategy(self, strategy_name: str, image_path: str) -> Dict:
        """Test a single quantization strategy"""
        logger.info(f"Starting test for strategy: {strategy_name}")
        
        result = {
            'strategy_name': strategy_name,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error_message': None,
            'model_loading': {},
            'ocr_inference': {}
        }
        
        try:
            # Clean up any previous model
            self.cleanup_model()
            
            # Load model with quantization
            logger.debug(f"Step 1: Loading model with {strategy_name} quantization")
            success, message, loading_stats = self.load_model_with_quantization(
                strategy_name, 
                self.quantization_strategies[strategy_name]
            )
            
            result['model_loading'] = {
                'success': success,
                'message': message,
                **loading_stats
            }
            
            if not success:
                result['error_message'] = f"Model loading failed: {message}"
                logger.error(f"Strategy {strategy_name} failed at model loading: {message}")
                return result
            
            logger.info(f"Model loaded successfully for {strategy_name}")
            
            # Test OCR inference
            logger.debug(f"Step 2: Testing OCR inference for {strategy_name}")
            success, message, inference_stats = self.test_ocr_inference(image_path)
            
            result['ocr_inference'] = {
                'success': success,
                'message': message,
                **inference_stats
            }
            
            if not success:
                result['error_message'] = f"OCR inference failed: {message}"
                logger.error(f"Strategy {strategy_name} failed at OCR inference: {message}")
                return result
            
            logger.info(f"OCR inference completed successfully for {strategy_name}")
            result['success'] = True
            
        except Exception as e:
            error_msg = f"Unexpected error testing {strategy_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            result['error_message'] = error_msg
        
        finally:
            # Always clean up after each test
            logger.debug(f"Cleaning up after {strategy_name} test")
            self.cleanup_model()
        
        logger.info(f"Completed test for strategy: {strategy_name}, Success: {result['success']}")
        return result
    
    def run_all_tests(self, image_path: str) -> List[Dict]:
        """Run tests for all quantization strategies"""
        logger.info("Starting comprehensive quantization testing")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Strategies to test: {list(self.quantization_strategies.keys())}")
        
        results = []
        
        for strategy_name in self.quantization_strategies.keys():
            logger.info(f"Testing strategy {len(results) + 1}/{len(self.quantization_strategies)}: {strategy_name}")
            
            result = self.test_strategy(strategy_name, image_path)
            results.append(result)
            
            # Log summary for this strategy
            if result['success']:
                load_time = result['model_loading'].get('load_time_seconds', 'N/A')
                inference_time = result['ocr_inference'].get('inference_time_seconds', 'N/A')
                logger.info(f"Strategy {strategy_name} - Load: {load_time}s, Inference: {inference_time}s")
            else:
                logger.warning(f"Strategy {strategy_name} failed: {result['error_message']}")
        
        logger.info("All quantization tests completed")
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save test results to JSON file"""
        logger.debug(f"Saving results to: {output_path}")
        
        try:
            # Add metadata
            final_results = {
                'metadata': {
                    'test_timestamp': datetime.now().isoformat(),
                    'model_name': self.model_name,
                    'total_strategies_tested': len(results),
                    'successful_strategies': sum(1 for r in results if r['success']),
                    'python_version': sys.version,
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
                },
                'results': results
            }
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_path}")
            
            # Print summary
            self.print_summary(results)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of test results"""
        logger.info("\n" + "="*60)
        logger.info("QUANTIZATION TEST SUMMARY")
        logger.info("="*60)
        
        for result in results:
            strategy = result['strategy_name']
            success = result['success']
            
            if success:
                load_time = result['model_loading'].get('load_time_seconds', 'N/A')
                inference_time = result['ocr_inference'].get('inference_time_seconds', 'N/A')
                
                # Memory usage
                loading_memory = result['model_loading'].get('memory_usage', {})
                gpu_usage = [f"{k}: {v}GB" for k, v in loading_memory.items() if 'gpu' in k and 'diff' in k and v > 0]
                ram_usage = [f"{k}: {v}GB" for k, v in loading_memory.items() if 'ram' in k and 'diff' in k and v > 0]
                
                logger.info(f"\n✓ {strategy}:")
                logger.info(f"  Load Time: {load_time}s")
                logger.info(f"  Inference Time: {inference_time}s")
                if gpu_usage:
                    logger.info(f"  GPU Usage: {', '.join(gpu_usage)}")
                if ram_usage:
                    logger.info(f"  RAM Usage: {', '.join(ram_usage)}")
            else:
                logger.info(f"\n✗ {strategy}: FAILED")
                logger.info(f"  Error: {result['error_message']}")
        
        logger.info("\n" + "="*60)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test Kosmos 2.5 4-bit quantization strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kosmos2_5_quantization_test.py test_image.jpg results.json
  python kosmos2_5_quantization_test.py /path/to/image.png /path/to/output.json
  python kosmos2_5_quantization_test.py image.jpg results.json --model microsoft/kosmos-2.5
        """
    )
    
    parser.add_argument(
        'input_image',
        help='Path to input image for OCR testing'
    )
    
    parser.add_argument(
        'output_file',
        help='Path to output JSON file for results'
    )
    
    parser.add_argument(
        '--model',
        default='microsoft/kosmos-2.5',
        help='Hugging Face model name (default: microsoft/kosmos-2.5)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info("Starting Kosmos 2.5 quantization testing script")
    logger.info(f"Input image: {args.input_image}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Model: {args.model}")
    
    # Validate input image
    if not os.path.exists(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        sys.exit(1)
    
    # Validate output directory
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize tester
        logger.debug("Initializing quantization tester")
        tester = Kosmos25QuantizationTester(args.model)
        
        # Run all tests
        logger.info("Starting quantization tests")
        results = tester.run_all_tests(args.input_image)
        
        # Save results
        logger.info("Saving results")
        tester.save_results(results, args.output_file)
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
