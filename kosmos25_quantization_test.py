#!/usr/bin/env python3
"""
Kosmos2.5 4-bit Quantization Testing Script
Tests various 4-bit quantization strategies and evaluates OCR/Markdown performance
"""

import os
import gc
import json
import time
import psutil
import torch
from datetime import datetime
from PIL import Image
from transformers import (
    AutoProcessor, 
    Kosmos2_5ForConditionalGeneration,
    BitsAndBytesConfig
)
import GPUtil
from typing import Dict, List, Tuple, Optional

class MemoryMonitor:
    """Monitor GPU and RAM memory usage"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}
        
        # RAM usage
        ram = psutil.virtual_memory()
        memory_info['ram_used_gb'] = ram.used / (1024**3)
        memory_info['ram_available_gb'] = ram.available / (1024**3)
        memory_info['ram_percent'] = ram.percent
        
        # GPU usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
            memory_info['gpu_allocated_gb'] = gpu_memory
            memory_info['gpu_cached_gb'] = gpu_memory_cached
            
            # Additional GPU info using GPUtil
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    memory_info['gpu_total_gb'] = gpu.memoryTotal / 1024
                    memory_info['gpu_used_gb'] = gpu.memoryUsed / 1024
                    memory_info['gpu_free_gb'] = gpu.memoryFree / 1024
                    memory_info['gpu_utilization_percent'] = gpu.load * 100
            except:
                pass
        
        return memory_info

class Kosmos25QuantTester:
    """Test different 4-bit quantization strategies for Kosmos2.5"""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2.5"):
        self.model_name = model_name
        self.processor = None
        self.memory_monitor = MemoryMonitor()
        self.results = []
        
    def setup_quantization_configs(self) -> Dict[str, BitsAndBytesConfig]:
        """Setup different 4-bit quantization configurations"""
        configs = {
            "nf4": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            ),
            "nf4_double_quant": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "fp4": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            ),
            "fp4_double_quant": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "nf4_bf16": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            ),
            "fp4_bf16": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
        }
        return configs
    
    def load_model_with_config(self, config_name: str, quantization_config: BitsAndBytesConfig) -> Tuple[object, Dict]:
        """Load model with specific quantization configuration"""
        print(f"\n=== Loading model with {config_name} quantization ===")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Record initial memory
        initial_memory = self.memory_monitor.get_memory_info()
        
        start_time = time.time()
        
        try:
            # Load processor if not already loaded
            if self.processor is None:
                print("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with quantization
            print(f"Loading model with {config_name}...")
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            load_time = time.time() - start_time
            
            # Record post-load memory
            post_load_memory = self.memory_monitor.get_memory_info()
            
            # Calculate memory usage
            memory_usage = {
                'ram_increase_gb': post_load_memory['ram_used_gb'] - initial_memory['ram_used_gb'],
                'gpu_allocated_gb': post_load_memory.get('gpu_allocated_gb', 0),
                'gpu_cached_gb': post_load_memory.get('gpu_cached_gb', 0),
                'load_time_seconds': load_time
            }
            
            print(f"Model loaded successfully in {load_time:.2f}s")
            print(f"RAM increase: {memory_usage['ram_increase_gb']:.2f} GB")
            print(f"GPU allocated: {memory_usage.get('gpu_allocated_gb', 0):.2f} GB")
            
            return model, memory_usage
            
        except Exception as e:
            print(f"Error loading model with {config_name}: {str(e)}")
            return None, {'error': str(e)}
    
    def test_ocr_markdown(self, model, image_path: str, task_type: str = "ocr") -> Dict:
        """Test OCR or Markdown generation on the provided image"""
        try:
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare prompts based on task type
            if task_type.lower() == "ocr":
                prompt = "<ocr>"
            elif task_type.lower() == "markdown":
                prompt = "<md>"
            else:
                prompt = "<ocr>"  # default to OCR
            
            # Record inference start time and memory
            start_time = time.time()
            initial_memory = self.memory_monitor.get_memory_info()
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            inference_time = time.time() - start_time
            post_inference_memory = self.memory_monitor.get_memory_info()
            
            # Extract the actual generated content (remove the prompt)
            if prompt in generated_text:
                output_text = generated_text.split(prompt, 1)[-1].strip()
            else:
                output_text = generated_text.strip()
            
            return {
                'success': True,
                'output_text': output_text,
                'inference_time_seconds': inference_time,
                'memory_during_inference': post_inference_memory,
                'task_type': task_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'task_type': task_type
            }
    
    def run_comprehensive_test(self, image_path: str, output_file: str = "kosmos25_quant_results.json"):
        """Run comprehensive testing of all quantization strategies"""
        
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            return
        
        print(f"Starting comprehensive Kosmos2.5 quantization testing...")
        print(f"Image: {image_path}")
        print(f"Results will be saved to: {output_file}")
        print(f"Test started at: {datetime.now()}")
        
        # Get quantization configurations
        configs = self.setup_quantization_configs()
        
        # Test each configuration
        for config_name, quant_config in configs.items():
            print(f"\n{'='*60}")
            print(f"Testing configuration: {config_name}")
            print(f"{'='*60}")
            
            # Load model
            model, load_info = self.load_model_with_config(config_name, quant_config)
            
            if model is None:
                result = {
                    'config_name': config_name,
                    'load_successful': False,
                    'load_error': load_info.get('error', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)
                continue
            
            # Test OCR task
            print("\nTesting OCR task...")
            ocr_result = self.test_ocr_markdown(model, image_path, "ocr")
            
            # Test Markdown task
            print("Testing Markdown task...")
            markdown_result = self.test_ocr_markdown(model, image_path, "markdown")
            
            # Compile results for this configuration
            result = {
                'config_name': config_name,
                'quantization_config': {
                    'load_in_4bit': True,
                    'bnb_4bit_quant_type': quant_config.bnb_4bit_quant_type,
                    'bnb_4bit_compute_dtype': str(quant_config.bnb_4bit_compute_dtype),
                    'bnb_4bit_use_double_quant': quant_config.bnb_4bit_use_double_quant,
                },
                'load_successful': True,
                'load_info': load_info,
                'ocr_result': ocr_result,
                'markdown_result': markdown_result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            # Print summary for this config
            print(f"\n--- Summary for {config_name} ---")
            print(f"Load time: {load_info.get('load_time_seconds', 0):.2f}s")
            print(f"GPU memory: {load_info.get('gpu_allocated_gb', 0):.2f} GB")
            print(f"OCR inference time: {ocr_result.get('inference_time_seconds', 0):.2f}s")
            print(f"OCR success: {ocr_result.get('success', False)}")
            print(f"Markdown inference time: {markdown_result.get('inference_time_seconds', 0):.2f}s")
            print(f"Markdown success: {markdown_result.get('success', False)}")
            
            # Clean up model to free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"Completed testing {config_name}")
        
        # Save results
        self.save_results(output_file)
        print(f"\n{'='*60}")
        print("Testing completed! Results saved to:", output_file)
    
    def save_results(self, output_file: str):
        """Save test results to JSON file"""
        
        # Add system information
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'system_memory_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if torch.cuda.is_available():
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Prepare final results
        final_results = {
            'system_info': system_info,
            'test_results': self.results,
            'summary': self.generate_summary()
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    def generate_summary(self) -> Dict:
        """Generate a summary of test results"""
        successful_configs = [r for r in self.results if r.get('load_successful', False)]
        
        if not successful_configs:
            return {'message': 'No configurations loaded successfully'}
        
        summary = {
            'total_configs_tested': len(self.results),
            'successful_configs': len(successful_configs),
            'failed_configs': len(self.results) - len(successful_configs)
        }
        
        # Find best performing configurations
        ocr_times = [(r['config_name'], r['ocr_result'].get('inference_time_seconds', float('inf'))) 
                     for r in successful_configs if r['ocr_result'].get('success', False)]
        
        markdown_times = [(r['config_name'], r['markdown_result'].get('inference_time_seconds', float('inf'))) 
                         for r in successful_configs if r['markdown_result'].get('success', False)]
        
        memory_usage = [(r['config_name'], r['load_info'].get('gpu_allocated_gb', float('inf'))) 
                       for r in successful_configs]
        
        if ocr_times:
            summary['fastest_ocr'] = min(ocr_times, key=lambda x: x[1])
        
        if markdown_times:
            summary['fastest_markdown'] = min(markdown_times, key=lambda x: x[1])
        
        if memory_usage:
            summary['lowest_memory'] = min(memory_usage, key=lambda x: x[1])
        
        return summary

def main():
    """Main function to run the quantization testing"""
    
    # Configuration
    image_path = "test_image.jpg"  # Update this path to your test image
    output_file = f"kosmos25_quant_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create a sample test image if it doesn't exist
    if not os.path.exists(image_path):
        print(f"Creating sample test image at {image_path}")
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple image with text for OCR testing
        img = Image.new('RGB', (800, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text
        text_lines = [
            "Kosmos2.5 Quantization Test",
            "This is a test image for OCR and Markdown generation.",
            "Testing 4-bit quantization strategies:",
            "• NF4 quantization",
            "• FP4 quantization", 
            "• Double quantization variants",
            "• Different compute dtypes (float16, bfloat16)"
        ]
        
        y_offset = 50
        for line in text_lines:
            draw.text((50, y_offset), line, fill='black')
            y_offset += 40
        
        img.save(image_path)
        print(f"Sample image created at {image_path}")
    
    # Run the tests
    tester = Kosmos25QuantTester()
    tester.run_comprehensive_test(image_path, output_file)

if __name__ == "__main__":
    main()
