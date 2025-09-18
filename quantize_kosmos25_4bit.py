#!/usr/bin/env python3
"""
Static 4-bit Quantization Script for Kosmos2.5
Handles multiple quantization strategies and saves models to disk
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import psutil
import GPUtil
from transformers import (
    Kosmos2_5ForConditionalGeneration,
    Kosmos2_5Processor,
    BitsAndBytesConfig
)
from transformers.utils import is_bitsandbytes_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantization.log'),
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
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    memory_info.update({
                        'gpu_used_gb': gpu.memoryUsed / 1024,
                        'gpu_total_gb': gpu.memoryTotal / 1024,
                        'gpu_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                    })
            except:
                pass
                
        return memory_info

class Kosmos25Quantizer:
    """Handles 4-bit quantization of Kosmos2.5 model"""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2.5", output_dir: str = "./quantized_models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = MemoryMonitor()
        
        logger.info(f"[DEBUG] 🚀 Initializing Kosmos25Quantizer")
        logger.info(f"[DEBUG] 📁 Model: {model_name}")
        logger.info(f"[DEBUG] 💾 Output directory: {output_dir}")
        logger.info(f"[DEBUG] 🖥️ CUDA available: {torch.cuda.is_available()}")
        logger.info(f"[DEBUG] 🔧 BitsAndBytes available: {is_bitsandbytes_available()}")
        
    def get_quantization_configs(self) -> Dict[str, BitsAndBytesConfig]:
        """Define different 4-bit quantization configurations"""
        configs = {}
        
        if is_bitsandbytes_available():
            logger.info(f"[DEBUG] ⚙️ Creating quantization configs")
            
            # NF4 quantization
            configs["nf4"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.uint8
            )
            
            # FP4 quantization
            configs["fp4"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.uint8
            )
            
            # NF4 with bfloat16
            configs["nf4_bf16"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.uint8
            )
            
            logger.info(f"[DEBUG] ✅ Created {len(configs)} quantization configs")
        else:
            logger.warning(f"[DEBUG] ⚠️ BitsAndBytes not available, no configs created")
            
        return configs
    
    def quantize_and_save(self, quant_type: str, config: BitsAndBytesConfig) -> Dict[str, Any]:
        """Quantize model and save to disk"""
        logger.info(f"[DEBUG] 🎯 Starting quantization: {quant_type}")
        start_time = time.time()
        
        try:
            # Memory before loading
            mem_before = self.monitor.get_memory_info()
            logger.info(f"[DEBUG] 📊 Memory before loading: {mem_before}")
            
            # Load model with quantization
            logger.info(f"[DEBUG] 📥 Loading model with {quant_type} quantization...")
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load processor
            logger.info(f"[DEBUG] 🔄 Loading processor...")
            processor = Kosmos2_5Processor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Memory after loading
            mem_after = self.monitor.get_memory_info()
            logger.info(f"[DEBUG] 📊 Memory after loading: {mem_after}")
            
            # Save paths
            model_save_path = self.output_dir / f"kosmos25_{quant_type}"
            logger.info(f"[DEBUG] 💾 Saving to: {model_save_path}")
            
            # Save model and processor
            model.save_pretrained(model_save_path, safe_serialization=True)
            processor.save_pretrained(model_save_path)
            
            # Create metadata
            metadata = {
                "quantization_type": quant_type,
                "model_name": self.model_name,
                "config": {
                    "load_in_4bit": config.load_in_4bit,
                    "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
                    "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
                    "bnb_4bit_compute_dtype": str(config.bnb_4bit_compute_dtype),
                },
                "quantization_time": time.time() - start_time,
                "memory_usage": {
                    "before": mem_before,
                    "after": mem_after
                },
                "model_size_mb": self._get_model_size(model_save_path)
            }
            
            # Save metadata
            with open(model_save_path / "quantization_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"[DEBUG] ✅ {quant_type} quantization completed in {metadata['quantization_time']:.2f}s")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
            return metadata
            
        except Exception as e:
            error_info = {
                "quantization_type": quant_type,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "time_failed": time.time() - start_time
            }
            logger.error(f"[DEBUG] ❌ {quant_type} quantization failed: {e}")
            logger.error(f"[DEBUG] 📋 Traceback: {traceback.format_exc()}")
            return error_info
    
    def _get_model_size(self, model_path: Path) -> float:
        """Calculate model size in MB"""
        try:
            total_size = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"[DEBUG] ⚠️ Could not calculate model size: {e}")
            return 0.0
    
    def quantize_all(self) -> Dict[str, Any]:
        """Quantize using all available strategies"""
        logger.info(f"[DEBUG] 🚀 Starting batch quantization")
        
        configs = self.get_quantization_configs()
        results = {"quantizations": {}, "summary": {}}
        
        total_start_time = time.time()
        
        for quant_type, config in configs.items():
            logger.info(f"[DEBUG] 🔄 Processing {quant_type}...")
            result = self.quantize_and_save(quant_type, config)
            results["quantizations"][quant_type] = result
            
            # Brief pause between quantizations
            time.sleep(2)
        
        # Create summary
        results["summary"] = {
            "total_time": time.time() - total_start_time,
            "successful": len([r for r in results["quantizations"].values() if "error" not in r]),
            "failed": len([r for r in results["quantizations"].values() if "error" in r]),
            "output_directory": str(self.output_dir)
        }
        
        # Save results
        results_file = self.output_dir / "quantization_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[DEBUG] 🎉 Batch quantization completed")
        logger.info(f"[DEBUG] 📊 Results: {results['summary']['successful']} successful, {results['summary']['failed']} failed")
        logger.info(f"[DEBUG] 💾 Results saved to: {results_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Quantize Kosmos2.5 model using 4-bit strategies")
    parser.add_argument("--model", default="microsoft/kosmos-2.5", help="Model name or path")
    parser.add_argument("--output-dir", default="./quantized_models", help="Output directory")
    parser.add_argument("--quant-type", choices=["nf4", "fp4", "nf4_bf16", "all"], 
                       default="all", help="Quantization type")
    
    args = parser.parse_args()
    
    logger.info(f"[DEBUG] 🎬 Script started with args: {vars(args)}")
    
    quantizer = Kosmos25Quantizer(args.model, args.output_dir)
    
    if args.quant_type == "all":
        results = quantizer.quantize_all()
    else:
        configs = quantizer.get_quantization_configs()
        if args.quant_type in configs:
            result = quantizer.quantize_and_save(args.quant_type, configs[args.quant_type])
            results = {"quantizations": {args.quant_type: result}}
        else:
            logger.error(f"[DEBUG] ❌ Unknown quantization type: {args.quant_type}")
            sys.exit(1)
    
    logger.info(f"[DEBUG] 🏁 Script completed")
    
if __name__ == "__main__":
    main()
