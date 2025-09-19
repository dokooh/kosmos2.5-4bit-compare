#!/usr/bin/env python3
"""
Simple test script to check if quantized models can be loaded successfully
"""

import sys
import torch
from pathlib import Path
from transformers import Kosmos2_5ForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import time

def test_model_loading(model_path):
    """Test loading a quantized model"""
    print(f"Testing model at: {model_path}")
    
    try:
        # Check if model directory exists
        model_dir = Path(model_path)
        if not model_dir.exists():
            print(f"❌ Model directory does not exist: {model_dir}")
            return False
            
        # Check for config file
        config_file = model_dir / "config.json"
        if not config_file.exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        print(f"✅ Model directory exists with config.json")
        
        # Try to determine if this is a quantized model
        quantization_config = None
        try:
            # Check if this appears to be a quantized model by looking for safetensors files
            safetensor_files = list(model_dir.glob("*.safetensors"))
            print(f"Found {len(safetensor_files)} safetensors files")
            
            # For quantized models, we assume quantization config
            if len(safetensor_files) > 0:
                print("Detected quantized model, using BitsAndBytesConfig")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
        except Exception as e:
            print(f"Error checking quantization: {e}")
        
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")
        print("✅ Processor loaded successfully")
        
        print("Loading model...")
        start_time = time.time()
        
        if quantization_config:
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
            
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Get model info
        print(f"Model type: {type(model)}")
        print(f"Device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model_loading.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = test_model_loading(model_path)
    
    if success:
        print("🎉 Model loading test PASSED")
        sys.exit(0)
    else:
        print("💥 Model loading test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()