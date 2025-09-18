# Kosmos2.5 4-bit Quantization Testing Script

This script comprehensively tests various 4-bit quantization strategies for the Microsoft Kosmos2.5 model, evaluating OCR and Markdown generation performance while monitoring memory usage.

## Features

- **Multiple Quantization Strategies**: Tests 6 different 4-bit quantization configurations:
  - NF4 with float16
  - NF4 with double quantization 
  - FP4 with float16
  - FP4 with double quantization
  - NF4 with bfloat16
  - FP4 with bfloat16

- **Performance Evaluation**: Tests both OCR and Markdown generation tasks
- **Memory Monitoring**: Tracks GPU and RAM usage during model loading and inference
- **Comprehensive Reporting**: Saves detailed results in JSON format with timing and memory metrics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have CUDA installed if you want to use GPU acceleration.

## Usage

### Basic Usage

```python
python kosmos25_quantization_test.py
```

### Custom Usage

```python
from kosmos25_quantization_test import Kosmos25QuantTester

# Initialize tester
tester = Kosmos25QuantTester("microsoft/kosmos-2.5")

# Run tests with your own image
tester.run_comprehensive_test("path/to/your/image.jpg", "custom_results.json")
```

### Configuration Options

You can modify the quantization configurations in the `setup_quantization_configs()` method:

```python
configs = {
    "custom_nf4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
}
```

## Output

The script generates a comprehensive JSON report including:

- **System Information**: Hardware specs, CUDA version, etc.
- **Per-Configuration Results**: 
  - Model loading time and memory usage
  - OCR task performance and output
  - Markdown task performance and output
  - Inference timing
- **Summary Statistics**: Best performing configurations by speed and memory usage

### Sample Output Structure

```json
{
  "system_info": {
    "model_name": "microsoft/kosmos-2.5",
    "torch_version": "2.1.0",
    "cuda_available": true,
    "gpu_name": "NVIDIA RTX 4090"
  },
  "test_results": [
    {
      "config_name": "nf4",
      "load_successful": true,
      "load_info": {
        "load_time_seconds": 15.2,
        "gpu_allocated_gb": 3.4
      },
      "ocr_result": {
        "success": true,
        "output_text": "Extracted OCR text...",
        "inference_time_seconds": 2.1
      }
    }
  ],
  "summary": {
    "fastest_ocr": ["nf4", 2.1],
    "lowest_memory": ["fp4", 3.2]
  }
}
```

## Test Image

If no test image is provided, the script automatically creates a sample image with text content suitable for OCR testing.

## Memory Monitoring

The script monitors:
- **RAM Usage**: Before and after model loading
- **GPU Memory**: Allocated and cached memory
- **GPU Utilization**: During inference (if GPUtil is available)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or try configurations with lower memory usage
2. **Model Loading Errors**: Ensure you have sufficient RAM and stable internet connection
3. **Quantization Errors**: Update `bitsandbytes` to the latest version

### Memory Optimization

- The script automatically clears GPU memory between tests
- Each model is deleted after testing to free memory
- Use `torch.cuda.empty_cache()` if you encounter memory issues

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- At least 8GB GPU memory

## Model Information

This script tests the Microsoft Kosmos2.5 model:
- **Model**: `microsoft/kosmos-2.5`
- **Tasks**: OCR (Optical Character Recognition) and Markdown generation
- **Documentation**: [Hugging Face Kosmos2.5](https://huggingface.co/docs/transformers/main/en/model_doc/kosmos2_5)

## License

This script is provided as-is for testing and evaluation purposes.
