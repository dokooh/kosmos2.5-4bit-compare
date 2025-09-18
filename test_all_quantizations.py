#!/usr/bin/env python3
"""
Batch Testing Script for All Quantization Strategies
Orchestrates quantization and testing with comprehensive reporting
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BatchTester:
    """Orchestrates batch testing of quantized models"""
    
    def __init__(self, quantized_models_dir: str, test_image: str, output_dir: str):
        self.quantized_models_dir = Path(quantized_models_dir)
        self.test_image = Path(test_image)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[DEBUG] 🚀 Initializing BatchTester")
        logger.info(f"[DEBUG] 📁 Quantized models dir: {quantized_models_dir}")
        logger.info(f"[DEBUG] 🖼️ Test image: {test_image}")
        logger.info(f"[DEBUG] 📊 Output dir: {output_dir}")
        
    def find_quantized_models(self) -> List[Path]:
        """Find all quantized model directories"""
        model_dirs = []
        
        for item in self.quantized_models_dir.iterdir():
            if item.is_dir() and item.name.startswith("kosmos25_"):
                config_file = item / "config.json"
                if config_file.exists():
                    model_dirs.append(item)
                    logger.info(f"[DEBUG] 📦 Found model: {item.name}")
        
        logger.info(f"[DEBUG] 📊 Total models found: {len(model_dirs)}")
        return model_dirs
    
    def test_single_model(self, model_path: Path, task: str = "ocr") -> Dict[str, Any]:
        """Test a single quantized model"""
        logger.info(f"[DEBUG] 🎯 Testing model: {model_path.name} with task: {task}")
        
        start_time = time.time()
        
        try:
            # Prepare command
            cmd = [
                sys.executable, "ocr_inference_kosmos25.py",
                str(model_path),
                str(self.test_image),
                "--task", task
            ]
            
            logger.info(f"[DEBUG] 🚀 Running command: {' '.join(cmd)}")
            
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse JSON output
                try:
                    output_data = json.loads(result.stdout.split('\n')[-2])  # Get last JSON line
                    
                    test_result = {
                        "model_name": model_path.name,
                        "task": task,
                        "success": True,
                        "total_time": total_time,
                        "inference_time": output_data.get("inference_time", 0),
                        "generated_text": output_data.get("generated_text", ""),
                        "memory_usage": output_data.get("memory_usage", {}),
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                    
                    logger.info(f"[DEBUG] ✅ {model_path.name} test successful in {total_time:.2f}s")
                    
                except json.JSONDecodeError as e:
                    test_result = {
                        "model_name": model_path.name,
                        "task": task,
                        "success": False,
                        "total_time": total_time,
                        "error": f"JSON decode error: {e}",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                    logger.error(f"[DEBUG] ❌ {model_path.name} JSON decode failed: {e}")
            else:
                test_result = {
                    "model_name": model_path.name,
                    "task": task,
                    "success": False,
                    "total_time": total_time,
                    "error": f"Process failed with code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                logger.error(f"[DEBUG] ❌ {model_path.name} test failed with code {result.returncode}")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            test_result = {
                "model_name": model_path.name,
                "task": task,
                "success": False,
                "total_time": time.time() - start_time,
                "error": "Timeout (300s)"
            }
            logger.error(f"[DEBUG] ⏰ {model_path.name} test timed out")
            return test_result
            
        except Exception as e:
            test_result = {
                "model_name": model_path.name,
                "task": task,
                "success": False,
                "total_time": time.time() - start_time,
                "error": str(e)
            }
            logger.error(f"[DEBUG] ❌ {model_path.name} test exception: {e}")
            return test_result
    
    def run_batch_tests(self, tasks: List[str] = ["ocr", "markdown"]) -> Dict[str, Any]:
        """Run batch tests on all models"""
        logger.info(f"[DEBUG] 🚀 Starting batch tests")
        
        models = self.find_quantized_models()
        if not models:
            logger.error(f"[DEBUG] ❌ No quantized models found")
            return {"error": "No models found"}
        
        results = {
            "test_config": {
                "test_image": str(self.test_image),
                "tasks": tasks,
                "models_tested": len(models)
            },
            "results": {},
            "summary": {}
        }
        
        start_time = time.time()
        
        # Test each model with each task
        for task in tasks:
            logger.info(f"[DEBUG] 🎯 Testing task: {task}")
            results["results"][task] = []
            
            for model_path in models:
                result = self.test_single_model(model_path, task)
                results["results"][task].append(result)
                
                # Brief pause between tests
                time.sleep(1)
        
        # Generate summary
        total_time = time.time() - start_time
        total_tests = len(models) * len(tasks)
        successful_tests = sum(
            1 for task_results in results["results"].values() 
            for result in task_results if result["success"]
        )
        
        results["summary"] = {
            "total_time": total_time,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        }
        
        logger.info(f"[DEBUG] 🎉 Batch tests completed")
        logger.info(f"[DEBUG] 📊 Results: {successful_tests}/{total_tests} successful ({results['summary']['success_rate']:.1f}%)")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate comprehensive report"""
        logger.info(f"[DEBUG] 📊 Generating reports...")
        
        # Save raw results
        results_file = self.output_dir / "batch_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[DEBUG] 💾 Raw results saved to: {results_file}")
        
        # Generate CSV report
        try:
            csv_data = []
            for task, task_results in results["results"].items():
                for result in task_results:
                    csv_row = {
                        "model_name": result["model_name"],
                        "task": result["task"],
                        "success": result["success"],
                        "total_time": result.get("total_time", 0),
                        "inference_time": result.get("inference_time", 0),
                        "error": result.get("error", ""),
                        "text_length": len(result.get("generated_text", "")),
                    }
                    
                    # Add memory usage if available
                    memory = result.get("memory_usage", {})
                    if memory:
                        csv_row.update({
                            "ram_before_gb": memory.get("before", {}).get("ram_used_gb", 0),
                            "ram_after_gb": memory.get("after", {}).get("ram_used_gb", 0),
                            "gpu_before_gb": memory.get("before", {}).get("gpu_used_gb", 0),
                            "gpu_after_gb": memory.get("after", {}).get("gpu_used_gb", 0),
                        })
                    
                    csv_data.append(csv_row)
            
            # Save CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_file = self.output_dir / "batch_test_report.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"[DEBUG] 📊 CSV report saved to: {csv_file}")
                
                # Print summary table
                print("\n" + "="*80)
                print("BATCH TEST SUMMARY")
                print("="*80)
                print(f"Total Tests: {results['summary']['total_tests']}")
                print(f"Successful: {results['summary']['successful_tests']}")
                print(f"Failed: {results['summary']['failed_tests']}")
                print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
                print(f"Total Time: {results['summary']['total_time']:.2f}s")
                print("="*80)
                
                # Print per-model results
                print("\nPER-MODEL RESULTS:")
                print("-" * 80)
                summary_df = df.groupby(['model_name', 'task']).agg({
                    'success': 'first',
                    'total_time': 'mean',
                    'text_length': 'mean'
                }).round(2)
                print(summary_df)
                
        except Exception as e:
            logger.error(f"[DEBUG] ❌ Failed to generate CSV report: {e}")
        
        # Generate HTML report
        try:
            html_content = self._generate_html_report(results)
            html_file = self.output_dir / "batch_test_report.html"
            with open(html_file, "w") as f:
                f.write(html_content)
            logger.info(f"[DEBUG] 🌐 HTML report saved to: {html_file}")
            
        except Exception as e:
            logger.error(f"[DEBUG] ❌ Failed to generate HTML report: {e}")
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kosmos2.5 4-bit Quantization Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .error {{ background-color: #ffe8e8; }}
                .success {{ background-color: #e8f5e8; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .code {{ font-family: monospace; background-color: #f5f5f5; padding: 2px 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Kosmos2.5 4-bit Quantization Test Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Total Tests:</strong> {results['summary']['total_tests']}</p>
                <p><strong>Successful:</strong> {results['summary']['successful_tests']}</p>
                <p><strong>Failed:</strong> {results['summary']['failed_tests']}</p>
                <p><strong>Success Rate:</strong> {results['summary']['success_rate']:.1f}%</p>
                <p><strong>Total Time:</strong> {results['summary']['total_time']:.2f}s</p>
            </div>
        """
        
        for task, task_results in results["results"].items():
            html += f"""
            <h2>Task: {task.upper()}</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Total Time (s)</th>
                    <th>Inference Time (s)</th>
                    <th>Text Length</th>
                    <th>Error</th>
                </tr>
            """
            
            for result in task_results:
                status_class = "success" if result["success"] else "error"
                status_text = "✅ Success" if result["success"] else "❌ Failed"
                error_text = result.get("error", "")[:100] + ("..." if len(result.get("error", "")) > 100 else "")
                
                html += f"""
                <tr class="{status_class}">
                    <td><span class="code">{result['model_name']}</span></td>
                    <td>{status_text}</td>
                    <td>{result.get('total_time', 0):.2f}</td>
                    <td>{result.get('inference_time', 0):.2f}</td>
                    <td>{len(result.get('generated_text', ''))}</td>
                    <td>{error_text}</td>
                </tr>
                """
            
            html += "</table>"
        
        html += """
        </body>
        </html>
        """
        
        return html

def main():
    parser = argparse.ArgumentParser(description="Batch test all quantized Kosmos2.5 models")
    parser.add_argument("--models-dir", default="./quantized_models", 
                       help="Directory containing quantized models")
    parser.add_argument("--test-image", required=True,
                       help="Path to test image")
    parser.add_argument("--output-dir", default="./test_results",
                       help="Output directory for results")
    parser.add_argument("--tasks", nargs="+", default=["ocr", "markdown"],
                       choices=["ocr", "markdown"],
                       help="Tasks to test")
    
    args = parser.parse_args()
    
    logger.info(f"[DEBUG] 🎬 Batch testing started with args: {vars(args)}")
    
    # Validate inputs
    if not Path(args.test_image).exists():
        logger.error(f"[DEBUG] ❌ Test image not found: {args.test_image}")
        sys.exit(1)
    
    if not Path(args.models_dir).exists():
        logger.error(f"[DEBUG] ❌ Models directory not found: {args.models_dir}")
        sys.exit(1)
    
    # Run batch tests
    tester = BatchTester(args.models_dir, args.test_image, args.output_dir)
    results = tester.run_batch_tests(args.tasks)
    
    if "error" in results:
        logger.error(f"[DEBUG] ❌ Batch testing failed: {results['error']}")
        sys.exit(1)
    
    # Generate reports
    tester.generate_report(results)
    
    logger.info(f"[DEBUG] 🏁 Batch testing completed successfully")

if __name__ == "__main__":
    main()
