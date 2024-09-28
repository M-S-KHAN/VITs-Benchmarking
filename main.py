from data.dataset_preparer import DatasetPreparer
from utils.utils import load_benchmark_data
from benchmarking.detection_benchmark import run_benchmark
from benchmarking.visualization import VisualizationUtils
from config import DET_MODELS_TO_BENCHMARK
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    existing_data = load_benchmark_data()
    if existing_data:
        benchmark_data = existing_data
    else:
        data_loader = DatasetPreparer()
        benchmark_data = data_loader.load_benchmark_data()

    all_results = []
    for model_type, model_name in DET_MODELS_TO_BENCHMARK:
        print(f"Starting benchmark for {model_name}")
        try:
            print(f'{model_type} {model_name}')
            result = run_benchmark(model_type, model_name, benchmark_data)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")

    print("\nComparison of Average Metrics:")
    metrics = ["mean_iou", "ap_50", "ap_75", "avg_inference_time"]
    print(f"{'Model':<30} " + " ".join(f"{m:>12}" for m in metrics))
    for result in all_results:
        model_name = result["model_name"].split("/")[-1]
        metrics_values = [f"{result['avg_metrics'][m]:.4f}" for m in metrics]
        print(f"{model_name:<30} " + " ".join(f"{v:>12}" for v in metrics_values))

    # Save detailed results
    with open('vision_transformer_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info("Detailed results saved to vision_transformer_benchmark_results.json")

    # Create visualization graphs
    VisualizationUtils.create_comparison_graphs(all_results, task_type="detection")

if __name__ == "__main__":
    main()