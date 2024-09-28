from data.dataset_preparer import DatasetPreparer
from utils.utils import load_benchmark_data
from benchmarking.visualization import VisualizationUtils
import logging
import json
from benchmarking.depth_benchmark import run_benchmark
from config import DEPTH_MODELS_TO_BENCHMARK

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    existing_data = load_benchmark_data()
    if existing_data:
        benchmark_data = existing_data
    else:
        data_loader = DatasetPreparer()
        benchmark_data = data_loader.load_benchmark_data()

    all_results = []
    for model_type, model_name in DEPTH_MODELS_TO_BENCHMARK:
        print(f"Starting benchmark for {model_name}")
        try:
            print(f"{model_type} {model_name}")
            result = run_benchmark(model_type, model_name, benchmark_data)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")

    print("\nComparison of Average Metrics:")
    metrics = [
        "obj_size_consistency",
        "depth_consistency",
        "edge_comparison",
        "avg_inference_time",
    ]
    print(f"{'Model':<30} " + " ".join(f"{m:>20}" for m in metrics))
    for result in all_results:
        model_name = result["model_name"].split("/")[-1]
        metrics_values = [f"{result['avg_metrics'][m]:.4f}" for m in metrics]
        print(f"{model_name:<30} " + " ".join(f"{v:>20}" for v in metrics_values))

    # Save detailed results
    with open("depth_estimation_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Detailed results saved to depth_estimation_benchmark_results.json")

    # Create visualization graphs
    VisualizationUtils.create_comparison_graphs(all_results, task_type="depth")


if __name__ == "__main__":
    main()
