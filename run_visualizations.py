import json
import os
from benchmarking.visualization import VisualizationUtils


def load_json_results(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def run_visualizations():
    # Load results
    depth_results = load_json_results("depth_estimation_benchmark_results.json")
    detection_results = load_json_results("vision_transformer_benchmark_results.json")

    # Create output directory
    os.makedirs("visualization_results", exist_ok=True)

    # Depth estimation visualizations
    VisualizationUtils.create_comparison_graphs(depth_results, task_type="depth")
    VisualizationUtils.plot_metric_distribution(depth_results, task_type="depth")

    # Object detection visualizations
    VisualizationUtils.create_comparison_graphs(
        detection_results, task_type="detection"
    )
    VisualizationUtils.plot_metric_distribution(
        detection_results, task_type="detection"
    )
    # VisualizationUtils.plot_precision_recall_curve(
    #     detection_results, task_type="detection"
    # )
    # VisualizationUtils.plot_confusion_matrix(detection_results, task_type="detection")

    # Combined visualizations
    VisualizationUtils.plot_performance_comparison(depth_results, detection_results)

    print(
        "All visualizations have been generated and saved in the 'visualization_results' directory."
    )


if __name__ == "__main__":
    run_visualizations()
