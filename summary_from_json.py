import json
import numpy as np


def generate_detailed_summary(depth_file, detection_file):
    with open(depth_file, "r") as f:
        depth_results = json.load(f)

    with open(detection_file, "r") as f:
        detection_results = json.load(f)

    summary = []

    # Depth Estimation Summary
    summary.append("1. Depth Estimation Benchmark Summary:")
    depth_models = [r["model_name"].split("/")[-1] for r in depth_results]
    depth_metrics = [
        "obj_size_consistency",
        "depth_consistency",
        "edge_comparison",
        "avg_inference_time",
    ]

    for metric in depth_metrics:
        values = [r["avg_metrics"][metric] for r in depth_results]
        best_model = depth_models[
            np.argmax(values) if metric != "avg_inference_time" else np.argmin(values)
        ]
        best_value = max(values) if metric != "avg_inference_time" else min(values)
        summary.append(f"   - {metric.replace('_', ' ').title()}:")
        summary.append(
            f"     Best model: {best_model} with a value of {best_value:.4f}"
        )
        summary.append("     All models:")
        for model, value in zip(depth_models, values):
            summary.append(f"       {model}: {value:.4f}")
        summary.append("")

    # Object Detection Summary
    summary.append("\n2. Object Detection Benchmark Summary:")
    detection_models = [r["model_name"].split("/")[-1] for r in detection_results]
    detection_metrics = ["mean_iou", "ap_50", "ap_75", "avg_inference_time"]

    for metric in detection_metrics:
        values = [r["avg_metrics"][metric] for r in detection_results]
        best_model = detection_models[
            np.argmax(values) if metric != "avg_inference_time" else np.argmin(values)
        ]
        best_value = max(values) if metric != "avg_inference_time" else min(values)
        summary.append(f"   - {metric.replace('_', ' ').title()}:")
        summary.append(
            f"     Best model: {best_model} with a value of {best_value:.4f}"
        )
        summary.append("     All models:")
        for model, value in zip(detection_models, values):
            summary.append(f"       {model}: {value:.4f}")
        summary.append("")

    # Visualization Summaries
    summary.append("\n3. Visualization Summaries:")

    # Bar Charts
    summary.append("   a. Bar Charts:")
    summary.append(
        "      - Depth Estimation: Bar charts comparing obj_size_consistency, depth_consistency, and edge_comparison across all models."
    )
    summary.append(
        "        This visualization allows for easy comparison of these metrics across different depth estimation models."
    )
    summary.append("        Key observations:")
    for metric in ["obj_size_consistency", "depth_consistency", "edge_comparison"]:
        values = [r["avg_metrics"][metric] for r in depth_results]
        best_model = depth_models[np.argmax(values)]
        worst_model = depth_models[np.argmin(values)]
        summary.append(
            f"          * {metric.replace('_', ' ').title()}: Best - {best_model} ({max(values):.4f}), Worst - {worst_model} ({min(values):.4f})"
        )

    summary.append(
        "\n      - Object Detection: Bar charts comparing mean_iou, ap_50, and ap_75 across all models."
    )
    summary.append(
        "        This visualization provides a clear comparison of these metrics for different object detection models."
    )
    summary.append("        Key observations:")
    for metric in ["mean_iou", "ap_50", "ap_75"]:
        values = [r["avg_metrics"][metric] for r in detection_results]
        best_model = detection_models[np.argmax(values)]
        worst_model = detection_models[np.argmin(values)]
        summary.append(
            f"          * {metric.replace('_', ' ').title()}: Best - {best_model} ({max(values):.4f}), Worst - {worst_model} ({min(values):.4f})"
        )

    # Scatter Plots
    summary.append("\n   b. Scatter Plots:")
    summary.append(
        "      - Depth Estimation: Scatter plot of depth_consistency vs edge_comparison for all models."
    )
    summary.append(
        "        This plot helps visualize the trade-off between depth consistency and edge comparison performance."
    )
    summary.append("        Key observations:")
    depth_consistency = [r["avg_metrics"]["depth_consistency"] for r in depth_results]
    edge_comparison = [r["avg_metrics"]["edge_comparison"] for r in depth_results]
    best_overall = depth_models[
        np.argmax(np.array(depth_consistency) * np.array(edge_comparison))
    ]
    summary.append(
        f"          * Best overall performance (considering both metrics): {best_overall}"
    )
    summary.append(
        f"          * Range of depth_consistency: {min(depth_consistency):.4f} to {max(depth_consistency):.4f}"
    )
    summary.append(
        f"          * Range of edge_comparison: {min(edge_comparison):.4f} to {max(edge_comparison):.4f}"
    )

    summary.append(
        "\n      - Object Detection: Scatter plot of mean_iou vs ap_50 for all models."
    )
    summary.append(
        "        This plot illustrates the relationship between mean IoU and AP@50 for different models."
    )
    summary.append("        Key observations:")
    mean_iou = [r["avg_metrics"]["mean_iou"] for r in detection_results]
    ap_50 = [r["avg_metrics"]["ap_50"] for r in detection_results]
    best_overall = detection_models[np.argmax(np.array(mean_iou) * np.array(ap_50))]
    summary.append(
        f"          * Best overall performance (considering both metrics): {best_overall}"
    )
    summary.append(
        f"          * Range of mean_iou: {min(mean_iou):.4f} to {max(mean_iou):.4f}"
    )
    summary.append(f"          * Range of ap_50: {min(ap_50):.4f} to {max(ap_50):.4f}")

    # Performance vs Complexity
    summary.append("\n   c. Performance vs Complexity:")
    summary.append(
        "      - Plots showing the relationship between model performance and model complexity (if available)."
    )
    summary.append(
        "        Depth Estimation: edge_comparison vs number of parameters (if available)"
    )
    summary.append(
        "        Object Detection: mean_iou vs number of parameters (if available)"
    )
    summary.append(
        "        These plots help understand the trade-off between model performance and complexity."
    )
    summary.append(
        "        Note: Model complexity information is not available in the current benchmark results."
    )

    # Precision-Recall Curve
    summary.append("\n   d. Precision-Recall Curve:")
    summary.append("      - Object Detection: Precision-Recall curves for all models.")
    summary.append(
        "        These curves show the trade-off between precision and recall at various threshold settings."
    )
    summary.append("        Key observations:")
    summary.append(
        "          * Models with curves closer to the top-right corner perform better."
    )
    summary.append(
        "          * The area under the curve (AUC) is a good indicator of overall performance."
    )
    # Note: Actual AUC values would require additional calculation

    # Metric Distribution
    summary.append("\n   e. Metric Distribution:")
    summary.append(
        "      - Violin plots showing the distribution of each metric across all images for each model."
    )
    summary.append(
        "        These plots provide insights into the consistency and variability of model performance."
    )
    summary.append("        Key observations:")
    for task, results, metrics in [
        (
            "Depth Estimation",
            depth_results,
            ["obj_size_consistency", "depth_consistency", "edge_comparison"],
        ),
        ("Object Detection", detection_results, ["mean_iou", "ap_50", "ap_75"]),
    ]:
        summary.append(f"          * {task}:")
        for metric in metrics:
            summary.append(f"            - {metric.replace('_', ' ').title()}:")
            for result in results:
                model = result["model_name"].split("/")[-1]
                values = [r["metrics"][metric] for r in result["detailed_results"]]
                summary.append(
                    f"              {model}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}"
                )

    # Performance Comparison
    summary.append("\n   f. Performance Comparison:")
    summary.append(
        "      - A side-by-side comparison of the best performing depth estimation and object detection models."
    )
    best_depth = depth_models[
        np.argmax([r["avg_metrics"]["edge_comparison"] for r in depth_results])
    ]
    best_detection = detection_models[
        np.argmax([r["avg_metrics"]["mean_iou"] for r in detection_results])
    ]
    summary.append(f"        Best Depth Estimation model: {best_depth}")
    summary.append(f"        Best Object Detection model: {best_detection}")
    summary.append(
        "        This comparison highlights the top performers in each task and their key metrics."
    )

    return "\n".join(summary)


# Usage
summary = generate_detailed_summary(
    "depth_estimation_benchmark_results.json",
    "vision_transformer_benchmark_results.json",
)
print(summary)
