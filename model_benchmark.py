import torch
from benchmark_config import BenchmarkConfig
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import StructuralSimilarityIndexMeasure
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from torchmetrics.segmentation import MeanIoU
import torch
from torchvision.ops import box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from PIL import Image
import time


class ModelBenchmark:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")

    def benchmark_model(self, model_type, model_name, variant, dataset):
        feature_extractor, model = self.config.load_model(model_type, model_name, variant)
        model.to(self.device)
        results = []

        for image_path, ground_truth in tqdm(dataset, desc=f"Benchmarking {model_name} - {variant}"):
            image = Image.open(image_path).convert("RGB")
            inputs = feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                start_time = time.time()
                outputs = model(**inputs)
                end_time = time.time()
            
            inference_time = end_time - start_time
            
            # Process predictions
            pred_boxes = outputs.pred_boxes[0].cpu()
            pred_scores = outputs.logits[0].softmax(-1).max(-1).values.cpu()
            
            # Filter predictions based on confidence threshold
            keep = pred_scores > self.config.CONFIDENCE_THRESHOLD
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            
            results.append({
                "image_path": image_path,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "ground_truth": ground_truth,
                "inference_time": inference_time
            })

        return results

    def calculate_metrics(self, results):
        metrics = {
            "mean_iou": [],
            "mean_giou": [],
            "ap_50": [],
            "ap_75": [],
            "inference_time": []
        }

        for result in results:
            pred_boxes = result["pred_boxes"]
            gt_boxes = torch.tensor(result["ground_truth"]["boxes"])
            
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Calculate IoU and GIoU
                iou = box_iou(pred_boxes, gt_boxes)
                giou = generalized_box_iou(pred_boxes, gt_boxes)
                
                # Assign predictions to ground truth boxes
                cost_matrix = -iou.numpy()
                pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
                
                # Calculate metrics
                metrics["mean_iou"].append(iou[pred_indices, gt_indices].mean().item())
                metrics["mean_giou"].append(giou[pred_indices, gt_indices].mean().item())
                metrics["ap_50"].append((iou[pred_indices, gt_indices] > 0.5).float().mean().item())
                metrics["ap_75"].append((iou[pred_indices, gt_indices] > 0.75).float().mean().item())
            
            metrics["inference_time"].append(result["inference_time"])

        # Calculate average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key])

        return metrics

    def run_benchmark(self, dataset, num_runs=5):
        all_results = {}

        for model_type in ["object_detection", "depth_estimation"]:
            models = (self.config.OBJECT_DETECTION_MODELS if model_type == "object_detection" 
                      else self.config.DEPTH_ESTIMATION_MODELS)
            
            for model_name, variants in models.items():
                model_results = {}
                for variant in variants:
                    variant_results = []
                    for _ in range(num_runs):
                        results = self.benchmark_model(model_type, model_name, variant, dataset)
                        metrics = self.calculate_metrics(results)
                        variant_results.append(metrics)
                    
                    # Calculate average and standard deviation of metrics
                    avg_metrics = {k: np.mean([r[k] for r in variant_results]) for k in variant_results[0]}
                    std_metrics = {k: np.std([r[k] for r in variant_results]) for k in variant_results[0]}
                    
                    model_results[variant] = {
                        "avg_metrics": avg_metrics,
                        "std_metrics": std_metrics
                    }
                
                all_results[f"{model_name}"] = model_results

        return all_results

    def analyze_dataset_impact(self, all_results, dataset):
        impact_analysis = {}
        
        # Analyze dataset characteristics
        image_sizes = [Image.open(item[0]).size for item in dataset]
        avg_image_size = np.mean(image_sizes, axis=0)
        
        object_densities = [len(item[1]['boxes']) / (size[0] * size[1]) for item, size in zip(dataset, image_sizes)]
        avg_object_density = np.mean(object_densities)
        
        for model_name, model_results in all_results.items():
            impact_analysis[model_name] = {
                "performance_vs_image_size": self._correlation(model_results, "mean_iou", image_sizes),
                "performance_vs_object_density": self._correlation(model_results, "mean_iou", object_densities)
            }
        
        return impact_analysis, {"avg_image_size": avg_image_size, "avg_object_density": avg_object_density}

    def _correlation(self, model_results, metric, dataset_property):
        # Calculate correlation between model performance and dataset property
        performance = [variant["avg_metrics"][metric] for variant in model_results.values()]
        return np.corrcoef(performance, dataset_property)[0, 1]

    def report_results(self, all_results, impact_analysis, dataset_info):
        print("Benchmark Results:")
        for model_name, model_results in all_results.items():
            print(f"\n{model_name}:")
            for variant, metrics in model_results.items():
                print(f"  {variant}:")
                for metric, value in metrics["avg_metrics"].items():
                    print(f"    {metric}: {value:.4f} ± {metrics['std_metrics'][metric]:.4f}")
        
        print("\nDataset Impact Analysis:")
        print(f"Average Image Size: {dataset_info['avg_image_size']}")
        print(f"Average Object Density: {dataset_info['avg_object_density']:.6f}")
        
        for model_name, analysis in impact_analysis.items():
            print(f"\n{model_name}:")
            print(f"  Performance vs Image Size Correlation: {analysis['performance_vs_image_size']:.4f}")
            print(f"  Performance vs Object Density Correlation: {analysis['performance_vs_object_density']:.4f}")
        
        print("\nLimitations and Considerations:")
        print("- Models were trained on different datasets with potentially different class labels.")
        print("- The evaluation uses bounding box metrics that don't rely on specific class names.")
        print("- Performance may vary based on dataset characteristics and the specific use case.")
        print("- Multiple runs were performed to account for variability, but results may still be affected by randomness.")
        print("- The test dataset may not be representative of all possible real-world scenarios.")

    def visualize_comparison(
        self, image, od_outputs, depth_outputs=None, ground_truth=None
    ):
        """
        Visualizes results from multiple models for comparison.

        Parameters:
        - image: The input image as a numpy array.
        - od_outputs: Dictionary of object detection outputs from different models.
        - depth_outputs: Optional; Dictionary of depth estimation outputs from different models.
        - ground_truth: Optional; ground truth annotations to overlay.
        """

        n_models = len(od_outputs)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        for idx, (model_name, od_output) in enumerate(od_outputs.items()):
            ax_od = axes[0, idx]
            ax_od.imshow(image)

            for pred in od_output:
                x, y, w, h = pred["bbox"]
                rect = plt.Rectangle(
                    (x, y), w, h, edgecolor="red", fill=False, linewidth=2
                )
                ax_od.add_patch(rect)
                ax_od.text(
                    x,
                    y,
                    f'{pred["category"]}: {pred["score"]:.2f}',
                    bbox=dict(facecolor="white", alpha=0.7),
                    fontsize=8,
                )

            if ground_truth:
                for gt in ground_truth:
                    x, y, w, h = gt["bbox"]
                    rect = plt.Rectangle(
                        (x, y), w, h, edgecolor="green", fill=False, linewidth=1
                    )
                    ax_od.add_patch(rect)
                    ax_od.text(
                        x,
                        y + h + 5,
                        gt["category"],
                        bbox=dict(facecolor="green", alpha=0.7),
                        fontsize=8,
                    )

            ax_od.set_title(f"Object Detection: {model_name}")
            ax_od.axis("off")

            if depth_outputs:
                ax_depth = axes[1, idx]
                depth_map = (
                    depth_outputs[model_name]["predicted_depth"].squeeze().cpu().numpy()
                )
                depth_map = (depth_map - depth_map.min()) / (
                    depth_map.max() - depth_map.min()
                )

                im = ax_depth.imshow(depth_map, cmap="plasma")
                ax_depth.set_title(f"Depth Estimation: {model_name}")
                ax_depth.axis("off")

                divider = make_axes_locatable(ax_depth)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.show()

    def visualize_results(
        self,
        image,
        object_detection_output,
        depth_estimation_output=None,
        ground_truth=None,
    ):
        """
        Visualizes the results of object detection and depth estimation on an image.

        Parameters:
        - image: The input image as a numpy array.
        - object_detection_output: Model predictions including bounding boxes and labels.
        - depth_estimation_output: Optional; depth map prediction.
        - ground_truth: Optional; ground truth annotations to overlay.
        """

        fig, axes = plt.subplots(
            1, 2 if depth_estimation_output is not None else 1, figsize=(20, 10)
        )

        if depth_estimation_output is not None:
            ax_od, ax_depth = axes
        else:
            ax_od = axes if isinstance(axes, np.ndarray) else axes

        # Object Detection Visualization
        ax_od.imshow(image)

        # Overlay predicted bounding boxes
        for pred in object_detection_output:
            x, y, w, h = pred["bbox"]  # Assuming format [x, y, width, height]
            rect = plt.Rectangle((x, y), w, h, edgecolor="red", fill=False, linewidth=2)
            ax_od.add_patch(rect)
            ax_od.text(
                x,
                y,
                f'{pred["category"]}: {pred["score"]:.2f}',
                bbox=dict(facecolor="white", alpha=0.7),
                fontsize=8,
            )

        # Optional: Overlay ground truth boxes
        if ground_truth:
            for gt in ground_truth:
                x, y, w, h = gt["bbox"]
                rect = plt.Rectangle(
                    (x, y), w, h, edgecolor="green", fill=False, linewidth=1
                )
                ax_od.add_patch(rect)
                ax_od.text(
                    x,
                    y + h + 5,
                    gt["category"],
                    bbox=dict(facecolor="green", alpha=0.7),
                    fontsize=8,
                )

        ax_od.set_title("Object Detection")
        ax_od.axis("off")

        # Depth Estimation Visualization
        if depth_estimation_output is not None:
            depth_map = (
                depth_estimation_output["predicted_depth"].squeeze().cpu().numpy()
            )

            # Normalize depth map for visualization
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )

            im = ax_depth.imshow(depth_map, cmap="plasma")
            ax_depth.set_title("Depth Estimation")
            ax_depth.axis("off")

            # Add colorbar
            divider = make_axes_locatable(ax_depth)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.show()

    def create_comparison_graphs(self, all_results):
        for plot_type in self.config.PLOT_TYPES:
            if plot_type == "bar_chart":
                self._create_bar_charts(all_results)
            elif plot_type == "radar_chart":
                self._create_radar_charts(all_results)
            elif plot_type == "scatter_plot":
                self._create_scatter_plots(all_results)
            elif plot_type == "box_plot":
                self._create_box_plots(all_results)

    def _create_bar_charts(self, all_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        self._plot_bar_chart(
            ax1,
            all_results["object_detection"],
            "mAP",
            "Object Detection - Mean Average Precision",
        )
        self._plot_bar_chart(
            ax2,
            all_results["depth_estimation"],
            "RMSE",
            "Depth Estimation - Root Mean Square Error",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, "bar_charts.png"))
        plt.close()

    def _plot_bar_chart(self, ax, results, metric, title):
        models = list(results.keys())
        values = [
            np.mean([variant[metric] for variant in model_results.values()])
            for model_results in results.values()
        ]

        ax.bar(models, values)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xticklabels(models, rotation=45, ha="right")

    def _create_radar_charts(self, all_results):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(20, 10), subplot_kw=dict(projection="polar")
        )

        self._plot_radar_chart(
            ax1,
            all_results["object_detection"],
            self.config.OBJECT_DETECTION_METRICS,
            "Object Detection Metrics",
        )
        self._plot_radar_chart(
            ax2,
            all_results["depth_estimation"],
            self.config.DEPTH_ESTIMATION_METRICS,
            "Depth Estimation Metrics",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, "radar_charts.png"))
        plt.close()

    def _plot_radar_chart(self, ax, results, metrics, title):
        models = list(results.keys())
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

        for model, model_results in results.items():
            values = [
                np.mean([variant[metric] for variant in model_results.values()])
                for metric in metrics
            ]
            values = np.concatenate(
                (values, [values[0]])
            )  # repeat the first value to close the polygon
            ax.plot(np.concatenate((angles, [angles[0]])), values, label=model)
            ax.fill(np.concatenate((angles, [angles[0]])), values, alpha=0.1)

        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        ax.set_title(title)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    def _create_scatter_plots(self, all_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        self._plot_scatter(
            ax1,
            all_results["object_detection"],
            "mAP",
            "inference_time",
            "Object Detection",
        )
        self._plot_scatter(
            ax2,
            all_results["depth_estimation"],
            "RMSE",
            "inference_time",
            "Depth Estimation",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, "scatter_plots.png"))
        plt.close()

    def _plot_scatter(self, ax, results, metric_x, metric_y, title):
        for model, model_results in results.items():
            x = [variant[metric_x] for variant in model_results.values()]
            y = [variant[metric_y] for variant in model_results.values()]
            ax.scatter(x, y, label=model)

        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(title)
        ax.legend()

    def _create_box_plots(self, all_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        self._plot_box(
            ax1,
            all_results["object_detection"],
            self.config.COMPARISON_METRICS,
            "Object Detection",
        )
        self._plot_box(
            ax2,
            all_results["depth_estimation"],
            self.config.COMPARISON_METRICS,
            "Depth Estimation",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, "box_plots.png"))
        plt.close()

    def _plot_box(self, ax, results, metrics, title):
        data = []
        labels = []
        for metric in metrics:
            for model, model_results in results.items():
                values = [variant[metric] for variant in model_results.values()]
                data.append(values)
                labels.append(f"{model}\n({metric})")

        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)


