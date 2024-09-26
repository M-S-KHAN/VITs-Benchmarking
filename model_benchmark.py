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


class ModelBenchmark:
    def __init__(self):
        self.config = BenchmarkConfig()
        self.device = torch.device(
            self.config.DEVICE if torch.cuda.is_available() else "cpu"
        )

    def run_inference(self, feature_extractor, model, image):
        inputs = feature_extractor(images=image, return_tensors="pt").to(self.device)
        model.to(self.device)

        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()

        inference_time = end_time - start_time
        return outputs, inference_time

    def benchmark_model(self, model_type, model_name, variant, image_paths):
        feature_extractor, model = self.config.load_model(
            model_type, model_name, variant
        )
        results = []

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            outputs, inference_time = self.run_inference(
                feature_extractor, model, image
            )

            result = {
                "image_path": image_path,
                "outputs": outputs,
                "inference_time": inference_time,
            }
            results.append(result)

        return results

    def calculate_metrics(self, results, ground_truth, model_type):
        if model_type == "object_detection":
            return self.calculate_object_detection_metrics(results, ground_truth)
        elif model_type == "depth_estimation":
            return self.calculate_depth_estimation_metrics(results, ground_truth)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def calculate_object_detection_metrics(self, results, ground_truth):
        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox")

        for result, gt in zip(results, ground_truth):
            preds = [
                {
                    "boxes": result["outputs"]["pred_boxes"],
                    "scores": result["outputs"]["pred_scores"],
                    "labels": result["outputs"]["pred_labels"],
                }
            ]

            target = [
                {
                    "boxes": torch.tensor([ann["bbox"] for ann in gt]),
                    "labels": torch.tensor([ann["category_id"] for ann in gt]),
                }
            ]

            metric.update(preds, target)

        metrics = metric.compute()

        return {
            "mAP": metrics["map"].item(),
            "mAP_50": metrics["map_50"].item(),
            "mAP_75": metrics["map_75"].item(),
            "mAP_small": metrics["map_small"].item(),
            "mAP_medium": metrics["map_medium"].item(),
            "mAP_large": metrics["map_large"].item(),
            "average_precision": metrics["map_per_class"].mean().item(),
            "average_recall": metrics["mar_100"].item(),
            "inference_time": np.mean([r["inference_time"] for r in results]),
        }

    def calculate_depth_estimation_metrics(self, results, ground_truth):
        rmse_list, mae_list, ssim_list = [], [], []
        delta1_list, delta2_list, delta3_list = [], [], []

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        for result, gt in zip(results, ground_truth):
            pred_depth = result["outputs"]["predicted_depth"].squeeze().cpu().numpy()
            true_depth = gt["depth"].squeeze().cpu().numpy()

            # Normalize depths
            pred_depth = (pred_depth - pred_depth.min()) / (
                pred_depth.max() - pred_depth.min()
            )
            true_depth = (true_depth - true_depth.min()) / (
                true_depth.max() - true_depth.min()
            )

            # RMSE
            rmse = np.sqrt(mean_squared_error(true_depth, pred_depth))
            rmse_list.append(rmse)

            # MAE
            mae = mean_absolute_error(true_depth, pred_depth)
            mae_list.append(mae)

            # SSIM
            ssim = ssim_metric(
                torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(true_depth).unsqueeze(0).unsqueeze(0),
            )
            ssim_list.append(ssim.item())

            # Delta metrics
            thresh = np.maximum((true_depth / pred_depth), (pred_depth / true_depth))
            delta1 = (thresh < 1.25).mean()
            delta2 = (thresh < 1.25**2).mean()
            delta3 = (thresh < 1.25**3).mean()

            delta1_list.append(delta1)
            delta2_list.append(delta2)
            delta3_list.append(delta3)

        return {
            "RMSE": np.mean(rmse_list),
            "MAE": np.mean(mae_list),
            "SSIM": np.mean(ssim_list),
            "delta1": np.mean(delta1_list),
            "delta2": np.mean(delta2_list),
            "delta3": np.mean(delta3_list),
            "inference_time": np.mean([r["inference_time"] for r in results]),
        }

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

    def run_benchmark(self, image_paths, ground_truth):
        all_results = {"object_detection": {}, "depth_estimation": {}}

        for model_type in ["object_detection", "depth_estimation"]:
            models = (
                self.config.OBJECT_DETECTION_MODELS
                if model_type == "object_detection"
                else self.config.DEPTH_ESTIMATION_MODELS
            )

            for model_name, variants in models.items():
                model_results = {}
                for variant in variants:
                    print(f"Benchmarking {model_type} - {model_name} - {variant}")
                    results = self.benchmark_model(
                        model_type, model_name, variant, image_paths
                    )
                    metrics = self.calculate_metrics(results, ground_truth, model_type)
                    model_results[variant] = metrics
                all_results[model_type][model_name] = model_results

        self.create_comparison_graphs(all_results)
        return all_results
