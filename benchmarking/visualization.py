import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
import os
import pandas as pd

class VisualizationUtils:
    @staticmethod
    def save_plot(fig, filename):
        """Save the plot to a file."""
        os.makedirs('plots', exist_ok=True)
        fig.savefig(os.path.join('plots', filename), bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def create_comparison_graphs(all_results, task_type):
        """Creates various comparison graphs for multiple models and save them."""
        if task_type == "depth":
            VisualizationUtils.plot_depth_bar_chart(all_results)
            VisualizationUtils.plot_depth_scatter_plot(all_results)
            VisualizationUtils.plot_depth_box_plot(all_results)
            VisualizationUtils.plot_depth_heatmap(all_results)
            VisualizationUtils.plot_depth_radar_chart(all_results)
        elif task_type == "detection":
            VisualizationUtils.plot_detection_bar_chart(all_results)
            VisualizationUtils.plot_detection_scatter_plot(all_results)
            VisualizationUtils.plot_detection_box_plot(all_results)
            VisualizationUtils.plot_detection_heatmap(all_results)
            VisualizationUtils.plot_detection_radar_chart(all_results)
        VisualizationUtils.plot_inference_time_comparison(all_results, task_type)
        VisualizationUtils.plot_performance_vs_complexity(all_results, task_type)

    @staticmethod
    def plot_depth_bar_chart(all_results):
        """Plot average metrics as bar charts for depth estimation and save them."""
        metrics = ['obj_size_consistency', 'depth_consistency', 'edge_comparison']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Comparison of Depth Estimation Models', fontsize=16)
        
        for i, metric in enumerate(metrics):
            values = [result['avg_metrics'][metric] for result in all_results]
            axes[i].bar(model_names, values)
            axes[i].set_title(metric)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'depth_estimation_bar_charts.png')

    @staticmethod
    def plot_detection_bar_chart(all_results):
        """Plot average metrics as bar charts for object detection and save them."""
        metrics = ['mean_iou', 'ap_50', 'ap_75']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Comparison of Object Detection Models', fontsize=16)
        
        for i, metric in enumerate(metrics):
            values = [result['avg_metrics'][metric] for result in all_results]
            axes[i].bar(model_names, values)
            axes[i].set_title(metric)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'object_detection_bar_charts.png')

    @staticmethod
    def plot_depth_scatter_plot(all_results):
        """Plot metrics as scatter plots for depth estimation and save it."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for result in all_results:
            ax.scatter(result['avg_metrics']['depth_consistency'], 
                       result['avg_metrics']['edge_comparison'], 
                       label=result['model_name'].split('/')[-1], s=100)
        
        ax.set_xlabel('Depth Consistency')
        ax.set_ylabel('Edge Comparison')
        ax.set_title('Depth Estimation Model Comparison')
        ax.legend()
        
        VisualizationUtils.save_plot(fig, 'depth_estimation_scatter_plot.png')

    @staticmethod
    def plot_detection_scatter_plot(all_results):
        """Plot metrics as scatter plots for object detection and save it."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for result in all_results:
            ax.scatter(result['avg_metrics']['mean_iou'], 
                       result['avg_metrics']['ap_50'], 
                       label=result['model_name'].split('/')[-1], s=100)
        
        ax.set_xlabel('Mean IoU')
        ax.set_ylabel('AP@50')
        ax.set_title('Object Detection Model Comparison')
        ax.legend()
        
        VisualizationUtils.save_plot(fig, 'object_detection_scatter_plot.png')

    @staticmethod
    def plot_inference_time_comparison(all_results, task_type):
        """Plot inference time comparison for both tasks and save it."""
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        inference_times = [result['avg_metrics']['avg_inference_time'] for result in all_results]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(model_names, inference_times)
        ax.set_ylabel('Average Inference Time (s)')
        ax.set_title(f'Inference Time Comparison - {task_type.capitalize()} Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        VisualizationUtils.save_plot(fig, f'{task_type}_inference_time_comparison.png')

    @staticmethod
    def plot_depth_box_plot(all_results):
        """Plot box plots for depth estimation metrics."""
        metrics = ['obj_size_consistency', 'depth_consistency', 'edge_comparison']
        data = {metric: [] for metric in metrics}
        model_names = []

        for result in all_results:
            model_names.append(result['model_name'].split('/')[-1])
            for metric in metrics:
                data[metric].extend([r['metrics'][metric] for r in result['detailed_results']])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot([data[m] for m in metrics], labels=metrics)
        ax.set_title('Distribution of Depth Estimation Metrics')
        ax.set_ylabel('Metric Value')
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'depth_estimation_box_plot.png')

    @staticmethod
    def plot_detection_box_plot(all_results):
        """Plot box plots for object detection metrics."""
        metrics = ['mean_iou', 'ap_50', 'ap_75']
        data = {metric: [] for metric in metrics}
        model_names = []

        for result in all_results:
            model_names.append(result['model_name'].split('/')[-1])
            for metric in metrics:
                data[metric].extend([r['metrics'][metric] for r in result['detailed_results']])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot([data[m] for m in metrics], labels=metrics)
        ax.set_title('Distribution of Object Detection Metrics')
        ax.set_ylabel('Metric Value')
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'object_detection_box_plot.png')

    @staticmethod
    def plot_depth_heatmap(all_results):
        """Plot heatmap of depth estimation metrics."""
        metrics = ['obj_size_consistency', 'depth_consistency', 'edge_comparison', 'avg_inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        data = np.array([[result['avg_metrics'][m] for m in metrics] for result in all_results])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, annot=True, xticklabels=metrics, yticklabels=model_names, ax=ax)
        ax.set_title('Heatmap of Depth Estimation Metrics')
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'depth_estimation_heatmap.png')

    @staticmethod
    def plot_detection_heatmap(all_results):
        """Plot heatmap of object detection metrics."""
        metrics = ['mean_iou', 'ap_50', 'ap_75', 'avg_inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        data = np.array([[result['avg_metrics'][m] for m in metrics] for result in all_results])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, annot=True, xticklabels=metrics, yticklabels=model_names, ax=ax)
        ax.set_title('Heatmap of Object Detection Metrics')
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'object_detection_heatmap.png')

    @staticmethod
    def plot_depth_radar_chart(all_results):
        """Plot radar chart for depth estimation metrics."""
        metrics = ['obj_size_consistency', 'depth_consistency', 'edge_comparison', 'avg_inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]

        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        for result in all_results:
            values = [result['avg_metrics'][m] for m in metrics]
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'].split('/')[-1])
            ax.fill(angles, values, alpha=0.25)

        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
        ax.set_title('Radar Chart of Depth Estimation Metrics')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'depth_estimation_radar_chart.png')

    @staticmethod
    def plot_detection_radar_chart(all_results):
        """Plot radar chart for object detection metrics."""
        metrics = ['mean_iou', 'ap_50', 'ap_75', 'avg_inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]

        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        for result in all_results:
            values = [result['avg_metrics'][m] for m in metrics]
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'].split('/')[-1])
            ax.fill(angles, values, alpha=0.25)

        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
        ax.set_title('Radar Chart of Object Detection Metrics')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'object_detection_radar_chart.png')

    @staticmethod
    def plot_performance_vs_complexity(all_results, task_type):
        """Plot performance metrics against model complexity."""
        if task_type == "depth":
            performance_metric = 'edge_comparison'
        else:  # detection
            performance_metric = 'mean_iou'

        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        performance = [result['avg_metrics'][performance_metric] for result in all_results]
        inference_times = [result['avg_metrics']['avg_inference_time'] for result in all_results]
        
        # Use model name length as a proxy for complexity (you may want to replace this with a more meaningful measure)
        complexity = [len(name) for name in model_names]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.scatter(complexity, performance, color='blue', label=performance_metric)
        ax1.set_xlabel('Model Complexity (proxy)')
        ax1.set_ylabel(performance_metric, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.scatter(complexity, inference_times, color='red', label='Inference Time')
        ax2.set_ylabel('Inference Time (s)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f'{task_type.capitalize()} Performance vs Complexity')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, f'{task_type}_performance_vs_complexity.png')

    @staticmethod
    def visualize_depth_map(depth_map, filename='depth_map.png'):
        """Visualizes a depth map with a color gradient and save it."""
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(depth_map, cmap='plasma')
        plt.colorbar(im, label='Depth')
        plt.title('Depth Map')
        plt.axis('off')
        VisualizationUtils.save_plot(fig, filename)

    @staticmethod
    def visualize_detections(image_path, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_categories, filename='detections.png'):
        """Visualize predicted and ground truth bounding boxes on an image and save it."""
        image = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # Plot predicted boxes
        for i, box in enumerate(pred_boxes):
            x, y, w, h = box
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f'{pred_labels[i]}: {pred_scores[i]:.2f}', color='red', fontsize=12)

        # Plot ground truth boxes
        for i, box in enumerate(gt_boxes):
            x, y, w, h = box
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f'{gt_categories[i]}', color='blue', fontsize=12)

        plt.axis('off')
        VisualizationUtils.save_plot(fig, filename)

    @staticmethod
    def plot_precision_recall_curve(all_results, task_type):
        """Plot precision-recall curve for object detection models."""
        if task_type != "detection":
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        for result in all_results:
            precisions = []
            recalls = []
            for det_result in result['detailed_results']:
                precision, recall, _ = precision_recall_curve(det_result['gt_labels'], det_result['pred_scores'])
                precisions.append(precision)
                recalls.append(recall)
            
            avg_precision = np.mean(precisions, axis=0)
            avg_recall = np.mean(recalls, axis=0)
            
            ax.plot(avg_recall, avg_precision, label=result['model_name'].split('/')[-1])

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'precision_recall_curve.png')

    @staticmethod
    def plot_confusion_matrix(all_results, task_type):
        """Plot confusion matrix for object detection models."""
        if task_type != "detection":
            return

        for result in all_results:
            all_pred_labels = []
            all_gt_labels = []
            for det_result in result['detailed_results']:
                all_pred_labels.extend(det_result['pred_labels'])
                all_gt_labels.extend(det_result['gt_labels'])

            cm = confusion_matrix(all_gt_labels, all_pred_labels)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Confusion Matrix - {result["model_name"].split("/")[-1]}')
            plt.tight_layout()
            VisualizationUtils.save_plot(fig, f'confusion_matrix_{result["model_name"].split("/")[-1]}.png')

    @staticmethod
    def plot_metric_distribution(all_results, task_type):
        """Plot distribution of metrics for all models."""
        if task_type == "depth":
            metrics = ['obj_size_consistency', 'depth_consistency', 'edge_comparison']
        else:  # detection
            metrics = ['mean_iou', 'ap_50', 'ap_75']

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
        for i, metric in enumerate(metrics):
            data = []
            labels = []
            for result in all_results:
                data.append([r['metrics'][metric] for r in result['detailed_results']])
                labels.append(result['model_name'].split('/')[-1])
            
            axes[i].violinplot(data, showmeans=True, showmedians=True)
            axes[i].set_title(f'Distribution of {metric}')
            axes[i].set_xticks(range(1, len(labels) + 1))
            axes[i].set_xticklabels(labels, rotation=45, ha='right')

        plt.tight_layout()
        VisualizationUtils.save_plot(fig, f'{task_type}_metric_distribution.png')

    @staticmethod
    def plot_performance_comparison(depth_results, detection_results):
        """Plot performance comparison between depth estimation and object detection models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Depth estimation performance
        depth_models = [r['model_name'].split('/')[-1] for r in depth_results]
        depth_performance = [r['avg_metrics']['edge_comparison'] for r in depth_results]
        ax1.bar(depth_models, depth_performance)
        ax1.set_title('Depth Estimation Performance')
        ax1.set_ylabel('Edge Comparison Score')
        ax1.set_xticklabels(depth_models, rotation=45, ha='right')

        # Object detection performance
        detection_models = [r['model_name'].split('/')[-1] for r in detection_results]
        detection_performance = [r['avg_metrics']['mean_iou'] for r in detection_results]
        ax2.bar(detection_models, detection_performance)
        ax2.set_title('Object Detection Performance')
        ax2.set_ylabel('Mean IoU')
        ax2.set_xticklabels(detection_models, rotation=45, ha='right')

        plt.tight_layout()
        VisualizationUtils.save_plot(fig, 'performance_comparison.png')
