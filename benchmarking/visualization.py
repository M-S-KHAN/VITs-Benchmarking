import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class VisualizationUtils:
    @staticmethod
    def plot_bounding_boxes(image, boxes, labels=None, scores=None):
        """Plot bounding boxes on an image."""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            if labels is not None and scores is not None:
                ax.text(x, y, f'{labels[i]}: {scores[i]:.2f}', color='red', fontsize=12)
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_depth_map(depth_map):
        """Visualizes a depth map with a color gradient."""
        plt.figure(figsize=(12, 8))
        plt.imshow(depth_map, cmap='plasma')
        plt.colorbar(label='Depth')
        plt.title('Depth Map')
        plt.axis('off')
        plt.show()

    @staticmethod
    def create_comparison_graphs(all_results):
        """Creates various comparison graphs for multiple models."""
        VisualizationUtils.plot_bar_chart(all_results)
        VisualizationUtils.plot_radar_chart(all_results)
        VisualizationUtils.plot_scatter_plot(all_results)
        VisualizationUtils.plot_box_plot(all_results)

    @staticmethod
    def plot_bar_chart(all_results):
        """Plot average metrics as bar charts."""
        metrics = ['mean_iou', 'ap_50', 'ap_75', 'avg_inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Comparison of Models - Bar Charts', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [result['avg_metrics'][metric] for result in all_results]
            ax.bar(model_names, values)
            ax.set_title(metric)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_radar_chart(all_results):
        """Plot metrics as radar charts for comparative analysis."""
        metrics = ['mean_iou', 'ap_50', 'ap_75', 'avg_inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for result in all_results:
            values = [result['avg_metrics'][metric] for metric in metrics]
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'].split('/')[-1])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
        ax.set_title('Model Comparison - Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.show()

    @staticmethod
    def plot_scatter_plot(all_results):
        """Plot metrics as scatter plots."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for result in all_results:
            ax.scatter(result['avg_metrics']['mean_iou'], result['avg_metrics']['avg_inference_time'], 
                      label=result['model_name'].split('/')[-1], s=100)
        
        ax.set_xlabel('Mean IoU')
        ax.set_ylabel('Average Inference Time')
        ax.set_title('Model Comparison - Mean IoU vs Inference Time')
        ax.legend()
        
        plt.show()

    @staticmethod
    def plot_box_plot(all_results):
        """Plot metrics as box plots."""
        metrics = ['mean_iou', 'ap_50', 'ap_75', 'inference_time']
        model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Comparison of Models - Box Plots', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            data = []
            for result in all_results:
                metric_values = [item['metrics'][metric] for item in result['detailed_results'] if metric in item['metrics']]
                data.append(metric_values)
            
            ax.boxplot(data, labels=model_names)
            ax.set_title(metric)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()