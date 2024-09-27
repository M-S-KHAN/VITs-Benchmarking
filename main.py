# import torch
# from transformers import YolosFeatureExtractor, YolosForObjectDetection, DetrFeatureExtractor, DetrForObjectDetection, OwlViTProcessor, OwlViTForObjectDetection
# from torchvision.ops import box_iou
# from PIL import Image
# import json
# import time
# from tqdm import tqdm
# from data_loader import DataLoader
# from utils import load_benchmark_data
# import logging
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.patches import Rectangle

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# MODELS_TO_BENCHMARK = [
#     ("OWL-ViT", "google/owlvit-base-patch32"),
#     ("DETR", "facebook/detr-resnet-50"),
#     # ("DETR", "facebook/detr-resnet-101"),
#     ("YOLOS", "hustvl/yolos-small"),
#     # ("YOLOS", "hustvl/yolos-base"),
# ]

# class VisualizationUtils:
#     @staticmethod
#     def plot_bounding_boxes(image, boxes, labels=None, scores=None):
#         """Plot bounding boxes on an image."""
#         fig, ax = plt.subplots(1, figsize=(12, 8))
#         ax.imshow(image)
#         for i, box in enumerate(boxes):
#             x, y, w, h = box
#             rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
#             ax.add_patch(rect)
#             if labels is not None and scores is not None:
#                 ax.text(x, y, f'{labels[i]}: {scores[i]:.2f}', color='red', fontsize=12)
#         plt.axis('off')
#         plt.show()

#     @staticmethod
#     def plot_depth_map(depth_map):
#         """Visualizes a depth map with a color gradient."""
#         plt.figure(figsize=(12, 8))
#         plt.imshow(depth_map, cmap='plasma')
#         plt.colorbar(label='Depth')
#         plt.title('Depth Map')
#         plt.axis('off')
#         plt.show()

#     @staticmethod
#     def create_comparison_graphs(all_results):
#         """Creates various comparison graphs for multiple models."""
#         VisualizationUtils.plot_bar_chart(all_results)
#         VisualizationUtils.plot_radar_chart(all_results)
#         VisualizationUtils.plot_scatter_plot(all_results)
#         VisualizationUtils.plot_box_plot(all_results)

#     @staticmethod
#     def plot_bar_chart(all_results):
#         """Plot average metrics as bar charts."""
#         metrics = ['mean_iou', 'ap_50', 'ap_75', 'avg_inference_time']
#         model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
#         fig, axes = plt.subplots(2, 2, figsize=(20, 15))
#         fig.suptitle('Comparison of Models - Bar Charts', fontsize=16)
        
#         for i, metric in enumerate(metrics):
#             ax = axes[i // 2, i % 2]
#             values = [result['avg_metrics'][metric] for result in all_results]
#             ax.bar(model_names, values)
#             ax.set_title(metric)
#             ax.set_xticklabels(model_names, rotation=45, ha='right')
        
#         plt.tight_layout()
#         plt.show()

#     @staticmethod
#     def plot_radar_chart(all_results):
#         """Plot metrics as radar charts for comparative analysis."""
#         metrics = ['mean_iou', 'ap_50', 'ap_75', 'avg_inference_time']
#         model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
#         angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
#         angles = np.concatenate((angles, [angles[0]]))
        
#         fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
#         for result in all_results:
#             values = [result['avg_metrics'][metric] for metric in metrics]
#             values = np.concatenate((values, [values[0]]))
#             ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'].split('/')[-1])
#             ax.fill(angles, values, alpha=0.25)
        
#         ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
#         ax.set_title('Model Comparison - Radar Chart')
#         ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
#         plt.show()

#     @staticmethod
#     def plot_scatter_plot(all_results):
#         """Plot metrics as scatter plots."""
#         fig, ax = plt.subplots(figsize=(12, 8))
        
#         for result in all_results:
#             ax.scatter(result['avg_metrics']['mean_iou'], result['avg_metrics']['avg_inference_time'], 
#                       label=result['model_name'].split('/')[-1], s=100)
        
#         ax.set_xlabel('Mean IoU')
#         ax.set_ylabel('Average Inference Time')
#         ax.set_title('Model Comparison - Mean IoU vs Inference Time')
#         ax.legend()
        
#         plt.show()

#     @staticmethod
#     def plot_box_plot(all_results):
#         """Plot metrics as box plots."""
#         metrics = ['mean_iou', 'ap_50', 'ap_75', 'inference_time']
#         model_names = [result['model_name'].split('/')[-1] for result in all_results]
        
#         fig, axes = plt.subplots(2, 2, figsize=(20, 15))
#         fig.suptitle('Comparison of Models - Box Plots', fontsize=16)
        
#         for i, metric in enumerate(metrics):
#             ax = axes[i // 2, i % 2]
#             data = []
#             for result in all_results:
#                 metric_values = [item['metrics'][metric] for item in result['detailed_results'] if metric in item['metrics']]
#                 data.append(metric_values)
            
#             ax.boxplot(data, labels=model_names)
#             ax.set_title(metric)
#             ax.set_xticklabels(model_names, rotation=45, ha='right')
        
#         plt.tight_layout()
#         plt.show()

# class BoxConversionFactory:
#     @staticmethod
#     def get_converter(model_type):
#         if model_type == "YOLOS":
#             return BoxConversionFactory.convert_yolos_format
#         if model_type == "DETR":
#             return BoxConversionFactory.convert_detr_format
#         elif model_type == "OWL-ViT":
#             return BoxConversionFactory.convert_owlvit_format
#         else:
#             raise ValueError(f"Unsupported model type: {model_type}")

#     @staticmethod
#     def convert_yolos_format(boxes, image_size):
#         """Convert relative [center_x, center_y, width, height] to absolute [x1, y1, x2, y2]"""
#         height, width = image_size
#         cx, cy, w, h = boxes.unbind(-1)
#         x1 = (cx - 0.5 * w) * width
#         y1 = (cy - 0.5 * h) * height
#         x2 = (cx + 0.5 * w) * width
#         y2 = (cy + 0.5 * h) * height
#         return torch.stack((x1, y1, x2, y2), dim=-1)

#     @staticmethod
#     def convert_owlvit_format(boxes, image_size):
#         """Convert normalized [x1, y1, x2, y2] to absolute [x1, y1, x2, y2]"""
#         height, width = image_size
#         x1, y1, x2, y2 = boxes.unbind(-1)
#         x1 = x1 * width
#         y1 = y1 * height
#         x2 = x2 * width
#         y2 = y2 * height
#         return torch.stack((x1, y1, x2, y2), dim=-1)

#     @staticmethod
#     def convert_detr_format(boxes, image_size):
#         """Convert normalized [center_x, center_y, width, height] to absolute [x1, y1, x2, y2]"""
#         height, width = image_size
#         cx, cy, w, h = boxes.unbind(-1)
#         x1 = (cx - 0.5 * w) * width
#         y1 = (cy - 0.5 * h) * height
#         x2 = (cx + 0.5 * w) * width
#         y2 = (cy + 0.5 * h) * height
#         return torch.stack((x1, y1, x2, y2), dim=-1)

#     @staticmethod
#     def convert_gt_format(boxes, image_size):
#         """Convert absolute [x, y, width, height] to [x1, y1, x2, y2]"""
#         x, y, w, h = boxes.unbind(-1)
#         x2 = x + w
#         y2 = y + h
#         return torch.stack((x, y, x2, y2), dim=-1)

# class Benchmark:
#     def __init__(self, model_type, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
#         self.model_type = model_type
#         self.model_name = model_name
#         self.device = device
#         self.converter = BoxConversionFactory.get_converter(model_type)
#         self.initialize_model()

#     def initialize_model(self):
#         logger.info(f"Initializing {self.model_type} model: {self.model_name} on device: {self.device}")
#         try:
#             if self.model_type == "YOLOS":
#                 self.feature_extractor = YolosFeatureExtractor.from_pretrained(self.model_name)
#                 self.model = YolosForObjectDetection.from_pretrained(self.model_name).to(self.device)
#             elif self.model_type == "DETR":
#                 self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_name)
#                 self.model = DetrForObjectDetection.from_pretrained(self.model_name).to(self.device)
#             elif self.model_type == "OWL-ViT":
#                 self.feature_extractor = OwlViTProcessor.from_pretrained(self.model_name)
#                 self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
#             else:
#                 raise ValueError(f"Unsupported model type: {self.model_type}")
            
#             self.model.eval()
#             logger.info("Model initialized successfully")
#         except Exception as e:
#             logger.error(f"Error initializing model: {e}")
#             raise

#     def preprocess_image(self, image_path):
#       logger.info(f"Preprocessing image: {image_path}")
#       try:
#           image = Image.open(image_path).convert("RGB")
#           if self.model_type == "OWL-ViT":
#               # Use multiple generic queries
#               text_queries = ["object", "thing", "item", "person", "animal", "vehicle", "furniture"]
#               return self.feature_extractor(text=text_queries, images=image, return_tensors="pt")
#           else:
#               return self.feature_extractor(images=image, return_tensors="pt")
#       except Exception as e:
#           logger.error(f"Error preprocessing image: {e}")
#           raise

#     def postprocess_predictions(self, outputs, confidence_threshold=0.5):
#       logger.info("Postprocessing predictions")
#       try:
#           if self.model_type in ["YOLOS", "DETR"]:
#               probas = outputs.logits.softmax(-1)[0, :, :-1]
#               keep = probas.max(-1).values > confidence_threshold
              
#               boxes = outputs.pred_boxes[0, keep]
#               labels = probas[keep].argmax(-1)
#               scores = probas[keep].max(-1).values
#           elif self.model_type == "OWL-ViT":
#               # Lower confidence threshold for OWL-ViT
#               owl_vit_threshold = 0.1
#               target_sizes = torch.tensor([1.0, 1.0]).unsqueeze(0)  # Assuming normalized coordinates
#               results = self.feature_extractor.post_process_object_detection(outputs, threshold=owl_vit_threshold, target_sizes=target_sizes)[0]
#               boxes = results["boxes"]
#               labels = results["labels"]
#               scores = results["scores"]
              
#               # Print detailed information about OWL-ViT outputs
#               logger.info(f"OWL-ViT raw output: Boxes: {boxes.shape}, Labels: {labels.shape}, Scores: {scores.shape}")
#               logger.info(f"OWL-ViT scores: {scores}")
#               logger.info(f"OWL-ViT labels: {labels}")
#           else:
#               raise ValueError(f"Unsupported model type: {self.model_type}")
          
#           logger.info(f"Postprocessing complete. Boxes: {boxes.shape}, Labels: {labels.shape}, Scores: {scores.shape}")
#           return boxes.cpu(), labels.cpu(), scores.cpu()
#       except Exception as e:
#           logger.error(f"Error postprocessing predictions: {e}")
#           raise

#     def run_inference(self, image):
#         logger.info("Running inference")
#         try:
#             inputs = {k: v.to(self.device) for k, v in image.items()}
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#             logger.info("Inference completed successfully")
#             return outputs
#         except Exception as e:
#             logger.error(f"Error during inference: {e}")
#             raise

#     def calculate_metrics(self, pred_boxes, gt_boxes, image_size):
#         logger.info(f"Calculating metrics. Pred boxes: {pred_boxes.shape}, GT boxes: {gt_boxes.shape}")
#         if pred_boxes.nelement() == 0 or gt_boxes.nelement() == 0:
#             logger.warning(f"Empty prediction or ground truth. pred_boxes: {pred_boxes.shape}, gt_boxes: {gt_boxes.shape}")
#             return {"mean_iou": 0, "ap_50": 0, "ap_75": 0}

#         pred_boxes = self.converter(pred_boxes, image_size)
#         gt_boxes = BoxConversionFactory.convert_gt_format(gt_boxes, image_size)

#         try:
#             iou = box_iou(pred_boxes, gt_boxes)
#             metrics = {
#                 "mean_iou": iou.mean().item(),
#                 "ap_50": (iou.max(dim=1)[0] > 0.5).float().mean().item(),
#                 "ap_75": (iou.max(dim=1)[0] > 0.75).float().mean().item(),
#             }
#             logger.info(f"Metrics calculated: {metrics}")
#             return metrics
#         except Exception as e:
#             logger.error(f"Error calculating metrics: {e}")
#             logger.error(f"pred_boxes: {pred_boxes}")
#             logger.error(f"gt_boxes: {gt_boxes}")
#             return {"mean_iou": 0, "ap_50": 0, "ap_75": 0}

#     def benchmark(self, benchmark_data):
#         results = []
#         for item in tqdm(benchmark_data):
#             logger.info(f"Processing image: {item['image_path']}")
#             image_path = item['image_path']
#             annotations = item['annotations']
            
#             try:
#                 start_time = time.time()
                
#                 image = self.preprocess_image(image_path)
#                 outputs = self.run_inference(image)
#                 pred_boxes, pred_labels, pred_scores = self.postprocess_predictions(outputs)
                
#                 inference_time = time.time() - start_time
                
#                 with Image.open(image_path) as img:
#                     image_size = img.size[::-1]  # (height, width)
                
#                 gt_boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
#                 metrics = self.calculate_metrics(pred_boxes, gt_boxes, image_size)
#                 metrics["inference_time"] = inference_time
                
#                 results.append({
#                     "image_path": image_path,
#                     "pred_boxes": pred_boxes.tolist(),
#                     "pred_labels": pred_labels.tolist(),
#                     "pred_scores": pred_scores.tolist(),
#                     "gt_boxes": gt_boxes.tolist(),
#                     "gt_categories": [ann['category'] for ann in annotations],
#                     "metrics": metrics
#                 })
#                 logger.info(f"Processed image: {image_path}. Metrics: {metrics}")
#             except Exception as e:
#                 logger.error(f"Error processing image {image_path}: {e}")
        
#         return results

# def run_benchmark(model_type, model_name, benchmark_data):
#     logger.info(f"Starting benchmark for {model_name}")
#     print(f"Starting benchmark for {model_name}")
#     benchmark = Benchmark(model_type, model_name)
#     results = benchmark.benchmark(benchmark_data)
    
#     avg_metrics = {
#         "mean_iou": sum(r["metrics"]["mean_iou"] for r in results) / len(results),
#         "ap_50": sum(r["metrics"]["ap_50"] for r in results) / len(results),
#         "ap_75": sum(r["metrics"]["ap_75"] for r in results) / len(results),
#         "avg_inference_time": sum(r["metrics"]["inference_time"] for r in results) / len(results)
#     }
    
#     logger.info(f"Average Metrics for {model_name}:")
#     for metric, value in avg_metrics.items():
#         logger.info(f"{metric}: {value:.4f}")
    
#     return {
#         "model_name": model_name,
#         "avg_metrics": avg_metrics,
#         "detailed_results": results
#     }

# def main():
#     existing_data = load_benchmark_data()
#     if existing_data:
#         benchmark_data = existing_data
#     else:
#         data_loader = DataLoader()
#         benchmark_data = data_loader.load_benchmark_data()

#     all_results = []
    
#     for model_type, model_name in MODELS_TO_BENCHMARK:
#         print(f"Starting benchmark for {model_name}")
#         try:
#             print(f'{model_type} {model_name}')
#             result = run_benchmark(model_type, model_name, benchmark_data)
#             all_results.append(result)
#         except Exception as e:
#             logger.error(f"Error benchmarking {model_name}: {e}")
    
#     print("\nComparison of Average Metrics:")
#     metrics = ["mean_iou", "ap_50", "ap_75", "avg_inference_time"]
#     print(f"{'Model':<30} " + " ".join(f"{m:>12}" for m in metrics))
#     for result in all_results:
#         model_name = result["model_name"].split("/")[-1]
#         metrics_values = [f"{result['avg_metrics'][m]:.4f}" for m in metrics]
#         print(f"{model_name:<30} " + " ".join(f"{v:>12}" for v in metrics_values))
    
#     # Save detailed results
#     with open('vision_transformer_benchmark_results.json', 'w') as f:
#         json.dump(all_results, f, indent=2)
#     logger.info("Detailed results saved to vision_transformer_benchmark_results.json")

#     # Create visualization graphs
#     VisualizationUtils.create_comparison_graphs(all_results)

# if __name__ == "__main__":
#     main()


from data.data_loader import DataLoader
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
        data_loader = DataLoader()
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
    VisualizationUtils.create_comparison_graphs(all_results)

if __name__ == "__main__":
    main()