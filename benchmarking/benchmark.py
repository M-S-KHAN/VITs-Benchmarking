"""
benchmark.py

This module contains the Benchmark class and related functions for benchmarking object detection models.
"""

import torch
from torchvision.ops import box_iou
from PIL import Image
import time
from tqdm import tqdm
from benchmarking.box_conversion import BoxConversionFactory
from utils.utils import logger
from transformers import YolosForObjectDetection, YolosFeatureExtractor, DetrForObjectDetection, DetrFeatureExtractor, OwlViTForObjectDetection, OwlViTProcessor

class Benchmark:
    def __init__(self, model_type, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.converter = BoxConversionFactory.get_converter(model_type)
        self.initialize_model()

    def initialize_model(self):
        logger.info(f"Initializing {self.model_type} model: {self.model_name} on device: {self.device}")
        try:
            if self.model_type == "YOLOS":
                self.feature_extractor = YolosFeatureExtractor.from_pretrained(self.model_name)
                self.model = YolosForObjectDetection.from_pretrained(self.model_name).to(self.device)
            elif self.model_type == "DETR":
                self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_name)
                self.model = DetrForObjectDetection.from_pretrained(self.model_name).to(self.device)
            elif self.model_type == "OWL-ViT":
                self.feature_extractor = OwlViTProcessor.from_pretrained(self.model_name)
                self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model.eval()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def preprocess_image(self, image_path):
      logger.info(f"Preprocessing image: {image_path}")
      try:
          image = Image.open(image_path).convert("RGB")
          if self.model_type == "OWL-ViT":
              # Use multiple generic queries
              text_queries = ["object", "thing", "item", "person", "animal", "vehicle", "furniture"]
              return self.feature_extractor(text=text_queries, images=image, return_tensors="pt")
          else:
              return self.feature_extractor(images=image, return_tensors="pt")
      except Exception as e:
          logger.error(f"Error preprocessing image: {e}")
          raise

    def postprocess_predictions(self, outputs, confidence_threshold=0.5):
      logger.info("Postprocessing predictions")
      try:
          if self.model_type in ["YOLOS", "DETR"]:
              probas = outputs.logits.softmax(-1)[0, :, :-1]
              keep = probas.max(-1).values > confidence_threshold
              
              boxes = outputs.pred_boxes[0, keep]
              labels = probas[keep].argmax(-1)
              scores = probas[keep].max(-1).values
          elif self.model_type == "OWL-ViT":
              # Lower confidence threshold for OWL-ViT
              owl_vit_threshold = 0.1
              target_sizes = torch.tensor([1.0, 1.0]).unsqueeze(0)  # Assuming normalized coordinates
              results = self.feature_extractor.post_process_object_detection(outputs, threshold=owl_vit_threshold, target_sizes=target_sizes)[0]
              boxes = results["boxes"]
              labels = results["labels"]
              scores = results["scores"]
              
              # Print detailed information about OWL-ViT outputs
              logger.info(f"OWL-ViT raw output: Boxes: {boxes.shape}, Labels: {labels.shape}, Scores: {scores.shape}")
              logger.info(f"OWL-ViT scores: {scores}")
              logger.info(f"OWL-ViT labels: {labels}")
          else:
              raise ValueError(f"Unsupported model type: {self.model_type}")
          
          logger.info(f"Postprocessing complete. Boxes: {boxes.shape}, Labels: {labels.shape}, Scores: {scores.shape}")
          return boxes.cpu(), labels.cpu(), scores.cpu()
      except Exception as e:
          logger.error(f"Error postprocessing predictions: {e}")
          raise

    def run_inference(self, image):
        logger.info("Running inference")
        try:
            inputs = {k: v.to(self.device) for k, v in image.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logger.info("Inference completed successfully")
            return outputs
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def calculate_metrics(self, pred_boxes, gt_boxes, image_size):
        logger.info(f"Calculating metrics. Pred boxes: {pred_boxes.shape}, GT boxes: {gt_boxes.shape}")
        if pred_boxes.nelement() == 0 or gt_boxes.nelement() == 0:
            logger.warning(f"Empty prediction or ground truth. pred_boxes: {pred_boxes.shape}, gt_boxes: {gt_boxes.shape}")
            return {"mean_iou": 0, "ap_50": 0, "ap_75": 0}

        pred_boxes = self.converter(pred_boxes, image_size)
        gt_boxes = BoxConversionFactory.convert_gt_format(gt_boxes, image_size)

        try:
            iou = box_iou(pred_boxes, gt_boxes)
            metrics = {
                "mean_iou": iou.mean().item(),
                "ap_50": (iou.max(dim=1)[0] > 0.5).float().mean().item(),
                "ap_75": (iou.max(dim=1)[0] > 0.75).float().mean().item(),
            }
            logger.info(f"Metrics calculated: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logger.error(f"pred_boxes: {pred_boxes}")
            logger.error(f"gt_boxes: {gt_boxes}")
            return {"mean_iou": 0, "ap_50": 0, "ap_75": 0}

    def benchmark(self, benchmark_data):
        results = []
        for item in tqdm(benchmark_data):
            logger.info(f"Processing image: {item['image_path']}")
            image_path = item['image_path']
            annotations = item['annotations']
            
            try:
                start_time = time.time()
                
                image = self.preprocess_image(image_path)
                outputs = self.run_inference(image)
                pred_boxes, pred_labels, pred_scores = self.postprocess_predictions(outputs)
                
                inference_time = time.time() - start_time
                
                with Image.open(image_path) as img:
                    image_size = img.size[::-1]  # (height, width)
                
                gt_boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
                metrics = self.calculate_metrics(pred_boxes, gt_boxes, image_size)
                metrics["inference_time"] = inference_time
                
                results.append({
                    "image_path": image_path,
                    "pred_boxes": pred_boxes.tolist(),
                    "pred_labels": pred_labels.tolist(),
                    "pred_scores": pred_scores.tolist(),
                    "gt_boxes": gt_boxes.tolist(),
                    "gt_categories": [ann['category'] for ann in annotations],
                    "metrics": metrics
                })
                logger.info(f"Processed image: {image_path}. Metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        
        return results
    
def run_benchmark(model_type, model_name, benchmark_data):
    logger.info(f"Starting benchmark for {model_name}")
    print(f"Starting benchmark for {model_name}")
    benchmark = Benchmark(model_type, model_name)
    results = benchmark.benchmark(benchmark_data)
    
    avg_metrics = {
        "mean_iou": sum(r["metrics"]["mean_iou"] for r in results) / len(results),
        "ap_50": sum(r["metrics"]["ap_50"] for r in results) / len(results),
        "ap_75": sum(r["metrics"]["ap_75"] for r in results) / len(results),
        "avg_inference_time": sum(r["metrics"]["inference_time"] for r in results) / len(results)
    }
    
    logger.info(f"Average Metrics for {model_name}:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return {
        "model_name": model_name,
        "avg_metrics": avg_metrics,
        "detailed_results": results
    }