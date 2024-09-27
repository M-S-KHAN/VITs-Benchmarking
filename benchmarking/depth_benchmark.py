from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import cv2
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from utils.utils import logger


class DepthBenchmark:
    def __init__(
        self,
        model_type,
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.initialize_model()

    def initialize_model(self):
        logger.info(
            f"Initializing {self.model_type} model: {self.model_name} on device: {self.device}"
        )
        try:
            self.feature_extractor = DPTFeatureExtractor.from_pretrained(
                self.model_name
            )
            self.model = DPTForDepthEstimation.from_pretrained(self.model_name).to(
                self.device
            )
            self.model.eval()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def estimate_depth(self, image_path):
        logger.info(f"Estimating depth for image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = (
                torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            return prediction
        except Exception as e:
            logger.error(f"Error estimating depth: {e}")
            raise

    def calculate_metrics(self, depth_map, bboxes, rgb_image):
        obj_size_consistency = self.object_size_consistency(depth_map, bboxes)
        depth_consistency = np.std(depth_map)
        edge_comparison = self.edge_detection_comparison(rgb_image, depth_map)
        return {
            "obj_size_consistency": obj_size_consistency,
            "depth_consistency": depth_consistency,
            "edge_comparison": edge_comparison,
        }

    def object_size_consistency(self, depth_map, bboxes):
        consistencies = []
        for bbox in bboxes:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            obj_depth = depth_map[int(y1) : int(y2), int(x1) : int(x2)].mean()
            obj_size = w * h
            consistencies.append((obj_depth, obj_size))
        if len(consistencies) > 1:
            depths, sizes = zip(*consistencies)
            depth_size_ratio = np.array(sizes) / np.array(depths)
            return np.std(depth_size_ratio) / np.mean(depth_size_ratio)
        return 0

    def edge_detection_comparison(self, rgb_image, depth_map):
        rgb_edges = cv2.Canny(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), 100, 200)
        depth_edges = cv2.Canny(
            cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            100,
            200,
        )
        intersection = np.logical_and(rgb_edges, depth_edges)
        union = np.logical_or(rgb_edges, depth_edges)
        return np.sum(intersection) / np.sum(union)

    def benchmark(self, benchmark_data):
        results = []
        for item in tqdm(benchmark_data):
            image_path = item["image_path"]
            bboxes = [ann["bbox"] for ann in item["annotations"]]
            try:
                start_time = time.time()
                depth_map = self.estimate_depth(image_path)
                inference_time = time.time() - start_time
                rgb_image = np.array(Image.open(image_path).convert("RGB"))
                metrics = self.calculate_metrics(depth_map, bboxes, rgb_image)
                metrics["inference_time"] = inference_time
                results.append(
                    {
                        "image_path": image_path,
                        "depth_map": depth_map.tolist(),
                        "bboxes": bboxes,
                        "metrics": metrics,
                    }
                )
                logger.info(f"Processed image: {image_path}. Metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        return results


def run_benchmark(model_type, model_name, benchmark_data):
    logger.info(f"Starting benchmark for {model_name}")
    print(f"Starting benchmark for {model_name}")
    benchmark = DepthBenchmark(model_type, model_name)
    results = benchmark.benchmark(benchmark_data)
    avg_metrics = {
        "obj_size_consistency": sum(
            r["metrics"]["obj_size_consistency"] for r in results
        )
        / len(results),
        "depth_consistency": sum(r["metrics"]["depth_consistency"] for r in results)
        / len(results),
        "edge_comparison": sum(r["metrics"]["edge_comparison"] for r in results)
        / len(results),
        "avg_inference_time": sum(r["metrics"]["inference_time"] for r in results)
        / len(results),
    }
    logger.info(f"Average Metrics for {model_name}:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    return {
        "model_name": model_name,
        "avg_metrics": avg_metrics,
        "detailed_results": results,
    }
