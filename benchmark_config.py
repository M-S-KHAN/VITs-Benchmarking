from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, AutoModelForDepthEstimation

class BenchmarkConfig:
    # Object Detection Models
    OBJECT_DETECTION_MODELS = {
        "YOLOS": {
            "small": "hustvl/yolos-small",
            "base": "hustvl/yolos-base",
            "small-300": "hustvl/yolos-small-300",
        },
        "DETR": {
            "base": "facebook/detr-resnet-50",
            "large": "facebook/detr-resnet-101",
        },
        "Mask2Former": {
            "base": "facebook/mask2former-swin-base-coco-instance",
            "large": "facebook/mask2former-swin-large-coco-instance",
        },
        "YOLOF": {
            "base": "chenhaojie/yolof-r50-c5-1x",
        },
        "ConditionalDETR": {
            "base": "microsoft/conditional-detr-resnet-50",
        }
    }

    # Depth Estimation Models
    DEPTH_ESTIMATION_MODELS = {
        "DPT": {
            "small": "Intel/dpt-small",
            "base": "Intel/dpt-large",
        },
        "GLPDepth": {
            "base": "vinvino02/glpn-kitti",
        },
        "MiDaS": {
            "small": "Intel/midas-small",
            "base": "Intel/midas-base",
        },
        "AdaBins": {
            "base": "shariqfarooq/AdaBins",
        },
        "BTS": {
            "base": "Intel/bts-large",
        }
    }

    # Metrics
    OBJECT_DETECTION_METRICS = [
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "average_precision",
        "average_recall",
        "inference_time"
    ]

    DEPTH_ESTIMATION_METRICS = [
        "RMSE",
        "MAE",
        "SSIM",
        "delta1",
        "delta2",
        "delta3",
        "inference_time"
    ]

    # Inference settings
    BATCH_SIZE = 16
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    # Hardware settings
    DEVICE = "cuda"  # or "cpu" if CUDA is not available

    # Output settings
    RESULTS_DIR = "benchmark_results"
    VISUALIZATIONS_DIR = "visualizations"

    # Visualization settings
    PLOT_TYPES = [
        "bar_chart",
        "radar_chart",
        "scatter_plot",
        "box_plot"
    ]

    COMPARISON_METRICS = [
        "mAP",
        "inference_time",
        "RMSE",
        "MAE"
    ]

    # Color settings for visualizations
    COLORS = {
        "object_detection": "viridis",
        "depth_estimation": "plasma"
    }
    
    PLOT_TYPES = ["bar_chart", "radar_chart", "scatter_plot", "box_plot"]
    COMPARISON_METRICS = ["mAP", "inference_time", "RMSE", "MAE"]
    RESULTS_DIR = "benchmark_results"

    @staticmethod
    def load_model(model_type, model_name, variant):
        if model_type == "object_detection":
            model_path = BenchmarkConfig.OBJECT_DETECTION_MODELS[model_name][variant]
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            model = AutoModelForObjectDetection.from_pretrained(model_path)
        elif model_type == "depth_estimation":
            model_path = BenchmarkConfig.DEPTH_ESTIMATION_MODELS[model_name][variant]
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            model = AutoModelForDepthEstimation.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return feature_extractor, model