DET_MODELS_TO_BENCHMARK = [
    ("OWL-ViT", "google/owlvit-base-patch32"),
    ("DETR", "facebook/detr-resnet-50"),
    # ("DETR", "facebook/detr-resnet-101"),
    ("YOLOS", "hustvl/yolos-small"),
    # ("YOLOS", "hustvl/yolos-base"),
]

DEPTH_MODELS_TO_BENCHMARK = [
    ("DPT-Hybrid", "Intel/dpt-hybrid-midas"),
    ("DPT-Large", "Intel/dpt-large"),
    ("DPT-BEiT", "Intel/dpt-beit-base-384"),
]