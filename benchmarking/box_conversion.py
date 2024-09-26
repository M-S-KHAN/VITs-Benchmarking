import torch

class BoxConversionFactory:
    @staticmethod
    def get_converter(model_type):
        if model_type == "YOLOS":
            return BoxConversionFactory.convert_yolos_format
        if model_type == "DETR":
            return BoxConversionFactory.convert_detr_format
        elif model_type == "OWL-ViT":
            return BoxConversionFactory.convert_owlvit_format
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def convert_yolos_format(boxes, image_size):
        """Convert relative [center_x, center_y, width, height] to absolute [x1, y1, x2, y2]"""
        height, width = image_size
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * width
        y1 = (cy - 0.5 * h) * height
        x2 = (cx + 0.5 * w) * width
        y2 = (cy + 0.5 * h) * height
        return torch.stack((x1, y1, x2, y2), dim=-1)

    @staticmethod
    def convert_owlvit_format(boxes, image_size):
        """Convert normalized [x1, y1, x2, y2] to absolute [x1, y1, x2, y2]"""
        height, width = image_size
        x1, y1, x2, y2 = boxes.unbind(-1)
        x1 = x1 * width
        y1 = y1 * height
        x2 = x2 * width
        y2 = y2 * height
        return torch.stack((x1, y1, x2, y2), dim=-1)

    @staticmethod
    def convert_detr_format(boxes, image_size):
        """Convert normalized [center_x, center_y, width, height] to absolute [x1, y1, x2, y2]"""
        height, width = image_size
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * width
        y1 = (cy - 0.5 * h) * height
        x2 = (cx + 0.5 * w) * width
        y2 = (cy + 0.5 * h) * height
        return torch.stack((x1, y1, x2, y2), dim=-1)

    @staticmethod
    def convert_gt_format(boxes, image_size):
        """Convert absolute [x, y, width, height] to [x1, y1, x2, y2]"""
        x, y, w, h = boxes.unbind(-1)
        x2 = x + w
        y2 = y + h
        return torch.stack((x, y, x2, y2), dim=-1)