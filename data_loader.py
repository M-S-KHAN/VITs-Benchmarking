import os
import random
import xml.etree.ElementTree as ET
import logging
from config import Config
from utils import download_file, extract_file
import json

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_random_images(self, dataset_name, num_images):
        dataset_dir = os.path.join(Config.DATASET_DIR, dataset_name)
        image_files = []
        
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        return random.sample(image_files, min(num_images, len(image_files)))

    def load_coco_annotations(self, image_path):
        """Load COCO annotations for a given image."""
        ann_file = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'annotations', 'instances_val2017.json')
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        image_id = int(os.path.basename(image_path).split('.')[0])
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        unified_annotations = []
        for ann in annotations:
            unified_annotations.append({
                'category': next(cat['name'] for cat in coco_data['categories'] if cat['id'] == ann['category_id']),
                'bbox': ann['bbox'],  # [x, y, width, height]
            })
        
        return unified_annotations

    def load_pascal_voc_annotations(self, image_path):
        """Load Pascal VOC annotations for a given image."""
        ann_file = image_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
        if not os.path.exists(ann_file):
            print(f'Annotation file not found: {ann_file}')
            return []  # Return empty if not found
        
        try:
            with open(ann_file, 'r') as f:
                content = f.read().strip()
            if not content:
                print(f'Empty annotation file: {ann_file}')
                return []
            
            tree = ET.fromstring(content)
            unified_annotations = []
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    unified_annotations.append({
                        'category': obj.find('name').text if obj.find('name') is not None else 'unknown',
                        'bbox': [
                            float(bbox.find('xmin').text) if bbox.find('xmin') is not None else 0,
                            float(bbox.find('ymin').text) if bbox.find('ymin') is not None else 0,
                            float(bbox.find('xmax').text) - float(bbox.find('xmin').text) if bbox.find('xmax') is not None and bbox.find('xmin') is not None else 0,
                            float(bbox.find('ymax').text) - float(bbox.find('ymin').text) if bbox.find('ymax') is not None and bbox.find('ymin') is not None else 0
                        ],
                    })
            return unified_annotations
        except ET.ParseError as e:
            print(f'XML Parse Error: {e} for file: {ann_file}')
            with open(ann_file, 'r') as f:
                print(f'File content: {f.read()}')
            return []  # To handle parse errors
        except Exception as e:
            print(f'Unexpected error: {e} for file: {ann_file}')
            return []

    def load_kitti_annotations(self, image_path):
        """Load KITTI annotations for a given image."""
        # Try different possible locations for the label file
        possible_label_paths = [
            image_path.replace('image_2', 'label_2').replace('.png', '.txt'),
            image_path.replace('testing/image_2', 'training/label_2').replace('.png', '.txt'),
            os.path.join(os.path.dirname(os.path.dirname(image_path)), 'label_2', os.path.basename(image_path).replace('.png', '.txt'))
        ]
        
        ann_file = None
        for path in possible_label_paths:
            if os.path.exists(path):
                ann_file = path
                break
        
        if ann_file is None:
            print(f'Annotation file not found for image: {image_path}')
            print(f'Tried paths: {possible_label_paths}')
            return []  # Return empty annotations list if file not found
        
        unified_annotations = []
        try:
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 15:  # KITTI format should have at least 15 values
                        print(f'Invalid KITTI annotation line: {line}')
                        continue
                    unified_annotations.append({
                        'category': parts[0],
                        'bbox': [float(parts[4]), float(parts[5]), float(parts[6]) - float(parts[4]), float(parts[7]) - float(parts[5])],
                    })
        except Exception as e:
            print(f'Error reading KITTI annotation file {ann_file}: {str(e)}')
        
        return unified_annotations

    def load_benchmark_data(self):
        if not os.path.exists(Config.DATASET_DIR):
            os.makedirs(Config.DATASET_DIR)
        
        benchmark_images = []
        
        for dataset_name, urls in Config.DATASETS.items():
            self.logger.info(f"Processing {dataset_name} dataset...")
            dataset_dir = os.path.join(Config.DATASET_DIR, dataset_name)
            
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
                for key, url in urls.items():
                    filename = os.path.join(dataset_dir, f"{key}_{url.split('/')[-1]}")
                    download_file(url, filename)
                    extract_file(filename, dataset_dir)
            
            images = self.get_random_images(dataset_name, Config.IMAGES_PER_DATASET)
            for image_path in images:
                if dataset_name == "COCO":
                    annotations = self.load_coco_annotations(image_path)
                elif dataset_name == "Pascal_VOC":
                    annotations = self.load_pascal_voc_annotations(image_path)
                elif dataset_name == "KITTI":
                    annotations = self.load_kitti_annotations(image_path)
                
                benchmark_images.append({"image_path": image_path, "annotations": annotations})
        
        return benchmark_images[:Config.TOTAL_IMAGES]