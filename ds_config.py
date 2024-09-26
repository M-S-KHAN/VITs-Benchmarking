class Config:
    DATASETS = {
        "COCO": {
            "images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "Pascal_VOC": {
            "all": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        },
        "KITTI": {
            "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
            "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        },
    }

    DATASET_DIR = "datasets"
    BENCHMARK_DATA_FILE = "benchmark_data.json"
    IMAGES_PER_DATASET = 333
    TOTAL_IMAGES = 1000

    LOG_FILE = "data_loading.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
