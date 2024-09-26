import os
import requests
from tqdm import tqdm
import zipfile
import tarfile
import json
import logging
from config import Config

def setup_logging():
    logging.basicConfig(filename=Config.LOG_FILE, level=logging.INFO, format=Config.LOG_FORMAT)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def extract_file(filename, extract_dir):
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif filename.endswith('.tar'):
        with tarfile.open(filename, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)

def save_benchmark_data(benchmark_data):
    with open(Config.BENCHMARK_DATA_FILE, 'w') as f:
        json.dump(benchmark_data, f)

def load_benchmark_data():
    if os.path.exists(Config.BENCHMARK_DATA_FILE):
        with open(Config.BENCHMARK_DATA_FILE, 'r') as f:
            return json.load(f)
    return None