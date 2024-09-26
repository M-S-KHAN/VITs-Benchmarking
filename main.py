from data_loader import DataLoader
from utils import setup_logging, save_benchmark_data, load_benchmark_data
import logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Check if benchmark data already exists
    existing_data = load_benchmark_data()
    if existing_data:
        logger.info("Loaded existing benchmark data.")
        return existing_data

    # If not, create new benchmark data
    data_loader = DataLoader()
    benchmark_data = data_loader.load_benchmark_data()
    
    logger.info(f"Total images loaded for benchmarking: {len(benchmark_data)}")
    
    # Save benchmark data
    save_benchmark_data(benchmark_data)
    logger.info("Benchmark data saved to file.")

    return benchmark_data

if __name__ == "__main__":
    benchmark_data = main()
    print("Sample image paths and annotations:")
    for i in range(min(5, len(benchmark_data))):
        item = benchmark_data[i]
        print(f"Image: {item['image_path']}")
        print(f"Annotations: {item['annotations'][:2]}...")  # Print first two annotations
        print()