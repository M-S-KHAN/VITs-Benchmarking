# from data_loader import DataLoader
# from utils import setup_logging, save_benchmark_data, load_benchmark_data
# import logging

# def main():
#     setup_logging()
#     logger = logging.getLogger(__name__)

#     # Check if benchmark data already exists
#     existing_data = load_benchmark_data()
#     if existing_data:
#         logger.info("Loaded existing benchmark data.")
#         return existing_data

#     # If not, create new benchmark data
#     data_loader = DataLoader()
#     benchmark_data = data_loader.load_benchmark_data()
    
#     logger.info(f"Total images loaded for benchmarking: {len(benchmark_data)}")
    
#     # Save benchmark data
#     save_benchmark_data(benchmark_data)
#     logger.info("Benchmark data saved to file.")

#     return benchmark_data

# if __name__ == "__main__":
#     benchmark_data = main()
#     print("Sample image paths and annotations:")
#     for i in range(min(5, len(benchmark_data))):
#         item = benchmark_data[i]
#         print(f"Image: {item['image_path']}")
#         print(f"Annotations: {item['annotations'][:2]}...")  # Print first two annotations
#         print()


from data_loader import DataLoader
from model_benchmark import ModelBenchmark
from benchmark_config import BenchmarkConfig
from utils import setup_logging, save_benchmark_data, load_benchmark_data
import logging
import json

def create_dataset(benchmark_data):
    return [(item['image_path'], {'boxes': item['annotations']}) for item in benchmark_data]

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Check if benchmark data already exists
    existing_data = load_benchmark_data()
    if existing_data:
        logger.info("Loaded existing benchmark data.")
        benchmark_data = existing_data
    else:
        # If not, create new benchmark data
        data_loader = DataLoader()
        benchmark_data = data_loader.load_benchmark_data()
        
        logger.info(f"Total images loaded for benchmarking: {len(benchmark_data)}")
        
        # Save benchmark data
        save_benchmark_data(benchmark_data)
        logger.info("Benchmark data saved to file.")

    # Create dataset in the format expected by ModelBenchmark
    dataset = create_dataset(benchmark_data)

    # Initialize ModelBenchmark
    config = BenchmarkConfig()
    model_benchmark = ModelBenchmark(config)

    # Run the benchmark
    all_results = model_benchmark.run_benchmark(dataset, num_runs=5)

    # Analyze dataset impact
    impact_analysis, dataset_info = model_benchmark.analyze_dataset_impact(all_results, dataset)

    # Report results
    model_benchmark.report_results(all_results, impact_analysis, dataset_info)

    # Save benchmark results
    with open('benchmark_results.json', 'w') as f:
        json.dump({
            'all_results': all_results,
            'impact_analysis': impact_analysis,
            'dataset_info': dataset_info
        }, f, indent=2)
    logger.info("Benchmark results saved to benchmark_results.json")

    return all_results, impact_analysis, dataset_info

if __name__ == "__main__":
    all_results, impact_analysis, dataset_info = main()
    
    print("\nSample benchmark results:")
    for model_name, model_results in list(all_results.items())[:2]:  # Print results for first two models
        print(f"\n{model_name}:")
        for variant, metrics in list(model_results.items())[:2]:  # Print results for first two variants
            print(f"  {variant}:")
            for metric, value in metrics["avg_metrics"].items():
                print(f"    {metric}: {value:.4f} ± {metrics['std_metrics'][metric]:.4f}")