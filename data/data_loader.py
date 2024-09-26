import json

class DataLoader:
    def load_benchmark_data(self, data_path="benchmark_data.json"):
        """
        Load benchmark data from a JSON file.

        Args:
            data_path (str): The path to the JSON file containing the benchmark data.

        Returns:
            list: A list of benchmark data items.
        """
        with open(data_path, 'r') as f:
            benchmark_data = json.load(f)
        return benchmark_data