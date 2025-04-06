from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics


def average_of_averages(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("Metrics received at server:", metrics)  # Debugging
    
    if not metrics:
        return {}
    
    # Extract the number of data points and initialize accumulators
    data_points = [num_data_points for num_data_points, _ in metrics]
    num_columns = len(next(iter(metrics))[1])  # Determine the number of columns from the first metric
    
    # Initialize accumulators for column sums and total data points
    column_sums = [0.0] * num_columns
    total_count = 0
    
    # Accumulate weighted sums for each column
    for (num_data_points, metric) in metrics:
        total_count += num_data_points
        for i in range(num_columns):
            key = f"column_{i}_average"
            if key in metric:
                column_sums[i] += metric[key] * num_data_points
    
    # Calculate the weighted average for each column
    if total_count == 0:
        return {f"column_{i}_average": 0.0 for i in range(num_columns)}
    
    column_averages = [sum_ / total_count for sum_ in column_sums]
    
    return {f"column_{i}_average": avg for i, avg in enumerate(column_averages)}




# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=average_of_averages,
                  min_available_clients=3)


# Define config
config = ServerConfig(num_rounds=1)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:5007",
        config=config,
        strategy=strategy,
    )
