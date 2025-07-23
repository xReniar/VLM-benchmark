from vlm import VLM
import argparse
import yaml


def main():
    pass


if __name__ == "__main__":
    benchmark_config = yaml.safe_load(open("./configs/benchmark.yaml", "r"))
    test_config = benchmark_config["test"]
    tasks_config = benchmark_config["tasks"]

    TASK: str = benchmark_config["task"]
    DATASET: str = benchmark_config["dataset"]
    MODELS: list[str] = benchmark_config["models"]

    task_config: dict = tasks_config[TASK]