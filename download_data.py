from datasets import load_dataset
import argparse


ds = load_dataset("nanonets/key_information_extraction", split="test")

for x in ds:
    print(x.keys())

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

if __name__ == "__main__":
    pass