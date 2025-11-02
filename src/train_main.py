from pathlib import Path
import argparse

from data_parsing import parse_and_aggregate
from data_preprocessing import preprocess_parsed  
from train_model import train_rf_model

def run_pipeline(filename: str, verbose: bool = False):
    """
    Run model training pipeline:
    1. Parse JSON -> CSV
    2. Preprocess parsed data
    3. Train Model
    
    Args:
        filename: Name of input file (without extension)
        verbose: Print progress messages if True
    """
    if verbose:
        print(f"Starting pipeline for {filename}")
        
    # Step 1: Parse JSON
    parsed_path = parse_and_aggregate(
        filename=filename,
        verbose=verbose
    )
    if verbose:
        print(f"Parsing complete: {parsed_path}")

    # Step 2: Preprocess
    preprocessed_path = preprocess_parsed(
        filename=filename,
        verbose=verbose
    )
    if verbose:
        print(f"Preprocessing complete: {preprocessed_path}")

    # Step 3: Train Model
    model_path = train_rf_model(
        filename=filename,
        verbose=verbose
    )
    if verbose:
        print(f"Predictions saved to: {model_path}")
        
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Run model training pipeline")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Input filename without extension (e.g. 'dataset0')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages"
    )
    args = parser.parse_args()
    
    result = run_pipeline(args.filename, args.verbose)
    print(f"Pipeline complete. Final output: {result}")

if __name__ == "__main__":
    main()