import pandas as pd
import joblib
import argparse
from pathlib import Path

def predict_m6a(filename: str, verbose: bool = False) -> str:
    """
    Make m6A predictions on preprocessed data
    
    Args:
        filename: Name of input file without extension (e.g. 'dataset0')
        verbose: Print progress messages if True
    
    Returns:
        Path to predictions CSV file
    """
    if verbose:
        print('3. Running predict_m6a')

    # List Directories
    root_dir = Path(__file__).parents[1] # root
    data_dir = root_dir / "data" # root/data
    resources_dir = root_dir / "resources" # root/resources
    output_dir = root_dir / "output" # root/output
    
    filepath = data_dir / f"{filename}_preprocessed.csv" # root/data/dataset0_preprocessed.csv
    out_filepath = output_dir / f"{filename}_predictions.csv" # root/output/dataset0_predictions.csv
    
    if verbose:
        print(f"Loading preprocessed data from {filepath}")
    
    # Load data and separate features
    test_df = pd.read_csv(filepath)
    identifier_cols = ['transcript_id', 'transcript_position']
    X_test_id = test_df[identifier_cols]
    X_test = test_df.drop(columns=identifier_cols)
    
    # Load model and predict
    if verbose:
        print("Loading model and making predictions")
    
    model = joblib.load(resources_dir / 'rf_model.joblib')
    pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Format and save predictions
    predictions_df = pd.DataFrame({
        'transcript_id': X_test_id['transcript_id'],
        'transcript_position': X_test_id['transcript_position'],
        'score': pred_prob
    })
    
    if verbose:
        print(f"Saving predictions to {out_filepath}")
        print("\nFirst 5 predictions:")
        print(predictions_df.head(5))
    
    predictions_df.to_csv(out_filepath, index=False)
    return str(out_filepath)

def _cli():
    parser = argparse.ArgumentParser(description='Predict m6A modifications')
    parser.add_argument("--filename", type=str, required=True,
                       help='Specify filename without extension (e.g. dataset0)')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = predict_m6a(args.filename, args.verbose)
    print(result)

if __name__ == '__main__':
    _cli()