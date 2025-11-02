import pandas as pd
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# Helper Functions
def load_json_row(filepath):
    # Read one row at a time
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip(): 
                yield json.loads(line)

def extract_first3_from_row(row) -> Tuple[Optional[str], Optional[str], Optional[str], List]:
    # Extract transcript_id, transcript_position and sequence
    for transcript_id, v1 in row.items():
        for position, v2 in v1.items():
            for sequence, v3 in v2.items():
                first3 = [sublist[:9] for sublist in v3]
                return transcript_id, position, sequence, first3
    return None, None, None, []

def remove_outliers(data: List[float]) -> List[float]:
    if not data:
        return data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [x for x in data if lower <= x <= upper]


def aggregate_median(x: List[float]) -> float:
    return float(np.median(x)) if x else float('nan')

def compute_iqr(x: List[float]) -> float:
    if len(x) == 0:
        return float('nan')
    return float(np.percentile(x, 75) - np.percentile(x, 25))

# Main Function
def parse_and_aggregate(filename: str, verbose: bool = False) -> str:
    """
    Parse JSON file and create aggregated features CSV
    
    Args:
        filename: Name of input file without extension (e.g. 'dataset0')
        verbose: Print progress messages if True
    
    Returns:
        Path to output CSV file
    """
    if verbose:
        print("1. Running data_parsing")
    
    # List Directories
    root_dir = Path(__file__).parents[1] # root
    data_dir = root_dir / "data" # root/data
    
    filepath = data_dir / f"{filename}.json" # root/data/dataset0_test.json
    out_filepath = data_dir / f"{filename}_parsed.csv" # root/data/dataset0_test_parsed.csv
    
    # List Columns
    columns = ['transcript_id', 'transcript_position', 'sequence',
               'mean_0', 'std_0', 'dwell_0',
               'mean_1', 'std_1', 'dwell_1',
               'mean_2', 'std_2', 'dwell_2',
               'mean_0_iqr', 'std_0_iqr', 'dwell_0_iqr',
               'mean_1_iqr', 'std_1_iqr', 'dwell_1_iqr',
               'mean_2_iqr', 'std_2_iqr', 'dwell_2_iqr']


    # Create Empty Dataframe with custom columns
    pd.DataFrame(columns=columns).to_csv(out_filepath, index=False)
    
    if verbose:
        print('Parsing Data')

    count = 0
    # Data Parsing
    for row in load_json_row(filepath):
        transcript_id, position, sequence, features = extract_first3_from_row(row)
        if not features:
            continue

        feats = []
        for i in range(9):
            col = [x[i] for x in features if len(x) > i]
            feats.append(col)

        # Extract Median and IQR
        aggs_median = [aggregate_median(f) for f in feats]
        aggs_iqr = [compute_iqr(f) for f in feats]

        row_dict = {
            'transcript_id': transcript_id,
            'transcript_position': position,
            'sequence': sequence,
            # Median
            'mean_0': aggs_median[0],
            'std_0': aggs_median[1],
            'dwell_0': aggs_median[2],
            'mean_1': aggs_median[3],
            'std_1': aggs_median[4],
            'dwell_1': aggs_median[5],
            'mean_2': aggs_median[6],
            'std_2': aggs_median[7],
            'dwell_2': aggs_median[8],
            
            # IQR
            'mean_0_iqr': aggs_iqr[0],
            'std_0_iqr': aggs_iqr[1],
            'dwell_0_iqr': aggs_iqr[2],
            'mean_1_iqr': aggs_iqr[3],
            'std_1_iqr': aggs_iqr[4],
            'dwell_1_iqr': aggs_iqr[5],
            'mean_2_iqr': aggs_iqr[6],
            'std_2_iqr': aggs_iqr[7],
            'dwell_2_iqr': aggs_iqr[8]
        }
        
        # Output row to CSV
        pd.DataFrame([row_dict]).to_csv(out_filepath, mode='a', header=False, index=False)
        count += 1
        
    if verbose:
        print(f'Parsed and Summarized {filename}!')
        
    return str(out_filepath)

def _cli():
    # Allow User to Specify Options
    parser = argparse.ArgumentParser(description='Parse .json and summarize data')
    parser.add_argument("--filename", type=str, required=True,
                       help='Specify Filename without extension (e.g. dataset0)')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = parse_and_aggregate(args.filename, args.verbose)
    print(result)

if __name__ == '__main__':
    _cli()