import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from typing import Optional
from utils import custom_combiner


def preprocess_parsed(filename: str, verbose: bool = False) -> str:
    """
    Preprocess parsed data for model input
    
    Args:
        filename: Name of input file without extension (e.g. 'dataset0')
        verbose: Print progress messages if True
    
    Returns:
        Path to preprocessed CSV file
    """

    if verbose:
        print('2. Running data_preprocessing')
        
    # List Directories
    root_dir = Path(__file__).parents[1] # root
    data_dir = root_dir / "data" # root/data
    resources_dir = root_dir / "resources" # root/resources
    
    filepath = data_dir / f"{filename}_parsed.csv" # root/data/dataset0_test_parsed.csv
    out_filepath = data_dir / f"{filename}_preprocessed.csv" # root/data/dataset0_test_preprocessed.csv

    if verbose:
        print(f"Loading {filepath}")
    
    # 1. Load Data
    temp_df = pd.read_csv(filepath)
    
    # 2. Create new aggregated features
    temp_df['all_mean'] = temp_df[['mean_0', 'mean_1', 'mean_2']].mean(axis=1)
    temp_df['all_std'] = temp_df[['std_0', 'std_1', 'std_2']].mean(axis=1)
    temp_df['all_dwell'] = temp_df[['dwell_0', 'dwell_1', 'dwell_2']].mean(axis=1)

    # 3. Split sequence into 3 5-mers
    temp_df['seq_5mer_0'] = temp_df['sequence'].str[0:5]
    temp_df['seq_5mer_1'] = temp_df['sequence'].str[1:6]
    temp_df['seq_5mer_2'] = temp_df['sequence'].str[2:7]
    
    # 4. Load encoder and transform sequences
    enc = joblib.load(resources_dir / 'kmer_seq_encoder.joblib')
    seq_cols = ['seq_5mer_0', 
                'seq_5mer_1', 
                'seq_5mer_2']
    
    ohe = enc.transform(temp_df[seq_cols]).toarray()
    encoded_cols = enc.get_feature_names_out(seq_cols)
    ohe_df = pd.DataFrame(ohe, columns=encoded_cols, index=temp_df.index)
    
    # 5. Combine numeric and encoded features
    encoded_df = pd.concat([
        temp_df.drop(columns=['sequence'] + seq_cols),
        ohe_df
    ], axis=1)

    # 6. Log transform selected columns
    cols_to_log = [
        # Median
        'mean_0',
        'std_0',
        'mean_1',
        'std_1',
        'mean_2',
        'std_2',

        # IQR
        'mean_0_iqr',
        'std_0_iqr',
        'mean_1_iqr',
        'std_1_iqr',
        'dwell_1_iqr',
        'mean_2_iqr',
        'std_2_iqr',
        'dwell_2_iqr'
    ]
    
    for col in cols_to_log:
        log_col = "log_" + col

        safe_values = np.clip(encoded_df[col], a_min=1e-9, a_max=None)
        encoded_df[log_col] = np.log(safe_values)

        encoded_df[log_col].drop(columns=col, inplace=True)

    # 7. Reorder Features for Model Prediction
    encoded_df = encoded_df[[
        # Identifiers
        'transcript_id', 
        'transcript_position',

        # Transformed Median Columns
        'log_mean_0', 'log_mean_1', 'log_mean_2',
        'log_std_0', 'log_std_1', 'log_std_2', 
        'dwell_0', 'dwell_1', 'dwell_2',

        # One Hot Encoding
        'seq_5mer_0_AAAAC', 'seq_5mer_0_AAGAC', 'seq_5mer_0_AGAAC', 
        'seq_5mer_0_AGGAC', 'seq_5mer_0_ATAAC', 'seq_5mer_0_ATGAC', 
        'seq_5mer_0_CAAAC', 'seq_5mer_0_CAGAC', 'seq_5mer_0_CGAAC', 
        'seq_5mer_0_CGGAC', 'seq_5mer_0_CTAAC', 'seq_5mer_0_CTGAC', 
        'seq_5mer_0_GAAAC', 'seq_5mer_0_GAGAC', 'seq_5mer_0_GGAAC', 
        'seq_5mer_0_GGGAC', 'seq_5mer_0_GTAAC', 'seq_5mer_0_GTGAC', 
        'seq_5mer_0_TAAAC', 'seq_5mer_0_TAGAC', 'seq_5mer_0_TGAAC', 
        'seq_5mer_0_TGGAC', 'seq_5mer_0_TTAAC', 'seq_5mer_0_TTGAC', 
        'seq_5mer_1_AAACA', 'seq_5mer_1_AAACC', 'seq_5mer_1_AAACT', 
        'seq_5mer_1_AGACA', 'seq_5mer_1_AGACC', 'seq_5mer_1_AGACT', 
        'seq_5mer_1_GAACA', 'seq_5mer_1_GAACC', 'seq_5mer_1_GAACT', 
        'seq_5mer_1_GGACA', 'seq_5mer_1_GGACC', 'seq_5mer_1_GGACT', 
        'seq_5mer_1_TAACA', 'seq_5mer_1_TAACC', 'seq_5mer_1_TAACT', 
        'seq_5mer_1_TGACA', 'seq_5mer_1_TGACC', 'seq_5mer_1_TGACT', 
        'seq_5mer_2_AACAA', 'seq_5mer_2_AACAC', 'seq_5mer_2_AACAG', 
        'seq_5mer_2_AACAT', 'seq_5mer_2_AACCA', 'seq_5mer_2_AACCC', 
        'seq_5mer_2_AACCG', 'seq_5mer_2_AACCT', 'seq_5mer_2_AACTA', 
        'seq_5mer_2_AACTC', 'seq_5mer_2_AACTG', 'seq_5mer_2_AACTT', 
        'seq_5mer_2_GACAA', 'seq_5mer_2_GACAC', 'seq_5mer_2_GACAG', 
        'seq_5mer_2_GACAT', 'seq_5mer_2_GACCA', 'seq_5mer_2_GACCC', 
        'seq_5mer_2_GACCG', 'seq_5mer_2_GACCT', 'seq_5mer_2_GACTA', 
        'seq_5mer_2_GACTC', 'seq_5mer_2_GACTG', 'seq_5mer_2_GACTT', 
        # New Features
        'all_mean', 'all_std', 'all_dwell', 

        # Transformed IQR Columns
        'log_mean_0_iqr', 'log_mean_1_iqr', 'log_mean_2_iqr',
        'log_std_0_iqr', 'log_std_1_iqr', 'log_std_2_iqr',
        'dwell_0_iqr', 'log_dwell_1_iqr', 'log_dwell_2_iqr'
    ]]
    
    if verbose:
        # Ensure that there are 89 columns
        print(f'Number of Rows: {encoded_df.shape[0]}, Number of Columns: {encoded_df.shape[1]}') 
        print(f"Saving preprocessed data to {out_filepath}")
    
    encoded_df.to_csv(out_filepath, index=False)
    return str(out_filepath)

def _cli():
    parser = argparse.ArgumentParser(description='Preprocess parsed data')
    parser.add_argument("--filename", type=str, required=True,
                       help='Specify filename without extension (e.g. dataset0)')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = preprocess_parsed(args.filename, args.verbose)
    print(result)

if __name__ == '__main__':
    _cli()