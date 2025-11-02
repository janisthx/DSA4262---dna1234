import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def train_rf_model(data_filename: str, label_filename: str, verbose: bool = False) -> str:
    """
    Train a Random Forest Classifier on the provided dataset and save the trained model.

    Args:
        filename (str): Path to the CSV file containing the preprocessed dataset.
        verbose (bool): Print progress messages if True.
    
    Returns:
        Path to trained model object
    """
    if verbose:
        print("Running model_training.py")

    # List Directories
    root_dir = Path(__file__).parents[1] # root
    data_dir = root_dir / "data" # root/data
    resources_dir = root_dir / "resources" # root/resources

    filepath = data_dir / f"{data_filename}_preprocessed.csv" # root/data/dataset0_preprocessed.csv
    labelpath = data_dir / f"{label_filename}" # root/data/data.info.labelled
    out_filepath = resources_dir / f"{data_filename}_trained_model.joblib" # root/resources/dataset0_trained_model.joblib

    if verbose:
        print(f"Data Filepath: {filepath}")
        print(f"Label Filepath: {labelpath}")
        print(f"Output Model Filepath: {out_filepath}")

    # Load datasets
    df = pd.read_csv(filepath)
    label_df = pd.read_csv(labelpath)

    if verbose:
        print('Preprocessed Data Loaded')
        print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    # Merge features and labels on transcript_id and transcript_position
    merged_df = pd.merge(df, label_df, on=['transcript_id', 'transcript_position'], how='left') # will add on gene_id and label columns

    # Train Test Split
    # 1. Create the Combined Column to stratify on gene_id + label
    merged_df['combined'] = merged_df['gene_id'].astype(str) + '_' + merged_df['label'].astype(str)

    # 2. Count occurrences of the combined column
    counts = merged_df['combined'].value_counts()

    # 3. If count == 1, replace with just column2 ('label')
    merged_df['final_group'] = merged_df.apply(
        lambda row: row['label'] if counts[row['combined']] == 1 else row['combined'],
        axis=1
    )
    merged_df['final_group'] = merged_df['final_group'].astype(str)

    # Split into X and y
    X = merged_df.drop(columns=['gene_id', 'combined', 'label'])
    y = merged_df['label']

    # Split into 95% train and 5% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=merged_df['final_group'], test_size=0.05, random_state=42)
    X_train.drop(columns=['transcript_position', 'transcript_id', 'final_group'], inplace=True)
    X_test.drop(columns=['transcript_position', 'transcript_id', 'final_group'], inplace=True)

    if verbose:
        print("Train and Test Split Completed")
        print(f'X_train: {X_train.head(3)}')
        print(f'X_train Dimensions: {X_train.shape}')

        print(f'X_test: {X_test.head(3)}')
        print(f'X_test Dimensions: {X_test.shape}')


    # Use StratifiedKFold for Cross-Validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Instantiate Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Check Cross Validation Scores
    verbose_val = 0
    if verbose:
        verbose_val = 2

    train_cv_scores = cross_val_score(rf_model, X_train, y_train, 
                                      cv=cv, 
                                      scoring='average_precision', 
                                      n_jobs=-1, 
                                      verbose=verbose_val) 
    if verbose:
        print(f"Mean Cross-validation Average Precision Score on Training Set: {np.mean(train_cv_scores)}")
    
    # Fit the model on the 95% training data
    rf_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(rf_model, out_filepath)

    if verbose:
        print(f"Trained model saved to {out_filepath}")

    return str(out_filepath)

def _cli():
    parser = argparse.ArgumentParser(description="Train Random Forest Classifier")
    parser.add_argument(
        "--data_filename",
        type=str,
        required=True,
        help="Filename of the preprocessed dataset (without _preprocessed.csv suffix)",
    )
    parser.add_argument(
        "--label_filename",
        type=str,
        required=True,
        help="Filename of the labels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages",
    )
    args = parser.parse_args()
    result = train_rf_model(args.data_filename, args.label_filename, args.verbose)
    print(result)

if __name__ == "__main__":
    _cli()
