# %%
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# %%
def train_rf_model(filename, verbose): 
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

    filepath = data_dir / f"{filename}_preprocessed.csv" # root/data/dataset0_preprocessed.csv
    out_filepath = resources_dir / f"{filename}_trained_model.joblib" # root/resources/dataset0_trained_model.joblib


    # Load preprocessed data
    df = pd.read_csv(filepath)
    if verbose:
        print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    # Train Test Split
    # 1. Create the Combined Column to stratify on gene_id + label
    df['combined'] = df['gene_id'].astype(str) + '_' + df['label'].astype(str)

    # 2. Count occurrences of the combined column
    counts = df['combined'].value_counts()

    # 3. If count == 1, replace with just column2 ('label')
    df['final_group'] = df.apply(
        lambda row: row['label'] if counts[row['combined']] == 1 else row['combined'],
        axis=1
    )
    df['final_group'] = df['final_group'].astype(str)

    # Split into X and y
    X = df.drop(columns=['gene_id', 'combined', 'label'])
    y = df['label']

    # Split into 95% train and 5% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df['final_group'], test_size=0.05, random_state=42)
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
        "--filename",
        type=str,
        required=True,
        help="Filename of the preprocessed dataset (without _preprocessed.csv suffix)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages",
    )
    args = parser.parse_args()
    result = train_rf_model(args.filename, args.verbose)
    print(result)

if __name__ == "__main__":
    _cli()
