# DSA4262 - Identification of RNA Modifications from direct RNA-Seq data
By: Group DNA1234

## Table of Contents
* [Project Description](#project-description)
* [Directory Structure](#directory-structure)
* [Installation Guide](#installation-guide)
* [Run Pipeline](#run-pipeline)


## 1. Project Overview
RNA modifications are chemical changes to RNA bases that can influence gene expression and function. Among these, m6A is one of the most prevalent modifications found in mRNA. This project aims to detect m6A sites by developing a machine learning classifier trained on direct RNA-Seq data.  

Through feature engineering, model experimentation and hyperparameter tuning, a **Random Forest** model was selected as the best performer, optimized specifically for PR-AUC to address the class imbalance in the dataset.

To predict m6A modification, the pipeline takes in the raw `.json` dataset, parse and summarizes the data, performs preprocessing, and generates predictions using the **pre-trained Random Forest**. For more details, please refer to [Section 4: Run Pipeline](#4-run-pipeline).

## 2. Directory Structure
```
DSA4262---dna1234
├── data/
|   ├── model_training
|       └── dataset0.json
|   ├── dataset0_test.json
|
├── src/
|   └── data_parsing.py
|   └── data_preprocessing.py
|   └── predict_m6a.py
|   └── main.py
|   └── utils.py
|   └── __init__.py
|   └── model_training.py
|
├── resources/
|   └── kmer_seq_encoder.joblib
|   └── rf_model.joblib
|
├── output/
|
├── notebooks/
|
├── task2/
```



## 3. Installation Guide
1. Open AWS Ubuntu Instance
````
ssh -i <.pem> user@hostname
````

2. Clone the repository
````
git clone https://github.com/janisthx/DSA4262---dna1234.git
````

3. Install Pixi on Unix
````
curl -fsSL https://pixi.sh/install.sh | sh
````

4. Source your shell configuration
````
source ~/.bashrc
````

5. Verify the installation
````
pixi --version
````

## 4. Run Pipeline
1. Change to root directory of the project
````
cd ~/DSA4262---dna1234
````

2. Activate pixi environment
````
pixi install
````

3. Run pipeline to get predictions
````
pixi run python src/main.py --filename <filename> --verbose
````
* Example:
```` 
pixi run python src/main.py --filename dataset0_test.json --verbose
````

4. Check for prediction output
````
ls output
````