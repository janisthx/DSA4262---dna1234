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
1. Open AWS Ubuntu Instance:
````
ssh -i <.pem> user@hostname
````

2. Clone the repository:
````
git clone https://github.com/janisthx/DSA4262---dna1234.git
````

3. Install Pixi on Unix  
* Pixi is a fast, modern package manager for creating reproducible environments. It manages all project dependencies through a single `pixi.toml` file, ensuring consisten setups across systems. 

* To get started, install Pixi on your AWS Ubuntu instance before running any project commands:
````
curl -fsSL https://pixi.sh/install.sh | sh
````

4. Source your shell configuration  
* After installation, reload your shell configuration to activate Pixi:
````
source ~/.bashrc
````

5. Verify the installation  
* Confirm that Pixi was installed successfully:
````
pixi --version
````

## 4. Run Pipeline
Follow these steps to generate m6A modification predictions using the pre-trained Random Forest model:

1. Navigate to the project root directory
* Change to the main folder of the project, where all scripts, data, and configuration files are located.
* This ensures that all file paths used by the pipeline are correctly resolved.  

````
cd ~/DSA4262---dna1234
````


2. Activate the Pixi environment
* Install and activate all project dependencies defined in the `pixi.toml` file:

````
pixi install
````


3. Run pipeline to get predictions
* Execute the main script with your dataset to obtain predictions  
````
pixi run python src/main.py --filename <filename> --verbose
````
* Replace `<filename>` with the name of your dataset file without the extension `.json`
The `--verbose` flag enables detailed logging so that the progress can be monitored.  
* Example:
```` 
pixi run python src/main.py --filename dataset0_test.json --verbose
````


4. Check the output
* After the pipeline completes, the prediction results will be saved in the `output` directory.  
````
ls output
````
Look for your generated CSV file containing m6A modification predictions. These results can now be used for downstream analysis or visualization. 