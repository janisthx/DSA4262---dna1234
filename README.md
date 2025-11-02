# DSA4262 - Identification of RNA Modifications from direct RNA-Seq data
By: Group DNA1234

## Table of Contents
* [Project Description](#project-description)
* [Directory Structure](#directory-structure)
* [Installation Guide](#installation-guide)
* [Run Pipeline](#run-pipeline)



## 1. Project Description

## 2. Directory Structure

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

2. Activate pixi
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
ls data
````