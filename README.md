# MRACO: Multi-Relational Aggregation and Collaborative Optimization Framework for Drug-Drug Interaction Prediction
## Introduction
This repository provides the source code for the MRACO framework, a multi-relational knowledge graph-based model designed to predict potential drug-drug interactions (DDIs). Accurate DDI prediction is crucial for clinical safety and new drug development, as unforeseen interactions between drugs can lead to adverse reactions. MRACO leverages the structural and relational diversity in knowledge graphs, utilizing dual aggregation operations and a collaborative optimization strategy to enhance prediction accuracy and model robustness.
## Key Features
-   **Dual Aggregation**: Aggregates multi-type information to capture diverse semantic relationships of drug nodes in the knowledge graph.
-   **Collaborative Optimization**: Simplifies complex computations by employing a collaborative loss function to guide model training, thus reducing redundancy and enhancing stability.
-   **Efficient Feature Extraction**: Captures high-order neighborhood information for improved prediction stability and performance.
## Requirements
To run the MRACO framework, the following packages are required:
-   Python == 3.7
-   PyTorch == 1.6
-   PyTorch Geometric == 1.6
-   rdkit == 2020.09.1 (for data preprocessing)
**Arguments:**

-   `-s`: Seed for the random number generator. Default=0.
-   `-o`: Preprocessing operation to perform. Choose from `{all, generate_triplets, drug_data, split}`:
    -   `generate_triplets`: Generates triplets with negative samples.
    -   `drug_data`: Transforms drugs into graph representations.
    -   `split`: Stratified splitting of the dataset into 
    -   `-n_f` number of folds.
    -   `all`: Performs all the above operations at once.
-   `-t_r`: Test set ratio [0-1]. Default=0.2.
-   `-n_f`: Number of folds. Default=3.
## Repository Structure
-   `.idea`: Contains project-specific settings for IDE configurations.
-   `ourdata`: Includes datasets used for training and testing MRACO.
-   `plot`: Scripts for generating result plots and visualizations from experimental data.
-   `src`: Source code for auxiliary functions and model components.
-   `HG1`: The main directory for running experiments, contains primary code for implementing MRACO.
-   `model`: Contains the implementation of the MRACO framework, detailing the aggregation and optimization processes.

## Installation

1.  Clone this repository:
    ```
    bash
    git clone https://github.com/yourusername/MRACO.git 
    cd MRACO  
    ```
2.  Install required dependencies:
    ```
    bash
    pip install -r requirements.txt
    ```
## Usage

1.  **Data Preparation**: Place your dataset files in the `ourdata` directory, or use the pre-processed datasets provided.
2.  **Running the Model**: Navigate to the `HG1.py` directory and execute the main script:
    ```
    bash
    python HG1.py
    ```
## Results

MRACO achieves state-of-the-art performance in DDI prediction, leveraging high-order interactions in multi-relational knowledge graphs. Experimental results are presented in the paper, and case studies demonstrate MRACO’s capability for uncovering mechanisms underlying drug interactions.
## Citation

If you find this code useful, please cite our paper:
```
css
@article{YourPaper2025,   author = {Yu Wei, Lei Wang, Chang-Qin Yu, Yang Li, Meng-Meng Wei, Zhu-Hong You},  
title = {Dual Aggregation and Collaborative Optimization Framework for Drug-Drug Interaction Prediction in Multi-Relational Networks}, 
journal = {Journal Name},   year = {2025} }
```
## License

This project is licensed under the MIT License.
