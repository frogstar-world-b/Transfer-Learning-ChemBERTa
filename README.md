# Transfer Learning vs. Fine-Tuning ChemBERTa for Regression

## Summary

This repository demonstrates the use of fine-tuning vs transfer learning for a regression task with [ChemBERTa](https://arxiv.org/abs/2010.09885), a specialized BERT-like model applied to chemical SMILES data. SMILES (Simplified Molecular Input Line Entry System) is a notation for representing chemical structures as text. We explore when transfer learning might be more appropriate than fine-tuning ChemBERTa given our dataset, which is significantly smaller than the model's pre-training data (a few hundred vs 77 millions examples).

The regression task is to predict the pIC50 values for inhibiting the catalytic activity of Dihydrofolate Reductase ([DHFR](https://en.wikipedia.org/wiki/Dihydrofolate_reductase)) in homo sapiens. DHFR is a crucial enzyme in the folate metabolic pathway, and inhibiting its catalytic activity can disrupt the production of tetrahydrofolate, which is necessary for DNA synthesis. This disruption can slow down or prevent cancer cell replication, making DHFR an important target for cancer treatment.

pIC50 is a measure of a substance's potency, representing the negative logarithm (base 10) of its Inhibitory Concentration at 50% (IC50). 


## Dataset

Downloaded from https://github.com/KISysBio/qsar-models/tree/master, the dataset consists of SMILES representations, molecular descriptors, and corresponding pIC50 values. 


## Requirements

Before running the notebooks, ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- Transformers library (Hugging Face)
- XGBoost
- NumPy
- pandas
- scikit-learn
- scipy
- matplotlib
- tqdm
- RDKit

You can install these packages using Conda and pip:

Using Conda (recommended for RDKit):

```bash
conda create -n myenv python=3.7  # Create a new Conda environment (optional)
conda activate myenv             # Activate the Conda environment (if created)
conda install -c conda-forge rdkit
pip install torch transformers xgboost numpy pandas scikit-learn scipy matplotlib tqdm
```

This will ensure proper installation of RDKit through Conda, which is a common practice in cheminformatics. Make sure to create and activate a Conda environment as needed.


## Notebooks

### 1. Data Preprocessing and Light EDA

- Notebook: `preprocessing.ipynb`
- This notebook covers data preprocessing and exploratory data analysis (EDA).

### 2. Fine-Tuning ChemBERTa

- Notebook: `fine-tune.ipynb`
- In this notebook, ChemBERTa is fine-tuned using the Transformers library. The fine-tuned model is trained on the SMILES representations of molecules to predict pIC50 values.

### 3. Transfer Learning with ChemBERTa Embeddings using XGBoost

- Notebook: `transfer-learning.ipynb`
- This notebook demonstrates transfer learning using ChemBERTa embeddings of SMILES representations and XGBoost regressor. It explores whether pre-trained ChemBERTa embeddings enhance predictive performance compared to fine-tuning.

### 4. Transfer Learning with ChemBERTa Embeddings and Molecular Descriptors

- Notebook: `transfer-learning-plus-descriptors.ipynb`
- This notebook extends transfer learning by incorporating molecular descriptors alongside ChemBERTa embeddings. It evaluates the impact of additional molecular features on prediction accuracy.

## Results

The notebooks provide insights into the performance of different approaches. Metrics such as Mean Squared Error (MSE) and Spearman Correlation are used to evaluate model performance.

Given the dataset is small, fine-tuning was not sufficient to update the pre-trained weights significantly, and we ended up with nearly identical predictions for all the observations in the test set. 

Transfer-learning improved prediction significantly, which was improved even more with the addition of molecular descriptors as predictors.

