Credit Card Fraud Detection 
Author: DHWAJ JAIN & MAANYA RAICHURA
Group Number: #5 
Course: DTSC301 – Machine Learning for Data Science I


1. Project Overview

This project develops and evaluates a comprehensive machine learning pipeline for credit card fraud detection under severe class imbalance. The objective is not only to maximise predictive performance but also to understand model behaviour under realistic banking constraints, where false positives and false negatives carry asymmetric costs.

The pipeline includes:
Exploratory Data Analysis (EDA) before and after preprocessing
Feature engineering from temporal, monetary, and categorical attributes
Comparative modelling using multiple supervised learning algorithms
Hyperparameter tuning with randomized search
Evaluation using imbalance-aware metrics (ROC-AUC, PR-AUC, F1-score ...)
Error analysis and business-aligned interpretation

The emphasis is placed on interpretability, robustness, and operational feasibility, rather than accuracy alone.

2. Repository Structure
.
├── fraudTrain1.csv          # Training dataset
├── fraudTest.csv            # Test dataset
├── main.py                  # Main executable pipeline
└── README.txt               # Plain-text version of documentation

3. Environment Setup
3.1 Operating System
Windows 10 / 11
macOS (Intel or Apple Silicon)
Linux (Ubuntu 20.04 or later)

3.2 Python Version

Python 3.10+

Tested on Python 3.13.2 & 3.11.9 

To verify your Python version, run:

	python --version

If Python is not installed, download it from:
https://www.python.org/downloads/

Ensure that Python is added to PATH during installation (Windows users).

3.3 Recommended IDE / Code Editor
Any of the following can be used:
VS Code (Recommended) Download: https://code.visualstudio.com/

VS Code Extensions (Recommended)
Python (Microsoft)
Pylance
Jupyter (for inline plotting)


3.4 Virtual Environment (Recommended)

Create Virtual Environment, from the project directory:
	
	python -m venv venv

Activate the environment:

Windows: 

	venv\Scripts\activate


macOS / Linux: 

	source venv/bin/activate

After activation, your terminal should display:

	(venv)

3.3 Required Packages

Install dependencies using:

	pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost scipy shap lime lightgbm

To verify installation:

	pip list


4. Dataset Description and Access
4.1 Dataset Summary

The dataset consists of credit card transaction records with a binary fraud label (is_fraud). Fraud cases represent less than 0.57% of all transactions, making this a highly imbalanced classification problem.

Key feature groups include:

Transaction amount and merchant category
Temporal features (hour, day, month)
Location-related variables
Binary fraud indicator


4.2 Dataset Location

Shared Drive Link: https://drive.google.com/drive/folders/1g0nJlaGWAQW-JrlTrljRvXV5ynh9w-cw?usp=sharing 

Usage Instructions:

Download fraudTrain1.csv and fraudTest.csv
Place both files in the same directory as main.py
Do not rename the files

5. How to Run the Code

From the project directory:

	python main.py


The script executes sequentially:

Data loading
Basic EDA (raw data)
Preprocessing
Advanced EDA (post-cleaning)
Model training and hyperparameter tuning
Evaluation and visualisations
Final performance comparison


Execution time may be several minutes due to dataset size and hyperparameter tuning.

6. Model Selection Summary

Multiple models were trained and evaluated to understand different trade-offs between interpretability, recall, and precision:

Logistic Regression
Serves as a linear baseline. Highlights the limitations of naive thresholding under extreme class imbalance.

Decision Tree (Gini & Entropy)
Provides interpretable decision rules and strong recall. Useful for understanding feature-driven fraud patterns.

Random Forest
Reduces variance through bagging and improves stability. Demonstrates recall–precision trade-offs depending on tuning.

XGBoost
Achieves the strongest ranking performance (ROC-AUC) and high precision. Well-suited for high-confidence fraud flagging.

Support Vector Machine (RBF)
Captures non-linear boundaries but is computationally expensive. Used primarily for comparison.

Final model selection prioritises PR-AUC and F1-score over accuracy, reflecting real-world fraud detection requirements. The decision tree classifier is the best, followed by LightGBM & XGBoost



7. Reproducibility

To ensure experimental reproducibility:
Fixed random seeds were used wherever applicable:
	random_state=42 for model training and sampling
	RandomizedSearchCV(random_state=42)
Subsampling for tuning was performed using deterministic sampling.
All evaluation metrics are computed on a fixed test set.

Hardware Used
CPU-based execution (no GPU dependency)
Minimum recommended:
	16 GB RAM
	Quad-core CPU
Execution on lower-memory systems may increase runtime.

Due to randomized hyperparameter search and stochastic algorithms (e.g., Random Forest, XGBoost), minor metric fluctuations across runs are expected, though relative model rankings remain stable.


8. Known Issues and Important Notes

SMOTE was intentionally excluded from final training due to:
	Distortion of temporal and monetary distributions
	Unrealistic synthetic fraud patterns
	Reduced operational validity for banking systems

Accuracy is not a reliable metric due to extreme imbalance.
Threshold tuning is critical and static thresholds may not generalise across transaction types.
Dataset spans 2019–2020; real-world fraud patterns may have shifted since then. 
LightGBM, SHAP & LIME, did not install on Python 3.11.9 (Microsoft Store) without the virtual environment.

Author: DHWAJ JAIN & MAANYA RAICHURA
Course: DTSC301 – Machine Learning for Data Science I