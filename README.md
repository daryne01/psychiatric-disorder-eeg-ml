# Predicting Psychiatric Disorders Using EEG Data: A Machine Learning Approach

## Overview

This project investigates the classification of psychiatric disorders from EEG (Electroencephalography) data using supervised machine learning algorithms. Four classifiers — K-Nearest Neighbors, Random Forest, Support Vector Machine, and Logistic Regression — are evaluated on a Kaggle dataset containing 945 observations and 1,149 features.

The **Random Forest** model achieved the best performance with **88% test accuracy**, demonstrating strong potential for clinical decision support in psychiatric diagnosis.

## Dataset

**Source:** [EEG Psychiatric Disorders Dataset (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset)

| Property | Value |
|----------|-------|
| Observations | 945 (919 after cleaning) |
| Features | 1,149 |
| Target Variable | `main.disorder` |
| Classes | Anxiety Disorder, Mood Disorder, Schizophrenia, Addictive Disorder, Obsessive Compulsive Disorder, Trauma and Stress Related Disorder, Healthy Control |

Features include demographic data (age, sex, IQ, education), specific disorder labels, and EEG measurements across multiple brain regions (delta, theta, alpha, beta, gamma wave bands) along with coherence features.

## Project Structure

```
├── README.md
├── EEG_Psychiatric_Disorders_Dataset.csv       # Source dataset
├── ML_Coursework_MODIFIED.ipynb                # Full analysis notebook
├── cleaned_data.csv                            # Output after data cleaning step
├── preprocessed_data.csv                       # Output after preprocessing/scaling
└── coursework_machine_learning.pdf             # Coursework report
```

## Methodology

### Data Cleaning
1. Converted `main.disorder`, `specific.disorder`, and `sex` to string type for encoding.
2. One-hot encoded the `sex` column into binary `F` and `M` columns.
3. Dropped the `eeg.date` column (not relevant for classification).
4. Removed rows with missing values in `education` and `IQ` columns.
5. Dropped the empty `Unnamed: 122` column.
6. Checked for and confirmed no duplicate rows.

### Preprocessing
1. **Label Encoding** — Applied `LabelEncoder` to `main.disorder` and `specific.disorder`.
2. **Scaling** — Applied `MinMaxScaler` to normalize all numerical features to [0, 1].
3. **Feature Selection** — Selected top 50 features using `SelectKBest` with ANOVA F-test (`f_classif`) relative to the target variable.
4. **Standardisation** — Applied `StandardScaler` before model training.
5. **PCA** — Tested but rejected, as it reduced model accuracy by discarding discriminative information.

### Models Evaluated

| Model | Test Accuracy | Cross-Validation (Mean) |
|-------|:------------:|:-----------------------:|
| K-Nearest Neighbors | 0.4293 | 0.3918 |
| **Random Forest** | **0.8804** | **0.8544** |
| Support Vector Machine | 0.7609 | 0.6871 |
| Logistic Regression | 0.5380 | 0.5427 |

### Key Findings
- **Random Forest** was the top performer with low bias and low variance, achieving high precision and recall across most disorder classes.
- **SVM** was the second-best model (76% accuracy), benefiting from additional training data.
- **KNN** struggled with the high dimensionality of EEG data (curse of dimensionality) and was rejected.
- **Logistic Regression** underfit the data, suggesting the decision boundaries for this problem are non-linear.

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage

1. Place `EEG_Psychiatric_Disorders_Dataset.csv` in the same directory as the notebook.
2. Run the notebook:

```bash
jupyter notebook ML_Coursework_MODIFIED.ipynb
```

3. Execute all cells sequentially to reproduce: data cleaning → EDA → feature selection → model training → evaluation → comparison.

## References

1. Shagass, C., Roemer, R.A. and Straumanis, J.J., 1982. *Relationships between psychiatric diagnosis and some quantitative EEG variables.* Archives of General Psychiatry, 39(12), pp.1423–1435.
2. Cohen, M.X., 2017. *Where does EEG come from and what does it mean?* Trends in Neurosciences, 40(4), pp.208–218.
3. Hosseinifard, B., Moradi, M.H. and Rostami, R., 2013. *Classifying depression patients and normal subjects using machine learning techniques.* Computer Methods and Programs in Biomedicine, 109(3), pp.339–345.
4. Park, S.M. et al., 2021. *Identification of major psychiatric disorders from resting-state EEG using a machine learning approach.* Frontiers in Psychiatry, 12, p.707581.
5. Edla, D.R. et al., 2018. *Classification of EEG data for human mental state analysis using Random Forest Classifier.* Procedia Computer Science, 132, pp.1523–1532.
6. Guerrero, M.C., Parada, J.S. and Espitia, H.E., 2021. *EEG signal analysis using classification techniques: Logistic regression, artificial neural networks, support vector machines, and convolutional neural networks.* Heliyon, 7(6).
7. Mumtaz, W. et al., 2018. *A machine learning framework involving EEG-based functional connectivity to diagnose major depressive disorder (MDD).* Medical & Biological Engineering & Computing, 56, pp.233–246.

## Author

Student ID: 23012721 | Module: UFCFMJ-15-M
