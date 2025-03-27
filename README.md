# Network Intrusion Detection using CICIDS-2017 Dataset

## Overview
This project focuses on building and evaluating machine learning models for network intrusion detection using the **CICIDS-2017 dataset**. The dataset contains network traffic data labeled as normal or attack, making it suitable for binary and multi-class classification tasks.

## Dataset
The dataset can be downloaded from Kaggle:
[CICIDS-2017 Dataset](https://www.kaggle.com/datasets/sateeshkumar6289/cicids-2017-dataset/data)

The dataset includes various network flow features such as packet length, flow duration, protocol type, and more, which help distinguish normal traffic from attacks.

## Methodology
1. **Data Preprocessing**:
   - Handling missing values using **missingno** library
   - Feature scaling and encoding categorical variables
   - Splitting into training and testing sets

2. **Exploratory Data Analysis (EDA)**:
   - Correlation matrix to identify feature dependencies
   - Class distribution analysis

3. **Model Selection & Training**:
   - Trained multiple models including Logistic Regression, Random Forest, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)
   - Hyperparameter tuning using GridSearchCV
   - Cross-validation for robustness

4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix, ROC Curve, Precision-Recall Curve

## Models Used
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

## Key Findings
- Random Forest outperformed other models with the highest accuracy.
- The ROC curve and Precision-Recall curve confirmed excellent performance for most models.
- SVM showed strong classification performance, but training time was higher compared to other models.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno
```

### Running the Notebook
1. Download the dataset from Kaggle and place it in the working directory.
2. Open the Jupyter Notebook and run all cells sequentially.
3. Evaluate model performance using visualizations and metrics.

## Conclusion
This project demonstrates how machine learning models can effectively detect network intrusions. Future work could explore deep learning models or ensemble techniques to further enhance accuracy.

---

