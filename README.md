
# Project Title: IFRS 9 Expected Credit Loss (ECL) Modeling using Advanced Machine Learning Author: [Mohammed Nazrul Islam]

# Dataset: 
The dataset is collected from GitHub URL: https://raw.githubusercontent.com/LEARNEREA/Data_Science/main/Data/credit_risk_dataset.csv or https://www.kaggle.com/datasets/laotse/credit-risk-dataset. Subsequently, new features are derived as per the requirements of IFRS 9 for the calculation of PD. This project develops a predictive framework for credit risk under IFRS 9 standards. It leverages a dataset of over 32,500 entries to predict the Probability of Default (PD).

# Methodology

# Data Cleaning:	
Handled missing values by using median imputation for critical fields like person_emp_length and loan_int_rate, and used the IQR method to cap outliers in age and credit history, ensuring the data is robust for modeling.

# Data Quality: 
Implemented median imputation for missing employment length and interest rates. Outliers in age and credit history were capped using IQR boundaries.

# Risk-Specific Engineering: 
Categorical variables (Home Ownership, Loan Intent) were transformed into default probabilities. Credit grades were transformed using Weight of Evidence (WoE) to linearize relationships with risk. Particularly, the Weight of Evidence (WoE) for loan_grade and Probability Mapping (Target Encoding) for categorical variables like person_home_ownership.

# Addressing Data Bias: 
Balanced the target class using SMOTE, increasing the minority default class from ~5,600 to over 20,400 samples for more robust training.

# Model Performance: 
Ensemble models (Random Forest, XGBClassifier) outperformed traditional models. Based on a comprehensive review of all metrics, XGBClassifier, Random Forest, and GradientBoostingClassifier stand out as the strongest performers.

# Model Comparison	
Compared 12 different algorithms, ranging from simple Naive Bayes to complex ensembles like XGBoost and Gradient Boosting.

# 1. Accuracy (Training, Testing, and Validation):

Top Performers: Random Forest, XGBClassifier, and GradientBoostingClassifier consistently show the highest accuracy across training, testing, and validation sets. Random Forest and Decision Tree achieve perfect training accuracy (almost 1.0), indicating they fit the training data extremely well. However, Decision Tree's validation accuracy (0.889) is significantly lower than its training accuracy, suggesting overfitting. XGBClassifier and Random Forest maintain high validation accuracies (0.973 and 0.933, respectively), indicating good generalization.

Mid-Range: Logistic Regression, k-Nearest Neighbors, Naive Bayes, and AdaBoostClassifier show moderate accuracies (around 0.81-0.88).

Lower Performers: Linear SVC, Support Vector Machines, Perceptron, and Stochastic Gradient Descent tend to have the lowest accuracies, especially Perceptron and SGDClassifier.

# 2. Mean Squared Error (MSE):

Lower MSE is better. This metric is usually for regression, but can be applied to classification predictions. It indicates the average squared difference between actual and predicted values.
Top Performers (Lowest MSE): XGBClassifier (0.0699), Random Forest (0.0736), and GradientBoostingClassifier (0.095) have the lowest validation MSE, aligning with their high accuracy.
Higher MSE: Models with lower accuracy, like Perceptron and Stochastic Gradient Descent, naturally have higher MSE values.

# 3. Precision, Recall, and F1 Score (Training - weighted average):

These metrics are crucial for evaluating classification models, especially when class imbalance is present (even after SMOTE, it's good to consider how well the model identifies each class).
F1 Score (weighted average):
Top Performers: Random Forest (1.00), Decision Tree Classifier (0.999), and XGBClassifier (0.973) show excellent F1 scores, indicating a good balance between precision and recall on the training set. GradientBoostingClassifier (0.906) and AdaBoostClassifier (0.877) also perform very well.

The high training F1 scores for Random Forest and Decision Tree again hint at their ability to perfectly fit the training data, while their validation accuracies tell us about their generalization.

Precision and Recall (weighted average): The trends for precision and recall generally follow the F1-score. Models like Random Forest and XGBClassifier exhibit high precision and recall on the training set.

# 4. CPU Times:

This metric reflects the computational cost of training each model.

Fastest: Naive Bayes, Linear SVC, Perceptron, Stochastic Gradient Descent, and Decision Tree Classifier are generally very fast to train (under a second).
Moderate: XGBClassifier, Random Forest, GradientBoostingClassifier, and AdaBoostClassifier take a few seconds to train, which is reasonable for their performance.
Slowest: Support Vector Machines (SVC) is by far the slowest, taking over 2 minutes, even with probability=True enabled.

# 5. Validation AUC-ROC and Gini Coefficient:

Top Performers: XGBClassifier (AUC: 0.95, Gini: 0.897), GradientBoostingClassifier (AUC: 0.923, Gini: 0.846), and Random Forest (AUC: 0.93, Gini: 0.864) show the highest AUC-ROC and Gini Coefficients. This indicates their superior ability to discriminate between positive and negative classes on unseen data.

## Meeting the >0.75 AUC-ROC Threshold: AdaBoostClassifier (AUC: 0.893), Logistic Regression (AUC: 0.840), k-Nearest Neighbors (AUC: 0.807), and Decision Tree Classifier (AUC: 0.805) also meet this threshold.

Below Threshold: Support Vector Machines, Linear SVC, Naive Bayes, Perceptron, and Stochastic Gradient Descent fall below the 0.75 AUC-ROC benchmark.
Overall Conclusion:

# conclusion:
Based on a comprehensive review of all metrics, XGBClassifier, Random Forest, and GradientBoostingClassifier stand out as the strongest performers. They consistently achieve high accuracy, low MSE, excellent F1-scores, and superior AUC-ROC/Gini Coefficients. While Random Forest and Decision Tree can overfit the training data, XGBClassifier and GradientBoostingClassifier show robust generalization capabilities. The choice among the top three might come down to a balance between performance, interpretability, and specific business needs for False Positives vs. False Negatives.

# Disclaimer and acknowledgement:
This project is an applied learning exercise using publicly available GitHub repositories and is an adaptation of standard methods as inspired by the mentor, Eng. Golam Rabbany M.Eng in ICT at BUET, WINGS - Institute of Research, Innovation, Incubation( https://wingsiriic.com.bd/).

