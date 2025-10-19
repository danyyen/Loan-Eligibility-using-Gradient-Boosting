# Project Overview 
## Loan-Eligibility-using-Gradient-Boosting
Built a machine learning model to predict loan approval likelihood based on applicant credit, income, and employment data

### Description
In this dataset, you must explore and cleanse a dataset consisting of over 1,00,000 loan records to determine the best way to predict whether a loan applicant should be granted a loan or not. You must then build a machine learning model that returns the unique customer ID and a loan status label that indicates whether the loan should be given to that individual or not.

### Problem statement:
- Banks and credit institutions need quick, data-driven tools to evaluate loan applications.
- Manual screening is inconsistent and time-consuming. The bank wants to reduce ineligible approvals.
- They are most concerned about false positives — approving people who actually are not eligible.
- That means: They should prioritize high Precision, because High Precision = few false approvals. Low Precision means many ineligible people slipped through as “eligible".

### Tools & Tech Stack
- Languages: Python
- Libraries: scikit-learn, XGBoost, imbalanced-learn (SMOTE), fancyimpute, matplotlib, seaborn
- Environment: Spyder IDE
- Model: Gradient Boosting Classifier (best performance)
- Deployment-ready assets: GBM_Model_version.pkl, Output_LoanResult.csv

### Data and Preprocessing

**Dataset**  
Loan application data containing features such as income, credit score, debt ratio, and job history.

**Key Preprocessing Steps**
- Removed duplicate entries based on `Loan ID`
- Handled outliers using IQR capping and percentile thresholds
- Standardized inconsistent categorical values (e.g., merged `"HaveMortgage"` and `"Home Mortgage"`)
- Imputed missing values using:
- SoftImpute for numerical features - `KNN imputation` for categorical and mixed types
- Factorized and one-hot encoded categorical features
- Scaled numerical features to ensure consistent model performance

**Evaluation criteria**: To achieve a passing grade, the accuracy of the model has to be at least 70%.

### Modelling Approach
- Tried multiple models: Logistic Regression, Random Forest, XGBoost, and Gradient Boosting
- Used SMOTE to balance class distribution and handle imbalance between approved/rejected loans
- Evaluated each model with Mathew's Correlation Coefficient, ROC-AUC, F1-score, Precision, and Recall

### Model Deployment and Testing
- Saved final model using joblib.dump()
- Loaded model in a separate script for real-time predictions on test data
- Added logic to output “Loan Approved” or “Loan Rejected” status with probabilities
- Exported final predictions to 'Output_LoanResult.csv'

### Key Results and Insights
- Achieved 75% ROC-AUC and 58% precision
- Top predictive features: Term, Current loan amount, credit score, annual income and Home Ownership.
- These features provide valuable signals for lenders: 
    - Term: Longer loan durations may increase risk, guiding lenders to adjust interest rates or approval criteria.
    - Current Loan Amount: Higher amounts suggest greater financial burden, helping set loan caps or require additional documentation.
    - Credit Score: A direct measure of creditworthiness, enabling personalized offers and risk segmentation.
    - Annual Income: Indicates repayment ability, used to calculate debt-to-income ratios and tailor loan limits.
    - Home Ownership: Suggests financial stability and potential collateral, influencing approval likelihood and loan terms.
- Together, these insights empower lenders to make faster, fairer, and more informed loan decisions while minimizing default risk.



