# 🫁 Lung Cancer Survival Prediction Using Machine Learning

## 📌 Project Overview

Lung Cancer Survival Prediction is a machine learning project designed to analyze patient-related data and predict survival outcomes based on clinical, demographic, and lifestyle characteristics.

The project uses Python and machine learning techniques to explore patterns in lung cancer patient data, identify relevant features, and develop a predictive classification model. It includes data preprocessing, Exploratory Data Analysis (EDA), feature engineering, model training, and performance evaluation.

The primary objective is to investigate how machine learning can support healthcare data analysis by estimating patient survival outcomes from available historical data.

**Important:** This project is intended for educational and research purposes only. Its predictions are not medical advice and must not be used as the sole basis for clinical decisions.

## 🎯 Project Objectives

- Analyze lung cancer patient data using Exploratory Data Analysis.
- Clean and preprocess the dataset for machine learning.
- Handle missing values, categorical variables, and numerical features.
- Identify patterns and relationships between patient characteristics and survival outcomes.
- Apply feature engineering and encoding techniques.
- Build a Random Forest classification model.
- Evaluate model performance using appropriate classification metrics.
- Explore the factors associated with predicted survival outcomes.

## 📂 Dataset Description

The project uses a lung cancer patient dataset containing demographic, medical, and lifestyle information.

The dataset includes features such as:

| Feature | Description |
|---|---|
| Age | Patient's age |
| Gender | Patient's gender |
| Country | Country associated with the patient record |
| Diagnosis Date | Date of lung cancer diagnosis |
| Cancer Stage | Stage of lung cancer |
| Family History | Whether the patient has a family history of cancer |
| Smoking Status | Patient's smoking history or status |
| BMI | Body Mass Index |
| Cholesterol Level | Recorded cholesterol level |
| Hypertension | Whether the patient has hypertension |
| Target: `survived` | Survival outcome to be predicted |

*Note: Confirm the exact feature names, definitions, units, and target encoding against your dataset. The target values must be interpreted according to the dataset's documentation.*

## 🛠️ Technologies Used

- **Python** — Programming and model development
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Preprocessing, model training, and evaluation
- **Jupyter Notebook** — Interactive data analysis
- **Streamlit** — Optional interactive web application

## 🔄 Project Workflow

### 1. Data Loading and Understanding

- Load the dataset using Pandas.
- Examine the number of rows and columns.
- Inspect column names and data types.
- Review summary statistics.
- Identify missing values and duplicate records.

### 2. Data Cleaning and Preprocessing

- Handle missing values appropriately.
- Check for duplicate records.
- Convert columns to appropriate data types.
- Process date-related features.
- Encode categorical variables.
- Scale numerical features if required by the selected algorithm.
- Separate input features from the target variable.

Preprocessing should be fitted on the training data only, using a Scikit-learn pipeline where possible, to reduce data leakage.

### 3. Exploratory Data Analysis (EDA)

EDA helps uncover patterns and understand the dataset before training the model.

Key analyses include:

- Distribution of survival outcomes.
- Patient age distribution.
- Distribution of cancer stages.
- Smoking status and survival patterns.
- BMI and cholesterol distributions.
- Relationships between patient characteristics and survival.
- Correlation analysis of relevant numerical variables.

### 4. Feature Engineering

Feature engineering prepares existing variables for effective model training.

Possible techniques include:

- Extracting diagnosis year or month from the diagnosis date.
- Encoding categorical features.
- Investigating relevant clinical and demographic variables.
- Removing unique identifiers that do not provide useful predictive information.
- Checking for highly correlated or redundant features.

Features should be selected based on their availability at the intended prediction time. Variables that reveal future outcomes must not be used as model inputs.

### 5. Target Variable and Class Imbalance

The target variable is `survived`.

Before training, verify the meaning of each target value and examine the distribution of the classes.

If the dataset is imbalanced, the model may favor the majority class and perform poorly when identifying the minority class.

Potential approaches include:

- Using `class_weight="balanced"` with compatible classifiers.
- Applying stratified train-test splitting.
- Evaluating precision, recall, and F1-score for each class.
- Experimenting with resampling techniques on the training set only.
- Adjusting the classification threshold based on validation results.

Class imbalance should be addressed carefully rather than assuming that accuracy alone reflects model quality.

### 6. Machine Learning Model

The project uses the **Random Forest Classifier**, an ensemble learning algorithm that combines predictions from multiple decision trees.

Random Forest can model nonlinear relationships and interactions between features. It is suitable for exploring classification problems involving both numerical and encoded categorical variables.

The general workflow is:

1. Define the input features and target variable.
2. Split the data into training and testing sets.
3. Fit preprocessing transformations on the training data.
4. Train the Random Forest model.
5. Generate predictions on unseen test data.
6. Evaluate the results using multiple performance metrics.

Other classifiers, such as Logistic Regression and Decision Tree, can be evaluated for comparison.

### 7. Model Evaluation

The following metrics can be used to evaluate the trained classifier.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures the proportion of positive predictions that are correct |
| Recall | Measures the proportion of actual positive cases correctly identified |
| F1-score | Combines precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |
| ROC-AUC | Evaluates class discrimination using predicted probabilities |

For survival prediction, interpret the metrics in the context of the target encoding and the consequences of false-positive and false-negative predictions.

If the model predicts only one class, investigate class imbalance, target encoding, preprocessing, model configuration, and the training data before considering it suitable for use.

## 📊 Expected Analysis Outcomes

The project aims to investigate:

- Which patient characteristics are associated with survival outcomes.
- Whether survival outcomes vary across cancer stages or demographic groups.
- How smoking status and other recorded characteristics relate to survival.
- Whether Random Forest can distinguish between the target classes.
- Which features contribute most to the model's predictions.
- How class imbalance affects predictive performance.

Actual findings and evaluation scores should be added after the analysis is complete. Feature importance indicates predictive contribution, not necessarily causation.

## 📁 Project Structure

A suggested repository structure is:

```text
Lung-Cancer-Survival-Prediction/
│
├── data/
│   └── lung_cancer_dataset.csv
│
├── notebooks/
│   └── lung_cancer_analysis.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Adjust the structure to match the files actually present in your repository. Include `app.py` only if you have created a Streamlit application.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Visual Studio Code or Jupyter Notebook.
- Git, if cloning the repository.

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Folder

```bash
cd Lung-Cancer-Survival-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells in sequence.

### Optional: Run the Streamlit Application

If your project includes a properly configured `app.py`, run:

```bash
streamlit run app.py
```

Ensure that the application loads the same fitted preprocessing pipeline and model used during evaluation.

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
streamlit
```

Install them using:

```bash
pip install -r requirements.txt
```

## ⚠️ Limitations

- Predictive performance depends on dataset quality and representativeness.
- Historical associations do not establish causal relationships.
- Class imbalance can result in poor minority-class detection.
- Predictions may be affected by missing variables, measurement errors, and dataset bias.
- A model's performance on a test dataset does not guarantee performance on real-world patients.
- Clinical use would require independent validation, appropriate clinical oversight, and assessment of safety and fairness.

## 🚀 Future Enhancements

- Compare Random Forest with Logistic Regression, XGBoost, and other suitable classifiers.
- Apply hyperparameter tuning and cross-validation.
- Improve minority-class detection through appropriate imbalance-handling methods.
- Use SHAP or other explainability methods to investigate model predictions.
- Develop a Streamlit interface for demonstration purposes.
- Evaluate calibration and performance across relevant patient subgroups.
- Validate the model on independent, representative clinical data before considering any clinical application.

## 👩‍💻 Author

**Mansi Mishra**

B.Tech Student | Aspiring Data Analyst & Data Scientist

GitHub: [Visit My GitHub Profile](https://github.com/mansi16mini07-creator)

## 📄 License

This project is intended for educational and research purposes. Check the dataset's original license and usage conditions before redistributing the data. Consider adding an appropriate license for your code if you want others to reuse it.

## ⭐ Acknowledgements

Thanks to the dataset providers and the open-source Python community for the resources and tools used in this project.


