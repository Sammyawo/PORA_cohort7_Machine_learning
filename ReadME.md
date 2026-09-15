# Insurance Prediction API (FastAPI)
This project provides an API for predicting whether client would churn or not based on user input using a trained churn machine learning model.

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [1. Train and Save the Model](#1-train-and-save-the-model)
- [2. Run the FastAPI Backend](#2-deploy-fastapi)
- [Endpoints](#endpoints)
- [Example Input and Output](#example-input-and-output)
- [File Structure](#file-structure)
- [License](#license)

## Description
The FastAPI application loads the trained XGBoost model and exposes an endpoint for predicting if user `Exited` or not based on user's input. The model is trained on the following features:

- CreditScore
- Geography (one-hot encoded: Geography_Germany, Geography_Spain; France is the baseline)
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

The API predicts if customer `Exited` on these features.

## Requirements
To set up and run this project, you’ll need the following Python packages:

- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pandas`
- `joblib`
- `numpy`
- `xgboost`
- `pydantic`

You can install these dependencies by running:
```bash
pip install -r requirements.txt
```

## Getting Started
1. Train and Save Model
Train the XGBoost Classifier model using Scikit-Learn and XGBoost Classifier and save the trained model to files for deployment:
```bash
python model_pipeline.py
```
This will create the model.pkl file in the model/ directory.

2. The FastAPI application (api.py) loads the saved model and provides an endpoint for predictions. Run it using uvicorn:
```bash
uvicorn api:app --reload
```
This will start the FastAPI server at http://127.0.0.1:8000

## Endpoints
![API Image](src/)

- POST /api/predict
    - Description: Accepts feature values and returns 1 or 0 indicating whether client churned or not.
    - Input JSON:
    ```bash
    { 
        "credit_score": "",
        "geography": "",
        "age": "",
        "gender": "",
        "tenure": "",
        "balance": "",
        "num_of_products": "",
        "has_credit_card": "",
        "is_active_member": "", 
        "estimate_salary": "",
    }
    ```
    - Output JSON:
    ```bash
    {
        "churn_status": "stay",
        "churn_value": 1
    }
    ```
## Example Input and Output
Example Input:

Credit Score = 530 
Geography = France 
Age = 45 
Gender = Male 
Tenure = 5 
Balance = 8504003 
Num of Products = 3 
Has Credit Card = Yes 
Is Active Member = Yes 
Estimate Salary = 7392020

Example Output:

Predicted Churn Status: Stay/Leave

## File Structure
The project directory is structured as follows:

```text
📦 XGBoost Cohort Backend 
├─ data
│  └─ churn_dataset.csv
├─ model
│  └─ model.pkl
├─ src
├─ .gitignore
├─ api.py
├─ model_pipeline.py
├─ README.md
└─ requirements.txt
```

## License
This project is licensed under [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)