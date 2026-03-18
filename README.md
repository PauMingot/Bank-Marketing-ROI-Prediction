# Bank-Marketing-ROI-Prediction
Machine Learning models and business strategy to predict bank customer conversion and optimize marketing ROI.

# Predictive Modeling for Bank Customer Conversion 🏦📊

## Project Overview
This project aims to optimize a bank's telemarketing campaigns by predicting whether a client will subscribe to a term deposit. Instead of relying solely on statistical metrics, the core focus of this project is **translating machine learning predictions into actionable business strategies and calculating the expected ROI** for different marketing actions (calls, emails, personalized offers).

## 🚀 Business Value & Key Results
After evaluating multiple algorithms, **XGBoost** was selected as the optimal model (n_estimators: 200, max_depth: 4, learning_rate: 0.05). 

Based on the SHAP values and model interpretation, we identified the **Ideal Client Profile**:
* **Financials**: High average balance, no active mortgages, and no personal loans.
* **Demographics**: Divorced or single individuals with tertiary education.
* **Timing & Contact**: Best contacted in March or October via mobile phone.
* **History**: High conversion probability if they accepted previous campaigns and sufficient time has passed since the last contact.

### 💡 The Marketing Business Function
A custom Python function (`FuncionNegocioPython.py`) was developed to deploy the model in a real-world scenario. It automatically:
1. Segments current and future clients.
2. Decides the optimal contact strategy (Call vs. Do Not Call).
3. Assigns personalized offers to maximize conversion.
4. Calculates the final Net Profit (ROI) of the marketing campaign based on predicted successes.

## 🛠️ Tech Stack
* **Programming & Modeling:** Python (Scikit-learn, XGBoost, CatBoost, TensorFlow), R.
* **Data Manipulation:** Pandas, NumPy, SQL.
* **Data Visualization:** Matplotlib, Seaborn, PowerBI.
* **Algorithms Tested:** Logistic Regression, Decision Trees, Random Forest, XGBoost, CatBoost, Neural Networks, Poisson Regression.

## 📂 Repository Structure
* `XBOOST.py`: Data preprocessing, hyperparameter tuning, and final XGBoost model training.
* `FuncionNegocioPython.py`: The business-oriented script that calculates ROI and assigns marketing actions.
* `Alternative_Models/`: Contains scripts for CatBoost, Random Forest, and Neural Networks evaluated during the selection phase.
* `Final_Presentation.pdf`: Executive summary and visual insights of the project.
* `bank.csv`: The dataset used for training and testing.

## Authors
* Pau Mingot Lapiedra
* Ignacio Martínez Caballero
* Neus Signes Pedro
