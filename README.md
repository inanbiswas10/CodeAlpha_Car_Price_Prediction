# 🚗 AutoValuate AI - Used Car AI Valuation & Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Pytest-4%20Passed-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://YOUR-APP-NAME.streamlit.app)

**AutoValuate AI** is an enterprise-grade machine learning platform that predicts used car resale prices, simulates ownership sensitivity scenarios, calculates Total Cost of Ownership (TCO) and generates dynamic buyer/seller negotiation matrices

Developed as part of the **CodeAlpha Data Science Internship Program**

---

## 🌐 Live Demo

👉 **[Launch AutoValuate AI](https://auto-valuate-ai.streamlit.app/)**

---

## 🌟 Core Modules & Intelligence Views

| View | Description |
|------|-------------|
| **🚀 Executive Command Center** | Real-time vehicle valuation estimator with feature impact breakdowns and confidence ranges |
| **⚡ What-If Sensitivity Lab** | Live scenario simulator adjusting mileage and hold years to predict future price shifts |
| **⚔️ Competitive Market Intelligence** | Brand and body-type price positioning charts across historical market benchmarks |
| **🛡️ Buyer Negotiation Matrix** | Optimal offer strategies, deal rating badges and target negotiation thresholds |
| **🎯 Target Budget & Filter Engine** | Interactive inventory scanner filtering cars by budget ceiling, mileage, fuel and transmission |
| **📊 Cluster Archetype Analysis** | Unsupervised clustering highlighting market segments (Budget Daily, Mid-Range, Luxury) |
| **💰 TCO & Depreciation Estimator** | 5-year total ownership cost projections including fuel, maintenance, insurance and resale decay |
| **📈 Visual Analytics & Correlation Lab** | Pairwise feature distributions and feature correlation heatmaps |

---

## 🛠️ Tech Stack

- **Frontend / Web Portal**: Streamlit
- **Data Engine**: pandas, NumPy
- **Visualisation**: Plotly Express & Plotly Graph Objects
- **Machine Learning**: scikit-learn (Random Forest, Decision Tree, Linear Regression), XGBoost
- **Model Serialization**: joblib
- **Automated Testing**: pytest

---

## 📐 Machine Learning Model Performance

Multiple regression algorithms were evaluated on the CarDekho used car dataset:

| Model | R² Score | MAE (Lakhs ₹) | RMSE (Lakhs ₹) | Status |
|---|:---:|:---:|:---:|:---:|
| **Random Forest Regressor** | **0.9657** | **0.42** | **0.68** | **Selected Core Model** |
| XGBoost Regressor | 0.9520 | 0.48 | 0.79 | Evaluated Benchmark |
| Decision Tree Regressor | 0.9140 | 0.61 | 0.98 | Evaluated Benchmark |
| Linear Regression | 0.8420 | 0.85 | 1.32 | Baseline Benchmark |

*Selected Model: **Random Forest Regressor**, chosen for its superior variance coverage (R² = 96.57 %) and minimal prediction error*

---

## 📂 Project Structure

```text

CodeAlpha_Car_Price_Prediction/
├── .streamlit/
│   └── config.toml            # Custom dark UI theme configuration
├── app/
│   └── main.py                # Streamlit 8-window web application
├── data/
│   └── car_data.csv           # CarDekho used car dataset
├── models/
│   └── car_price_model.pkl    # Trained Random Forest regression pipeline
├── src/
│   ├── data_loader.py         # Data cleaning, ingestion & preprocessing
│   ├── features.py            # Vehicle age, mileage/yr & luxury feature engineering
│   └── train.py               # Model training & artifact serialization script
├── tests/
│   └── test_car_prediction.py # Automated pytest verification suite (4 tests)
├── .gitignore                 # Git exclusion settings
├── requirements.txt           # Production dependencies
└── README.md                  # Platform technical documentation
```

---

## 🚀 Local Installation & Setup

### 1. Clone Repository

```bash

git clone https://github.com/inanbiswas10/CodeAlpha_Car_Price_Prediction.git
cd CodeAlpha_Car_Price_Prediction
```

### 2. Create & Activate Virtual Environment

```bash

# On Windows PowerShell:

python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux:

python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash

pip install -r requirements.txt
```

### 4. Launch Application

```bash

streamlit run app/main.py
```

The dashboard will launch automatically in your browser at `http://localhost:8501`.

---

## 🔬 Automated Testing Suite

To run the automated Pytest test suite:

```bash

pytest tests/ -v
```

All 4 unit tests (data existence, data loader integrity, model artifact presence and model loading) pass

---

## 👤 Author & License

**Inan Biswas**

- GitHub: [inanbiswas10](https://github.com/inanbiswas10)
- LinkedIn: [Inan Biswas](https://www.linkedin.com/in/inanbiswas10)

Developed for educational and demonstration purposes as part of the CodeAlpha Data Science Internship Program
