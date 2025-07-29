#  Marketing Mix Modeling (MMM) Prototype

This project is a **prototype Marketing Mix Model (MMM)** system that ingests marketing spend & sales data, applies **adstock and saturation transformations**, trains different models (Linear, Ridge, Random Forest), and serves predictions via a **FastAPI REST API**.

The system is designed with **clean architecture, modularity, and extensibility** in mind. Future models like **Bayesian MMM (PyMC-Marketing)** or **Meta’s Robyn** can be plugged in easily using the `ModelFactory`.

---

## 🚀 Features

✅ Modular pipeline with **Strategy Pattern** for easy model swapping
✅ **Adstock & saturation transformations** for marketing effects
✅ **Lag, rolling mean, interaction, and time-based features** for better modeling
✅ REST API using **FastAPI** with `/train`, `/predict`, `/evaluate` endpoints
✅ **Parquet caching** for efficient data storage & retrieval
✅ Clean code following **SOLID, DRY, and KISS principles**
✅ Unit tests for **data ingestion, preprocessing, model service, and API endpoints**

##  Setup Instructions

```bash
# 1️⃣ Clone repository
git clone <repo_url>
cd mmm_project

# 2️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Start API server
uvicorn api.main:app --reload
````

---

## 📡 API Endpoints
# postman collection added

| Method | Endpoint    | Description                    |
| ------ | ----------- | ------------------------------ |
| POST   | `/train`    | Train the MMM model            |
| POST   | `/predict`  | Predict sales from spend input |
| GET    | `/evaluate` | Evaluate model performance     |

### 📌 Example Predict Request

```json
[
  {
    "date": "2023-01-01",
    "tv_spend": 20000,
    "radio_spend": 5000,
    "social_media_spend": 3000,
    "search_spend": 4000,
    "print_spend": 2000,
    "outdoor_spend": 1000
  }
]
```

---

## 📊 Model Comparison (Final Results)

| Model  | R²         | RMSE     | MAPE     | Rank           |
| ------ | ---------- | -------- | -------- | -------------- |
| Linear | **0.0079** | **1.97** | **322k** | ✅ Best         |
| Ridge  | 0.0014     | 1.98     | 323k     | Slightly worse |
| RF     | -0.034     | 2.01     | 330k     | Worst of three |

🔹 **Final Choice → Linear Regression** (best trade-off between simplicity and performance)

---

## 🔄 Step-by-Step Improvements

| Step | Change Made                             | Impact                                      |
| ---- | --------------------------------------- | -----------------------
| 1️⃣  | Baseline Linear/Ridge model             | Very poor R² (≈ -2.3)
| 2️⃣  | Added adstock & saturation              | Small improvement
| 3️⃣  | Added log-transform for features        | Stabilized coefficients
| 4️⃣  | Added lag & rolling mean features       | Slight R² improvement
| 5️⃣  | Added interaction & time-based features | Further small gains
| 6️⃣  | Switched target to log(sales)           | Final R² ≈ 0.008 (slightly positive)


---

## 🎯 Final Model Choice & Reasoning

✔ **Linear Regression** – simple, interpretable, and slightly better than Ridge and RF.

✔ **Ridge** – tested for regularization, but no major improvement.

❌ **Random Forest** – captured non-linear effects but underperformed.

---

## 🔮 Future Improvements

*  Use **Bayesian MMM (PyMC-Marketing)** with priors on adstock & saturation
*  Add **external drivers** (promotions, pricing, competition, holidays)
*  Perform **hyperparameter tuning** for adstock decay, alpha, gamma
*  Incorporate **seasonality/trend models (Prophet + MMM hybrid)**
*  Deploy with **MLflow for experiment tracking** and **Docker/Kubernetes*
*  Orchestration pipeline using Prefect or Apache Airflow
