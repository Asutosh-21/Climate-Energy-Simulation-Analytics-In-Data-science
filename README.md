# 🌍 Global Climate & Energy Simulation Analytics 

**Domain:** Energy • Sustainability • Environment  

This project forecasts energy demand, models pollution impact, and simulates renewable vs. non-renewable adoption scenarios.  
It demonstrates **end-to-end data science, machine learning, and simulation skills** that are valuable in consulting, energy, and climate-tech roles.



## 📌 Purpose & Problem Statement
**Problem:**  
Governments, utilities, and sustainability teams need to understand:
* How future energy demand will grow.
* How different renewable-adoption strategies affect emissions.
* The uncertainty in those predictions.

**Solution:**  
This project provides:
1. **Time-series forecasting** of energy demand and emissions.
2. **Scenario simulation** (Monte Carlo) to quantify uncertainty.
3. **Interactive visualization** so stakeholders can explore “what-if” outcomes.

---

**Workflow Diagram**  
Data Ingestion → Cleaning & Feature Engineering →
Time-Series Forecasting (Prophet & LSTM) →
Monte Carlo Simulation →
Interactive Dashboard (Plotly Dash)


---

## ⚙️ Tech Stack
| Layer            | Tools & Libraries                               |
|------------------|--------------------------------------------------|
| Data Handling    | Python, Pandas, NumPy                            |
| Forecasting      | Prophet, TensorFlow/Keras (LSTM), scikit-learn    |
| Simulation       | Monte Carlo (NumPy random sampling)               |
| Visualization    | Plotly, Dash, Matplotlib, Seaborn                 |
| Deployment (opt) | Docker, Gunicorn/Heroku or any cloud platform     |

---

## 🔑 How It Works – Step by Step

1. **Data Ingestion & Cleaning**  
   * Load multi-year climate and energy datasets.  
   * Handle missing values, resample to monthly/annual frequency.

2. **Exploratory Data Analysis**  
   * Visualize energy mix, pollution levels, and seasonal patterns.

3. **Time-Series Forecasting**  
   * **Prophet**: quick baseline forecasts with seasonality & uncertainty bands.  
   * **LSTM**: deep learning model for long-term, non-linear dependencies.

4. **Monte Carlo Simulation**  
   * Define probability distributions for demand growth & renewable adoption.  
   * Run thousands of simulations to estimate a range of future outcomes.

5. **Visualization & Dashboard**  
   * Plotly charts to compare actual vs. predicted values and scenario distributions.  
   * Optional Dash app for interactive “what-if” exploration.

---

## 🧩 Key Features
* Multi-model forecasting with saved model artifacts for easy reuse.
* Probabilistic simulation for risk analysis.
* Clean, modular architecture suitable for production or further research.
* Ready-to-deploy interactive dashboard.

---

## 📈 Results & Insights
* Accurate multi-year demand forecasting with clear uncertainty intervals.
* Quantitative assessment of how renewable-adoption rates influence emissions.
* Business-ready visuals for policy makers and energy planners.

## 🏃 Run locally (PowerShell)

Use the included helper to start the Streamlit dashboard. From the repo root run:

```powershell
# (optional) create and activate a venv if you want an isolated environment
python -m venv .venv
#.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r .\requirements.txt

# Start the app (recommended):
.\start-app.ps1

# Or run directly using python -m to avoid PATH issues:
python -m streamlit run .\app.py
```

Notes:
- If `.venv` exists, `start-app.ps1` will use its Python executable. Otherwise it falls back to the system `python`.
- If PowerShell blocks script execution, you may need to adjust execution policy (only if you trust the script):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## ☁️ Deploy to Streamlit Cloud

Push this repository to GitHub, then connect it from https://streamlit.io/cloud and choose the `main` branch. Streamlit Cloud will install packages from `requirements.txt` and run `streamlit run app.py` automatically.



