<div align="center">

<h1>StockCast</h1>

<p>A full-stack stock price prediction platform powered by a Hybrid LSTM-GRU deep learning model,<br/>built for NSE-listed stocks with an interactive React dashboard.</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-4169E1?style=flat-square&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite&logoColor=white" />
  <img src="https://img.shields.io/badge/Tailwind%20CSS-3-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" />
  <img src="https://img.shields.io/badge/Recharts-2.x-22c55e?style=flat-square&logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.x-EA580C?style=flat-square&logo=python&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/github/stars/gokulsivas/ML_stock_forecasting?style=flat-square&color=3b82f6" />
  <img src="https://img.shields.io/github/last-commit/gokulsivas/ML_stock_forecasting?style=flat-square&color=10b981" />
  <img src="https://img.shields.io/github/repo-size/gokulsivas/ML_stock_forecasting?style=flat-square&color=f59e0b" />
  <img src="https://img.shields.io/badge/license-MIT-8b5cf6?style=flat-square" />
</p>

</div>

---

## Overview

StockCast is a production-ready stock price forecasting web application trained on over 4.2 million rows of NSE historical data. It uses an ensemble deep learning model combining LSTM, GRU, and XGBoost to deliver iterative multi-day predictions for 400+ NIFTY-listed stocks. The platform features a modern dark/light themed React dashboard with real-time prediction charts, multi-stock comparison, and a personalised watchlist.

<br/>

<div align="center">
  <img width="3742" height="2063" alt="image" src="https://github.com/user-attachments/assets/2ffe85e4-31f8-4bab-97c2-eaf3bea72e86" />
  <br/>
</div>

---

## Features

- Iterative price predictions from 1 to 365 days ahead
- Ensemble model: Hybrid LSTM-GRU with XGBoost as a secondary estimator
- Trained on 400+ NIFTY stocks with over 581,000 sequences
- Directional accuracy tracking with returns-based modelling
- Multi-stock side-by-side prediction comparison
- Personal watchlist with 7-day quick forecasts
- CSV and Excel export for all prediction data
- JWT-based authentication with secure user sessions
- Full dark and light theme support
- Incremental database updates via yfinance

---

## Architecture

The diagram below shows the full system architecture, from the React frontend down through the FastAPI layer, the ML model stack, and the PostgreSQL data layer.

<div align="center">
  <img width="1440" height="1906" alt="image" src="https://github.com/user-attachments/assets/4b3d6083-1247-484a-9f5a-a8762c6c288e" />
</div>

---
## Pages

### Predictions

<div align="center">
  <img width="3673" height="2076" alt="image" src="https://github.com/user-attachments/assets/a88d72f4-de5b-4cc2-bdec-743b5bdd4cf0" />
  <br/>
</div>

<br/>

The Predictions page lets you select any NSE stock and generate a forecast up to 365 days into the future using the trained LSTM-GRU ensemble model. Results are displayed on an interactive line or area chart with zoom support, daily return tooltips, and one-click export to CSV or Excel.

<br/>

### Comparison

<div align="center">
  <img width="3723" height="2099" alt="image" src="https://github.com/user-attachments/assets/0af50e57-9cce-4140-a07c-bdfeac9a8837" />
  <img width="3636" height="626" alt="image" src="https://github.com/user-attachments/assets/2cac1c8f-65af-4a6a-9ec5-5cc7e72c29cc" />
  <br/>
</div>

<br/>

The Comparison page allows you to add multiple NSE stocks and view their predicted price trajectories plotted together on a single chart. Each stock is assigned a unique colour, and summary cards below the chart display the expected change percentage alongside a Buy, Hold, or Sell recommendation.

<br/>

### Watchlist

<div align="center">
  <img width="3746" height="1591" alt="image" src="https://github.com/user-attachments/assets/e7fac698-0973-40a6-b4c6-027e0d27e2f9" />
  <br/>
</div>

<br/>

The Watchlist page lets you save your favourite stocks and fetch their 7-day predictions in a single click. Each stock card shows the current price, predicted price, expected change, and a recommendation badge, giving you a quick glance at your entire tracked portfolio.

---

## Model Details

The prediction engine uses an iterative forecasting approach. Rather than predicting prices directly, the model predicts **log returns** for each time step, which are then compounded from the last known closing price to generate the full forecast horizon.

| Property | Value |
|---|---|
| Architecture | Hybrid LSTM-GRU + XGBoost |
| Training stocks | 400+ NIFTY stocks |
| Training sequences | 581,000+ |
| Sequence length | 60 trading days |
| Features | Close, Volume, RSI, MACD, Bollinger Bands, lag returns, volume spikes |
| Directional accuracy | ~54-58% |
| Max forecast horizon | 365 days |

---


## Authentication

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img width="1935" height="2073" alt="image" src="https://github.com/user-attachments/assets/65bb61c6-c7c8-49a0-ab99-1bede6ee17e0" />
        <br/>
      </td>
      <td align="center" width="50%">
        <img width="1939" height="2083" alt="image" src="https://github.com/user-attachments/assets/d543700d-1ae1-4fde-aad6-014b03046d47" />
        <br/>
      </td>
    </tr>
  </table>
</div>

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, LSTM, GRU, XGBoost |
| Backend | FastAPI, Python 3.11, JWT Auth |
| Database | PostgreSQL 15 (4.2M+ rows) |
| Data Pipeline | yfinance, pandas, scikit-learn |
| Frontend | React 18, Vite 5, Tailwind CSS 3 |
| Charts | Recharts |
| Export | SheetJS (xlsx), CSV Blob API |

---

## Project Structure
```
ML_stock_forecasting/
│
├── backend/
│ ├── main.py # FastAPI app and route definitions
│ ├── models.py # SQLAlchemy database models
│ ├── schemas.py # Pydantic request/response schemas
│ ├── predict.py # Iterative prediction engine
│ ├── train.py # LSTM-GRU model training script
│ └── database.py # PostgreSQL connection setup
│
├── frontend/
│ ├── src/
│ │ ├── pages/
│ │ │ ├── Predictions.jsx
│ │ │ ├── Comparison.jsx
│ │ │ └── Watchlist.jsx
│ │ ├── components/
│ │ │ ├── Navbar.jsx
│ │ │ └── LoadingSpinner.jsx
│ │ └── context/
│ │ ├── ThemeContext.jsx
│ │ └── AuthContext.jsx
│ └── vite.config.js
│
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- PostgreSQL 15
- CUDA-capable GPU (recommended for training)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/gokulsivas/ML_stock_forecasting.git
cd ML_stock_forecasting/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in DATABASE_URL, SECRET_KEY in .env

# Run the FastAPI server
uvicorn main:app --reload --port 8001
```

### Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Set VITE_API_URL=http://localhost:8001

# Start the development server
npm run dev
```

---

<div align="center">
  Built by <a href="https://github.com/gokulsivas">Gokul Sivasubramaniam</a>
</div>
