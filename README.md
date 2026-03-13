<p align="center">
  <h1 align="center">🧠 Finara ML Engine</h1>
  <p align="center"><strong>AI/ML Microservice for Explainable Financial Risk Assessment & Debt Optimization</strong></p>
</p>

---

## 📖 About Finara

**Finara** (_Financial Narrative_) is an explainable, probabilistic financial decision-support system built for students and young professionals. It integrates intelligent expense tracking with risk-aware debt optimization under cashflow uncertainty.

### ✨ Key Features

- **🎯 Explainable Risk Assessment (XAI)** — Gradient Boosting + SHAP values to predict default probability with per-feature explanations
- **📄 Smart Document Parsing** — Gemini 2.5 Flash multimodal extraction from receipts, e-statements, and BNPL bills
- **⚡ Debt Optimization** — AHP-weighted multi-criteria debt ranking with Monte Carlo cashflow simulation
- **📊 Strategy Comparison** — Side-by-side comparison of Snowball, Avalanche, and Finara AI strategies
- **🔐 Privacy-First** — Files processed in-memory, never permanently stored

---

## 🏗️ Architecture

This is the **ML Engine** component of the Finara ecosystem:

```
finara/
├── finara-ml/          ← You are here (Python/FastAPI)
├── finara-supabase/    # Database, Auth, Edge Functions
└── finara-web/         # Next.js Frontend
```

The ML engine exposes a FastAPI REST API consumed by `finara-web` (frontend) and `finara-supabase` (edge functions).

---

## 📡 API Modules

| Module | Endpoint | Method | Description |
|---|---|---|---|
| **XAI** | `POST /api/xai/explain-risk` | Gradient Boosting + SHAP + AHP | Explainable risk scoring & debt prioritization |
| **Parser** | `POST /api/parser/parse-document` | Gemini 2.5 Flash | Extract transactions from receipts & statements |
| **Optimizer** | `POST /api/optimizer/recommend` | Deterministic + Monte Carlo | Repayment strategy with scenario simulation |
| **Profiler** | `POST /api/profiler/analyze` | K-Means Clustering | Behavioral spending profiler _(Phase 5)_ |
| **Health** | `GET /health` | — | Service health check |

> All endpoints (except `/health`) require an API key via the `X-API-Key` header.

---

## 📂 Project Structure

```
finara-ml/
├── app/
│   ├── main.py                     # FastAPI app, CORS, API key auth
│   ├── schemas.py                  # Pydantic request/response models
│   ├── routers/
│   │   ├── xai.py                  # Explainable risk assessment endpoint
│   │   ├── optimizer.py            # Repayment optimization endpoint
│   │   ├── parser.py               # Document parsing endpoint
│   │   └── profiler.py             # Behavioral profiling endpoint
│   └── services/
│       ├── risk_model.py           # GradientBoosting training pipeline
│       ├── shap_explainer.py       # SHAP inference wrapper
│       ├── feature_engineering.py  # Feature extraction & AHP scoring
│       ├── document_parser.py      # Gemini multimodal extraction
│       ├── debt_environment.py     # RL environment (Gymnasium)
│       ├── rl_optimizer.py         # RL optimizer (stable-baselines3)
│       ├── spending_profiler.py    # K-Means behavioral clustering
│       └── mock_data_generator.py  # Synthetic training data (5000 samples)
├── trained_models/                 # Serialized .pkl model files
├── data/                           # Training CSV data
├── test_endpoints.py               # API endpoint tests
├── Dockerfile                      # Container configuration
├── railway.toml                    # Railway deployment config
├── run_local.sh                    # Local development script
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12** | Runtime |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **scikit-learn** | Gradient Boosting classifier for PD modeling |
| **SHAP** | Model explainability (TreeExplainer) |
| **NumPy / Pandas** | Data processing and feature engineering |
| **Google GenAI (Gemini 2.5 Flash)** | Multimodal document parsing |
| **Pillow** | Image processing for receipt OCR |
| **PyPDF2** | PDF text extraction |
| **stable-baselines3 / Gymnasium** | RL infrastructure |
| **Docker** | Containerization |
| **Railway** | Cloud deployment |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **Docker** (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd finara-ml

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
ML_API_KEY=your_secure_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

| Variable | Description |
|---|---|
| `ML_API_KEY` | API key for authenticating incoming requests |
| `GEMINI_API_KEY` | Google Gemini API key for document parsing |

### Train the ML Model

Before starting the server, generate training data and train the risk model:

```bash
# Generate synthetic training data (5000 samples)
python -m app.services.mock_data_generator

# Train the Gradient Boosting model
python -m app.services.risk_model
```

This will create model files in the `trained_models/` directory.

### Run the Server

```bash
# Option 1: Direct start
uvicorn app.main:app --reload --port 8000

# Option 2: Use the convenience script
chmod +x run_local.sh
./run_local.sh
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger documentation.

### Docker Deployment

```bash
# Build the image
docker build -t finara-ml .

# Run the container
docker run -p 8000:8000 --env-file .env finara-ml
```

### Railway Deployment

The project includes a `railway.toml` for one-click deployment on [Railway](https://railway.app):

```bash
railway up
```

Make sure to set `ML_API_KEY` and `GEMINI_API_KEY` as environment variables in your Railway project settings.

---

## 🔗 Integration with Other Finara Services

### ↔️ Integration with `finara-web`

The frontend calls the ML engine directly via Axios. Configure the following environment variables in `finara-web/.env`:

```env
NEXT_PUBLIC_ML_API_URL=http://localhost:8000    # ML engine URL
NEXT_PUBLIC_ML_API_KEY=your_secure_api_key      # Must match ML_API_KEY
```

The web app uses the ML engine for:
- **AI Insights page** → calls `/api/xai/explain-risk` for risk scoring & SHAP explanations
- **Smart Debt Optimizer** → calls `/api/optimizer/recommend` for repayment strategies
- **Transaction import** → calls `/api/parser/parse-document` for document extraction

### ↔️ Integration with `finara-supabase`

Supabase Edge Functions can call the ML engine as middleware. The ML engine's CORS is pre-configured to accept requests from both `localhost:3000` (dev) and your Supabase project URL.

---

## 🧪 Testing

Run the endpoint tests:

```bash
python test_endpoints.py
```

---

<p align="center">
  Built with 🧠 ML + 💡 Explainable AI + 🎯 Decision Science
</p>
