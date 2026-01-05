import mlflow
from fastapi import FastAPI, Request
from mlflow.tracking import MlflowClient
import joblib
import numpy as np
import pandas as pd
import time
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import pickle

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("model_api")


# -------------------------
# Load model at startup
# -------------------------
model = joblib.load("logRegModel.pkl")


# ---------------- App ----------------
app = FastAPI(title="Health Disease Prediction API")


# ---------------- Middleware ----------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.2f}ms"
    )
    return response

# ---------------- Metrics ----------------
Instrumentator().instrument(app).expose(app)



# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"
MODEL_NAME = "LogisticRegressionModel_V2"
MODEL_VERSION = 1
MODEL_STAGE = "Production"   # or Staging
DEST_PATH = "/tmp/downloaded_model"


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/loadModel")
def loadModel():

    # -----------------------------
    # Connect to MLflow
    # -----------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # -----------------------------
    # Get model version in stage
    # -----------------------------
    model_versions = client.get_latest_versions(
        name=MODEL_NAME
    )

    if not model_versions:
        raise Exception(f"No model found in stage {MODEL_STAGE}")

    model_version = model_versions[0]
    run_id = model_version.run_id
    artifact_path = model_version.source

    print(f"Downloading model version {model_version.version}")

    # -----------------------------
    # Download artifacts
    # -----------------------------
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_path,
        dst_path=DEST_PATH
    )

    print(f"Model downloaded to: {local_path}")

def loadModelFromLocalDirectory():
    return joblib.load(DEST_PATH + "/model.pkl")

@app.post("/predict")
def predict(data: dict):
    loaded_model = loadModelFromLocalDirectory()

    X = pd.DataFrame([data])
    pred = loaded_model.predict(X)[0]
    proba = loaded_model.predict_proba(X)[0]

    return {
        "prediction": int(pred),
        "confidence": float(max(proba))
    }


@app.post("/v2/predict")
def predict(data: dict):
    X = pd.DataFrame([data])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "prediction": int(pred),
        "confidence": float(max(proba))
    }