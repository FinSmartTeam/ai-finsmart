"""
FinSmart AI Service — FastAPI
Coding Camp 2026 | CC26-PSU407 | AI Engineer: Muhammad Syaiful

Jalankan: python -m uvicorn main:app --port 8000
Docs    : http://localhost:8000/docs
"""

import numpy as np
import pickle
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# INISIALISASI APP
# ============================================================
app = FastAPI(
    title="FinSmart AI Service",
    description="API klasifikasi kategori pengeluaran berbasis AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
model = joblib.load("xgb_model.pkl")

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_metadata.json") as f:
    metadata = json.load(f)

FITUR_KAT = [
    "PaymentMethod", "Location", "AccountType", "TransactionType",
    "DeviceUsed", "MerchantType", "LoyaltyProgram", "Weekday",
    "Month", "TimeOfDay", "MerchantName"
]

# ============================================================
# SCHEMA REQUEST
# ============================================================
class TransaksiInput(BaseModel):
    Amount: float
    PaymentMethod: str
    Location: str
    AccountType: str
    TransactionType: str
    DeviceUsed: str
    MerchantType: str
    LoyaltyProgram: str
    Weekday: str
    Month: str
    TimeOfDay: str
    MerchantName: str

    class Config:
        json_schema_extra = {
            "example": {
                "Amount": 250000,
                "PaymentMethod": "Credit Card",
                "Location": "Mumbai",
                "AccountType": "Savings",
                "TransactionType": "Debit",
                "DeviceUsed": "Mobile",
                "MerchantType": "Online Store",
                "LoyaltyProgram": "Yes",
                "Weekday": "Saturday",
                "Month": "May",
                "TimeOfDay": "Evening",
                "MerchantName": "Amazon"
            }
        }

# ============================================================
# HELPER — Preprocessing
# ============================================================
def preprocess(data: TransaksiInput) -> np.ndarray:
    row = {}
    d = data.dict()
    for col in FITUR_KAT:
        le = encoders[col]
        if d[col] not in le.classes_:
            raise HTTPException(
                status_code=422,
                detail=f"Nilai tidak valid untuk '{col}': '{d[col]}'. "
                       f"Nilai valid: {list(le.classes_)}"
            )
        row[col] = le.transform([d[col]])[0]
    row["Amount"] = scaler.transform([[d["Amount"]]])[0][0]
    return np.array([[row[f] for f in FITUR_KAT + ["Amount"]]])

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status" : "ok",
        "service": "FinSmart AI Service",
        "version": "2.0.0",
        "docs"   : "/docs"
    }


@app.post("/classify", tags=["Klasifikasi"])
def classify(transaksi: TransaksiInput):
    """
    Klasifikasi kategori pengeluaran dari data transaksi.

    Returns:
    - kategori: prediksi kategori pengeluaran
    - confidence: tingkat keyakinan model (%)
    - top3_prob: probabilitas 3 kategori teratas
    """
    X     = preprocess(transaksi)
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    kategori = encoders["Category"].inverse_transform([idx])[0]
    semua    = encoders["Category"].classes_
    top3     = dict(sorted(
        {k: round(float(p), 4) for k, p in zip(semua, proba)}.items(),
        key=lambda x: x[1], reverse=True
    )[:3])
    return {
        "kategori"  : kategori,
        "confidence": round(float(proba[idx]) * 100, 2),
        "top3_prob" : top3
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    """Informasi model yang sedang berjalan."""
    return metadata


@app.get("/valid-values", tags=["Info"])
def valid_values():
    """Daftar nilai valid untuk setiap fitur input."""
    return {col: list(encoders[col].classes_) for col in FITUR_KAT}


# ============================================================
# ENTRY POINT — Proteksi Windows Multiprocessing
# ============================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # ← fix Windows multiprocessing error

    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False   # ← reload=False untuk stabilitas di Windows
    )
