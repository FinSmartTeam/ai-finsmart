"""
FinSmart AI Service — FastAPI LENGKAP
Coding Camp 2026 | CC26-PSU407 | AI Engineer: Muhammad Syaiful

Endpoints:
  GET  /              — health check
  POST /classify      — klasifikasi kategori pengeluaran
  POST /finbot/chat   — chatbot keuangan FinBot
  POST /rekomendasi   — rekomendasi anggaran personal
  GET  /model-info    — info model
  GET  /valid-values  — nilai valid tiap fitur
  GET  /docs          — Swagger UI

Jalankan: uvicorn main_complete:app --reload --port 8000
"""

import numpy as np
import pickle
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from tensorflow import keras
from transformers import pipeline as hf_pipeline

# INISIALISASI APP
# ============================================================
app = FastAPI(
    title="FinSmart AI Service",
    description="API lengkap untuk klasifikasi pengeluaran, FinBot, dan rekomendasi anggaran",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# LOAD SEMUA MODEL & ARTIFACTS
# ============================================================
print("Loading model klasifikasi...")
model = keras.models.load_model("finsmart_model.keras")
with open("encoders.pkl", "rb") as f: encoders = pickle.load(f)
with open("scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
with open("model_metadata.json") as f: metadata = json.load(f)

print("Loading FinBot (Hugging Face)...")
finbot_clf = hf_pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
print("Semua model berhasil diload!")

FITUR_KAT = [
    "PaymentMethod", "Location", "AccountType",
    "TransactionType", "DeviceUsed", "MerchantType",
    "LoyaltyProgram", "Weekday", "Month", "TimeOfDay"
]

INTENT_LABELS = [
    "tips menabung", "cara mengatur anggaran", "kategori pengeluaran",
    "investasi untuk pemula", "cara mengurangi pengeluaran",
    "target tabungan", "rekomendasi keuangan", "pengertian literasi keuangan",
]

RESPONSE_DB = {
    "tips menabung": "Terapkan metode 50/30/20: 50% kebutuhan, 30% keinginan, 20% tabungan. Simpan di awal bulan sebelum digunakan!",
    "cara mengatur anggaran": "Catat semua pemasukan, bagi ke kategori, tentukan batas, dan pantau setiap hari. Gunakan fitur Anggaran di FinSmart!",
    "kategori pengeluaran": "FinSmart mengklasifikasikan ke 10 kategori: Food, Grocery, Transport, Entertainment, Healthcare, Electronics, Online Shopping, Travel, Clothing, Bills.",
    "investasi untuk pemula": "Pastikan punya dana darurat dulu. Lalu mulai dari Reksa Dana Pasar Uang atau Deposito. Cek fitur Kesiapan Investasi di FinSmart!",
    "cara mengurangi pengeluaran": "Identifikasi kategori terbesar dari dashboard, bedakan kebutuhan vs keinginan, dan pantau tren pengeluaranmu!",
    "target tabungan": "Tentukan tujuan spesifik, hitung nominal & deadline, bagi per bulan, lalu set target di FinSmart. Kami akan notifikasi progresmu!",
    "rekomendasi keuangan": "Lihat menu Analisis & Rekomendasi AI di FinSmart untuk insight personal berdasarkan data keuanganmu.",
    "pengertian literasi keuangan": "Kemampuan memahami & menggunakan konsep keuangan: mengelola income, menabung, investasi, dan membuat keputusan finansial yang bijak.",
}
RESPONSE_DEFAULT = "Maaf, saya belum bisa menjawab itu. Coba tanya tentang tips menabung, anggaran, investasi, atau kategori pengeluaran."

BOBOT_IDEAL = {
    "Food": 0.15, "Grocery": 0.10, "Transport": 0.10, "Bills": 0.10,
    "Healthcare": 0.05, "Entertainment": 0.05, "Online Shopping": 0.05,
    "Clothing": 0.05, "Electronics": 0.05, "Travel": 0.05,
}

INSTRUMEN = {
    "konservatif": ["Tabungan berjangka / Deposito", "Obligasi Negara (ORI/SBR)", "Reksa Dana Pasar Uang"],
    "moderat"    : ["Reksa Dana Pendapatan Tetap", "ORI/SBR", "P2P Lending OJK"],
    "agresif"    : ["Reksa Dana Saham", "Saham blue-chip (LQ45)", "ETF / Index Fund"],
}

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

class ChatInput(BaseModel):
    pertanyaan: str

class RekomendasiInput(BaseModel):
    income_bulanan: float
    tabungan_bulanan: float
    usia: int = 25
    pengeluaran: dict  # {kategori: jumlah}

# ENDPOINTS
# ============================================================
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "FinSmart AI Service", "version": "1.0.0"}

@app.post("/classify", tags=["Klasifikasi"])
def classify(transaksi: TransaksiInput):
    data = transaksi.dict()
    row = {}
    for col in FITUR_KAT:
        le = encoders[col]
        if data[col] not in le.classes_:
            raise HTTPException(422, f"Nilai tidak valid: {col}={data[col]}")
        row[col] = le.transform([data[col]])[0]
    row["Amount"] = scaler.transform([[data["Amount"]]])[0][0]
    X = np.array([[row[f] for f in FITUR_KAT + ["Amount"]]])
    proba = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(proba))
    kategori = encoders["Category"].inverse_transform([idx])[0]
    semua = encoders["Category"].classes_
    top3 = dict(sorted(
        {k: round(float(p), 4) for k, p in zip(semua, proba)}.items(),
        key=lambda x: x[1], reverse=True
    )[:3])
    return {"kategori": kategori, "confidence": round(float(proba[idx])*100, 2), "top3_prob": top3}

@app.post("/finbot/chat", tags=["FinBot"])
def finbot_chat(body: ChatInput):
    if not body.pertanyaan.strip():
        raise HTTPException(422, "Pertanyaan tidak boleh kosong.")
    result = finbot_clf(body.pertanyaan, candidate_labels=INTENT_LABELS,
                        hypothesis_template="Pertanyaan ini tentang {}.")
    intent = result["labels"][0]
    confidence = result["scores"][0]
    respons = RESPONSE_DB.get(intent, RESPONSE_DEFAULT) if confidence >= 0.30 else RESPONSE_DEFAULT
    return {"intent": intent, "confidence": round(confidence*100, 2), "respons": respons}

@app.post("/rekomendasi", tags=["Rekomendasi"])
def rekomendasi_anggaran(body: RekomendasiInput):
    income = body.income_bulanan
    anggaran = {}
    for kat, bobot in BOBOT_IDEAL.items():
        ideal  = income * bobot
        aktual = body.pengeluaran.get(kat, 0)
        anggaran[kat] = {
            "batas_ideal": round(ideal),
            "aktual": round(aktual),
            "selisih": round(aktual - ideal),
            "status": "Over" if aktual > ideal else "Aman",
        }
    rasio_tab = body.tabungan_bulanan / income if income > 0 else 0
    total_exp = sum(body.pengeluaran.values())
    skor = min(40, int(rasio_tab/0.20*40))
    sisa = max(0, (income - total_exp - body.tabungan_bulanan)/income)
    skor += min(30, int(sisa*100))
    ratio_exp = total_exp/income if income>0 else 1
    skor += 20 if ratio_exp<=0.50 else (10 if ratio_exp<=0.70 else 0)
    skor += 10 if body.usia<=25 else (7 if body.usia<=35 else 5)
    profil = "agresif" if skor>=85 else ("moderat" if skor>=70 else "konservatif")
    label  = ("Siap Investasi" if skor>=70 else
               "Hampir Siap" if skor>=45 else "Belum Siap")
    return {
        "rekomendasi_anggaran": anggaran,
        "kesiapan_investasi": {"skor": skor, "label": label, "profil_risiko": profil},
        "rekomendasi_instrumen": INSTRUMEN.get(profil, []),
        "over_budget": [k for k, v in anggaran.items() if v["status"]=="Over"],
    }

@app.get("/model-info", tags=["Info"])
def model_info(): return metadata

@app.get("/valid-values", tags=["Info"])
def valid_values():
    return {col: list(encoders[col].classes_) for col in FITUR_KAT}