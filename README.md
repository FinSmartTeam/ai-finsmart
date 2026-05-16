# FinSmart — AI Service

**Coding Camp 2026 | CC26-PSU407 | Tema: Revolusi Teknologi Keuangan (Fintech) untuk Generasi Muda**

AI Engineer: Muhammad Syaiful

---

## Deskripsi Proyek

FinSmart adalah aplikasi pengelolaan keuangan pribadi berbasis AI untuk generasi muda. Repository ini berisi seluruh komponen AI yang dibangun oleh AI Engineer, meliputi model klasifikasi pengeluaran, chatbot keuangan (FinBot), sistem rekomendasi kesiapan investasi, dan klasifikasi perilaku keuangan.

---

## Checklist Capstone

### Main Quest
| No | Komponen | Status |
|----|----------|--------|
| 1 | Model Deep Learning TensorFlow Functional API | Selesai |
| 2 | Custom Callback (FinSmartCallback) | Selesai |
| 3 | Simpan model format `.keras` | Selesai |
| 4 | Kode Inference (FinSmartInference class) | Selesai |

### Side Quest
| No | Komponen | Status |
|----|----------|--------|
| 1 | REST API FastAPI | Selesai |
| 2 | Akurasi >= 85% | Selesai (XGBoost 99.83%) |

---

## Arsitektur Pipeline AI

```
Input Transaksi (Front-End)
        |
        v
Back-End Node.js
        |
        v
AI Service FastAPI (Repository ini)
        |
        |--- POST /classify    --> XGBoost (Klasifikasi Kategori)
        |--- POST /behavior    --> Decision Tree (Perilaku Keuangan)
        |--- POST /rekomendasi --> Rule-Based BLR + SR (Kesiapan Investasi)
        |--- POST /finbot/chat --> Hugging Face BART (Chatbot)
        |
        v
Response JSON ke Front-End
```

---

## Model yang Digunakan

### 1. Klasifikasi Kategori Pengeluaran
- **Model utama**: XGBoost (`xgb_model.pkl`)
- **Model pendukung**: TensorFlow Functional API (`finsmart_model.keras`)
- **Input**: Amount, Payment_Method, Week_Day, Month, Time_Of_Day, MerchantName, Day
- **Output**: 10 kategori pengeluaran
- **Akurasi XGBoost**: 99.83%
- **Custom Callback**: FinSmartCallback

### 2. Klasifikasi Perilaku Keuangan
- **Model**: Decision Tree (`financial_type_classifier.pkl`) dari tim Data Scientist
- **Input**: Income, Needs, Wants, Savings, Total_Spending, Financial_Balance
- **Output**: hemat / normal / boros
- **Acuan**: Prinsip 50/30/20

### 3. FinBot — Chatbot Keuangan
- **Model**: Hugging Face `facebook/bart-large-mnli`
- **Teknik**: Zero-Shot Classification
- **Input**: Pertanyaan keuangan bahasa Indonesia
- **Output**: Respons edukasi keuangan (8 intent)

### 4. Rekomendasi Kesiapan Investasi
- **Metode**: Rule-Based (BLR + SR)
- **BLR** = Tabungan Total / Total Pengeluaran Bulanan
- **SR** = Tabungan Bulanan / Income Bulanan x 100%
- **Output**: Not Ready / Moderately Ready / Ready

---

## Kategori Pengeluaran (10 Kelas)

| No | Kategori |
|----|----------|
| 1 | Bills |
| 2 | Clothing |
| 3 | Electronics |
| 4 | Entertainment |
| 5 | Food |
| 6 | Grocery |
| 7 | Healthcare |
| 8 | Online Shopping |
| 9 | Transport |
| 10 | Travel |

---

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Health check |
| POST | `/classify` | Klasifikasi kategori pengeluaran |
| POST | `/behavior` | Klasifikasi perilaku keuangan |
| POST | `/rekomendasi` | Analisis kesiapan investasi |
| POST | `/finbot/chat` | Chatbot keuangan FinBot |
| GET | `/model-info` | Informasi model |
| GET | `/valid-values` | Nilai valid tiap fitur |
| GET | `/docs` | Swagger UI |

---

## Struktur File

```
ai-finsmart/
|
|-- FinSmart_AI_Engineer_UPDATED_v2.ipynb       # Notebook utama (Main Quest)
|-- FinSmart_FinBot_Rekomendasi_UPDATED_v2.ipynb # Notebook FinBot & Rekomendasi
|
|-- main.py                                      # FastAPI basic
|-- main_complete.py                             # FastAPI lengkap (v2.0.0)
|
|-- finsmart_model.keras                         # Model TF Functional API
|-- xgb_model.pkl                                # Model XGBoost final
|-- financial_type_classifier.pkl                # Model behavior (dari DS)
|-- encoders.pkl                                 # Label encoders
|-- scaler.pkl                                   # StandardScaler
|
|-- model_metadata.json                          # Metadata & performa model
|-- training_log.json                            # Log training per epoch
|-- arsitektur_ai.json                           # Dokumentasi arsitektur
|-- finbot_chat_history.json                     # Bukti FinBot berjalan
|-- contoh_rekomendasi.json                      # Bukti rekomendasi berjalan
|
|-- training_history.png                         # Grafik accuracy & loss
|-- confusion_matrix.png                         # Confusion matrix XGBoost
|
|-- requirements.txt                             # Library dependencies
```

---

## Cara Menjalankan API Lokal

```bash
pip install -r requirements.txt
uvicorn main_complete:app --host 0.0.0.0 --port 8000 --reload
```

Akses Swagger UI: `http://localhost:8000/docs`

---

## Contoh Request

### POST /classify
```json
{
  "Amount": 250000,
  "Payment_Method": "Credit Card",
  "Week_Day": "Saturday",
  "Month": "May",
  "Time_Of_Day": "Evening",
  "MerchantName": "Amazon",
  "Day": 10
}
```

### POST /behavior
```json
{
  "Income": 5000000,
  "Needs": 2500000,
  "Wants": 1500000,
  "Savings": 1000000,
  "Total_Spending": 4000000,
  "Financial_Balance": 1000000
}
```

### POST /rekomendasi
```json
{
  "tabungan_total": 15000000,
  "total_pengeluaran_bulanan": 3500000,
  "tabungan_bulanan": 1000000,
  "income_bulanan": 5000000
}
```

### POST /finbot/chat
```json
{
  "pertanyaan": "Bagaimana cara menabung lebih banyak setiap bulan?"
}
```

---

## Tim CC26-PSU407

| Nama | Peran |
|------|-------|
| Zahwa Putri Vanisa | Data Scientist |
| A. Rafly Sahrul | Full-Stack Web Developer (Back-End) |
| Ahmad Anta Wirangga | Full-Stack Web Developer (Front-End) |
| Fahira Zahra Fitria Rahma | Data Scientist |
| Muhammad Syaiful | AI Engineer |

---

## Repository Terkait

- **AI Service**: https://github.com/FinSmartTeam/ai-finsmart
- **Back-End**: https://github.com/FinSmartTeam/backend-FinSmart
- **Back-End URL**: https://backend-fin-smart.vercel.app
