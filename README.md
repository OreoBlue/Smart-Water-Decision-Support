# SmartWater Decision Support

**Dashboard Forecasting Kebutuhan Air, Analisis Tekanan Demand–Supply, dan Simulasi Dampak Data Center untuk Smart City**

SmartWater Decision Support merupakan project **Data Mining Competition ICONFEST 2026** dengan tema **Smart City**. Project ini dikembangkan sebagai prototype sistem pendukung keputusan untuk membantu pengelola wilayah memahami kebutuhan air di masa mendatang, membandingkannya dengan estimasi pasokan, serta mensimulasikan dampak tambahan kebutuhan air dari infrastruktur digital seperti **data center**.

---

## Anggota Tim

Dikembangkan oleh:

Jahe Bakar

- **Juan Stevan Immanuel** — Ketua Tim
- **Naail Safwan Saleh** — Anggota
- **Haidar Yahya** — Anggota


---

## Latar Belakang

Ketersediaan air menjadi salah satu isu penting dalam pembangunan kota yang berkelanjutan di Indonesia. Pertumbuhan penduduk, urbanisasi, perubahan pola iklim, serta perkembangan aktivitas ekonomi dan infrastruktur dapat meningkatkan tekanan terhadap sumber daya air.

Di sisi lain, perkembangan ekonomi digital juga mendorong pembangunan infrastruktur seperti **data center**. Selain membutuhkan energi listrik dalam jumlah besar, data center juga dapat menggunakan air untuk mendukung proses pendinginan. Kondisi ini menimbulkan kebutuhan baru untuk memahami bagaimana tambahan konsumsi air dapat memengaruhi keseimbangan antara permintaan dan pasokan air pada suatu wilayah.

Project ini dikembangkan untuk menjawab kebutuhan tersebut melalui pendekatan berbasis data. Sistem tidak hanya menghasilkan prakiraan kebutuhan air, tetapi juga menghubungkan hasil forecasting dengan estimasi pasokan, tingkat tekanan demand–supply, serta simulasi skenario data center.

Karena data publik Indonesia yang menyediakan **demand, supply, populasi, dan indikator shortage secara konsisten dalam periode historis panjang** masih terbatas, project ini menggunakan **California urban water-supplier data** sebagai **proxy dan benchmark metodologi**. California tidak diposisikan sebagai representasi kondisi Indonesia, tetapi sebagai contoh data untuk menguji apakah kerangka analitik yang dibangun dapat berjalan sebelum nantinya diterapkan menggunakan data lokal Indonesia yang lebih lengkap.

---

## Tujuan Project

SmartWater Decision Support dikembangkan untuk:

1. Memprakirakan kebutuhan air residensial hingga **12 bulan ke depan**.
2. Mengestimasi pasokan air per kapita sebagai pembanding terhadap kebutuhan.
3. Mengukur **tekanan demand–supply** sebagai indikator analitik.
4. Mengklasifikasikan kondisi menjadi **Low, Watch, dan High**.
5. Menampilkan ketidakpastian hasil forecasting melalui **prediction interval 80%**.
6. Mensimulasikan tambahan konsumsi air akibat operasional data center.
7. Membandingkan kondisi **baseline** dengan berbagai skenario pembangunan.
8. Menyediakan informasi yang lebih mudah dipahami untuk mendukung pengambilan keputusan berbasis data.

---

## Konsep Utama

Versi project saat ini memisahkan empat komponen utama:

### 1. Demand Forecasting
Model memprediksi **Residential Gallons per Capita per Day (R-GPCD)** untuk horizon 1–12 bulan.

### 2. Supply-Side Forecasting
Sistem memperkirakan **potable water supply per capita per day** menggunakan metode yang dipilih berdasarkan hasil evaluasi performa.

### 3. Demand–Supply Pressure Proxy
Tekanan dihitung dengan:

```text
Pressure = Forecast Residential Demand / Forecast Potable Supply
```

Nilai pressure kemudian dipetakan ke tiga status:

- **Low**
- **Watch**
- **High**

Threshold dibentuk dari distribusi empiris historis menggunakan kuantil, bukan dari ambang regulasi resmi.

### 4. Administrative Shortage Classification
Project juga menyediakan **DWR shortage signal** sebagai sinyal administratif tambahan. Sinyal ini tidak diperlakukan sebagai representasi langsung dari physical water availability karena label DWR sangat persisten dari waktu ke waktu.

---

## Fitur Utama Dashboard

### Water Demand Forecasting
- Forecast kebutuhan air residensial hingga 12 bulan.
- Visualisasi histori 36 bulan terakhir.
- Menampilkan forecast baseline.
- Menampilkan **prediction interval 80%**.
- Membandingkan model dengan seasonal-naive baseline.

### Water Supply Forecasting
- Estimasi potable supply per kapita.
- Digunakan sebagai pembanding terhadap forecast demand.
- Metode dipilih berdasarkan hasil validasi, bukan semata-mata karena kompleksitas algoritma.

### Demand–Supply Pressure Indicator
- Mengukur rasio demand terhadap supply.
- Menampilkan kondisi **Low / Watch / High**.
- Membandingkan baseline dengan skenario data center.
- Menampilkan batas empiris **Q60** dan **Q85**.

### Data Center Scenario Simulator
Pengguna dapat mengubah beberapa parameter:

- **Beban IT (MW)** — kapasitas komputasi data center.
- **Utilisasi rata-rata** — proporsi kapasitas IT yang digunakan.
- **Suhu udara (°C)** — kondisi temperatur lingkungan.
- **Kelembapan relatif (%)** — bersama suhu digunakan untuk menghitung kondisi wet-bulb.
- **Model pendinginan** — `Fixed approach` atau `Fixed cold water`.
- **Pengali efisiensi pendinginan (λ)** — nilai `λ < 1` menunjukkan skenario lebih hemat air, sedangkan `λ > 1` menunjukkan konsumsi lebih tinggi dari baseline skenario.

### Scenario Comparison
Dashboard membandingkan:

```text
Baseline
vs.
Data Center Scenario
```

untuk melihat perubahan pada:
- kebutuhan air;
- demand–supply pressure;
- status pressure;
- periode pertama kondisi High muncul.

### Executive Summary
Dashboard secara otomatis merangkum:
- tekanan puncak;
- bulan terjadinya tekanan puncak;
- perubahan tekanan akibat skenario;
- periode pertama status High muncul.

### DWR Shortage Signal
Sistem menampilkan:
- kelas shortage administratif;
- probabilitas DWR high-risk.

Fitur ini hanya digunakan sebagai **secondary signal**.

### Export Data
Seluruh output forecasting dan simulasi dapat diunduh dalam format **CSV** untuk analisis lanjutan, dokumentasi, laporan, dan pengembangan model berikutnya.

---

## Arsitektur Sistem

```text
Historical Water Data
        │
        ▼
Data Preprocessing
        │
        ├───────────────┐
        ▼               ▼
Demand Forecast     Supply Forecast
  1–12 Bulan        1–12 Bulan
        │               │
        └───────┬───────┘
                ▼
     Demand–Supply Pressure
                │
        ┌───────┴────────┐
        ▼                ▼
Pressure Status      DWR Signal
Low / Watch / High   Administrative
        │
        ▼
Data Center Scenario Simulation
        │
        ▼
Smart Water Decision Dashboard
```

---

## Dataset

Project menggunakan data historis pemasok air perkotaan California sebagai benchmark metodologi.

Dataset mencakup informasi seperti:
- residential GPCD;
- potable water supply;
- populasi;
- supplier / wilayah;
- hydro region;
- climate zone;
- shortage stage administratif.

Dataset California digunakan karena menyediakan struktur historis yang relatif konsisten untuk pengujian forecasting dan klasifikasi.

> **Catatan:** project ini tidak mengklaim bahwa model saat ini mewakili kondisi seluruh Indonesia atau bersifat global. Target pengembangan selanjutnya adalah menerapkan framework yang sama pada dataset lokal Indonesia ketika data demand–supply dengan resolusi temporal yang memadai tersedia.

---

## Metodologi

### Preprocessing
Tahapan preprocessing mencakup:
- pembersihan data;
- normalisasi tipe data;
- filtering nilai ekstrem;
- pembentukan time index;
- transformasi populasi;
- penyusunan lag dan rolling features;
- pemisahan train dan holdout secara temporal.

### Demand Forecasting
Model demand menggunakan pendekatan tree-based boosting dengan fitur:
- horizon;
- month;
- historical GPCD;
- rolling statistics;
- seasonal reference;
- population;
- climate zone;
- hydro region.

Evaluasi dilakukan menggunakan holdout periode masa depan agar menyerupai kondisi deployment.

### Supply Forecasting
Pada eksperimen supply, model boosting menghasilkan performa lebih rendah dibanding seasonal-naive. Karena itu, supply forecast menggunakan **seasonal-naive 12 bulan**.

Keputusan ini dibuat berdasarkan **validation performance**, bukan karena model sederhana selalu lebih baik.

### Prediction Interval
Uncertainty forecast ditampilkan menggunakan **80% prediction interval** yang dikalibrasi dari residual berdasarkan horizon.

### Pressure Classification
Pressure dihitung dari perbandingan antara predicted residential demand dan predicted potable supply.

Threshold:
- **Watch**: sekitar kuantil empiris Q60
- **High**: sekitar kuantil empiris Q85

Threshold ini merupakan indikator analitik dan **bukan ambang regulasi**.

### DWR Classification
DWR shortage stage dikelompokkan menjadi tiga kelas dan digunakan sebagai sinyal sekunder. Karena label DWR sangat persisten, akurasi klasifikasi tidak digunakan sebagai satu-satunya indikator performa.

---

## Hasil Evaluasi Model

| Metrik | Hasil |
|---|---:|
| Demand Forecast MAPE | **12,69%** |
| Seasonal-Naive Demand MAPE | 13,73% |
| H1 Demand MAPE | **11,93%** |
| H6 Demand MAPE | **12,49%** |
| H12 Demand MAPE | **13,49%** |
| 80% Prediction Interval Coverage | **85,03%** |
| Supply Forecast MAPE | **11,91%** |
| DWR Classification Accuracy | **78,16%** |
| Macro F1 Score | **72,70%** |
| High-Risk Recall | **87,11%** |

### Interpretasi

Demand forecasting mengungguli seasonal-naive baseline secara keseluruhan. Namun, selisih performa semakin kecil pada horizon lebih panjang, terutama H12. Hal ini menunjukkan bahwa penambahan variabel eksternal seperti curah hujan, suhu, drought index, reservoir storage, atau informasi iklim lainnya berpotensi meningkatkan performa forecasting jangka menengah hingga panjang.

Untuk supply, seasonal-naive dipertahankan karena menghasilkan MAPE sekitar **11,91%**, lebih baik dibanding eksperimen model boosting yang menghasilkan MAPE sekitar **13,32%**.

---

## Teknologi yang Digunakan

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Altair**
- **Machine Learning**
- **Time-Series Feature Engineering**
- **Scenario Analysis**

---

## Struktur Repository

```text
smart-water-decision-support/
│
├── app.py
├── core.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed.csv.gz
│
├── models/
│   ├── models.joblib
│   └── metrics.json
│
├── experiments/
│   └── benchmark_notes.md
│
├── docs/
│   ├── DEMO_SCRIPT.md
│   ├── METHODOLOGY.md
│   ├── MODEL_CARD.md
│   └── DATASETS.md
│
└── .streamlit/
    └── config.toml
```

---

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/smart-water-decision-support.git
cd smart-water-decision-support
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Dashboard

```bash
streamlit run app.py
```

### 4. Retraining Model

```bash
python train.py path/to/monthly_combined_dataset.csv
```

---

## Link Project

- **Repository:** [ISI LINK REPOSITORY]
- **Prototype / Deployment:** [ISI LINK PROTOTYPE]
- **Video Demo:** [ISI LINK VIDEO DEMO]

---

## Keterbatasan

1. Forecast demand hanya merepresentasikan **residential water demand**, bukan seluruh total kebutuhan air.
2. Supply forecast belum merepresentasikan stok waduk, groundwater, snowpack, atau keseluruhan physical water availability.
3. Pressure indicator merupakan **proxy analitik**, bukan indeks water stress regulatif.
4. DWR shortage stage merupakan label administratif dan memiliki persistence yang sangat tinggi.
5. Skenario data center merupakan **what-if simulation**, bukan hasil kalibrasi terhadap satu data center tertentu.
6. Dataset utama berasal dari California dan digunakan sebagai benchmark metodologi karena keterbatasan data publik Indonesia yang setara.
7. Model belum memasukkan variabel eksternal seperti curah hujan, temperatur, drought index, reservoir storage, dan groundwater level.

---

## Pengembangan Selanjutnya

- Integrasi data curah hujan.
- Integrasi temperatur dan climate anomaly.
- Integrasi drought index.
- Integrasi reservoir dan groundwater data.
- Spatial modeling.
- Penerapan pada data PDAM / Perumda Air Minum di Indonesia.
- Pengembangan early warning system.
- Scenario comparison untuk beberapa tipe infrastruktur.
- Probabilistic forecasting.
- Explainable AI menggunakan SHAP.

---

## Competition Positioning

Project ini diposisikan sebagai:

> **Framework Smart Water Decision Support yang dapat ditransfer ke berbagai wilayah, dengan benchmark pada data pemasok air perkotaan California, yang menggabungkan demand forecasting, supply benchmarking, uncertainty, demand–supply pressure, dan scenario analysis.**

Project ini **tidak mengklaim** bahwa model saat ini:
- dilatih menggunakan data global;
- secara langsung memprediksi physical reservoir availability; atau
- mewakili kondisi seluruh wilayah Indonesia.

Fokus utama project adalah menunjukkan bagaimana data mining dapat digunakan untuk membangun sistem pendukung keputusan yang membantu pengelola wilayah memahami kebutuhan air masa depan dan mengevaluasi dampak tambahan kebutuhan air dari pembangunan infrastruktur baru.

---

## ICONFEST 2026

Project ini dikembangkan untuk:

**Data Mining Competition — ICONFEST 2026**  
Tema: **Smart City**
