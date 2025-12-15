# 🌍 Aplikasi Analisis Data Gempa dengan Streamlit

Aplikasi web interaktif untuk analisis data gempa menggunakan Deep Embedded Clustering (DEC) dan Autoencoder untuk deteksi anomali.

## 📋 Fitur

- **Dashboard**: Overview statistik dan visualisasi data gempa
- **EDA (Exploratory Data Analysis)**: Visualisasi distribusi, korelasi, dan sebaran data
- **DEC Clustering**: Deep Embedded Clustering dengan visualisasi geografis per tahun
- **Deteksi Anomali**: Deteksi anomali menggunakan model Autoencoder
- **Data Exploration**: Eksplorasi dan filter data dengan interaktif

## 🚀 Cara Menjalankan

### 1. Persiapan

Pastikan Anda sudah memiliki file-file berikut:
- `data/katalog_gempa_v2.csv` - Data gempa
- `scaler_dec.pkl` - Scaler untuk preprocessing (dihasilkan dari notebook)
- `model/autoencoder_model.h5` - Model autoencoder yang sudah dilatih

**Catatan**: Jika file `scaler_dec.pkl` dan `model/autoencoder_model.h5` belum ada, jalankan notebook `Salinan_dari_DEC_Update_Revisi.ipynb` terlebih dahulu untuk menghasilkan file-file tersebut.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur File

```
Deploy/
├── streamlit_app.py              # Aplikasi Streamlit utama
├── generate_scaler.py             # Script helper untuk membuat scaler
├── requirements.txt               # Dependencies Python
├── README.md                      # Dokumentasi
├── data/
│   └── katalog_gempa_v2.csv      # Data gempa
├── model/
│   └── autoencoder_model.h5      # Model autoencoder
├── scaler_dec.pkl                # Scaler untuk preprocessing (dibuat oleh generate_scaler.py)
└── salinan_dari_dec_update_revisi.py  # Source code asli
```

## 🔧 Menghasilkan File yang Diperlukan

Jika file `scaler_dec.pkl` dan `model/autoencoder_model.h5` belum ada, ikuti langkah berikut:

### Opsi 1: Menggunakan Script Helper (Paling Mudah)

Jika hanya file `scaler_dec.pkl` yang hilang (dan `model/autoencoder_model.h5` sudah ada):

```bash
python generate_scaler.py
```

Script ini akan membuat file `scaler_dec.pkl` secara otomatis dari data yang ada.

### Opsi 2: Menjalankan Notebook Python

Jika kedua file belum ada:

1. Buka file `salinan_dari_dec_update_revisi.py`
2. Jalankan semua bagian hingga selesai
3. Pastikan file berikut dihasilkan:
   - `scaler_dec.pkl` (dari bagian "PERSIAPAN MODELING DEC")
   - `model/autoencoder_model.h5` (dari bagian "TRAINING AUTOENCODER")
   - `encoder_model.h5` (opsional, akan dibuat otomatis dari autoencoder jika tidak ada)

## 📊 Halaman Aplikasi

### 1. Dashboard
- Statistik utama (total data, rentang tahun, rata-rata magnitudo dan kedalaman)
- Grafik magnitudo per tahun
- Distribusi magnitudo
- Peta sebaran lokasi gempa

### 2. EDA - Exploratory Data Analysis
- Distribusi magnitudo
- Distribusi kedalaman
- Sebaran lokasi gempa (dengan filter tahun)
- Magnitudo vs waktu
- Korelasi kedalaman vs magnitudo

### 3. DEC Clustering
- Deep Embedded Clustering dengan ekstraksi latent space
- K-Means clustering per tahun
- Visualisasi geografis cluster dengan peta interaktif
- Analisis cluster dominan per tahun
- Silhouette score untuk evaluasi clustering

### 4. Deteksi Anomali
- Input parameter gempa (magnitude, depth, latitude, longitude)
- Deteksi anomali menggunakan reconstruction error
- Visualisasi perbandingan error dengan threshold
- Batch anomaly detection untuk seluruh dataset

### 5. Data Exploration
- Filter data berdasarkan tahun, magnitudo, dan kedalaman
- Tabel data interaktif
- Statistik deskriptif
- Download data sebagai CSV

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Framework untuk membuat aplikasi web
- **Pandas**: Manipulasi dan analisis data
- **NumPy**: Komputasi numerik
- **Matplotlib & Seaborn**: Visualisasi statis
- **Plotly**: Visualisasi interaktif
- **TensorFlow/Keras**: Deep learning untuk autoencoder
- **Scikit-learn**: Machine learning utilities

## 📝 Catatan

- Threshold anomali saat ini: `0.0001606588873608411` (dihitung menggunakan mean + 3×std)
- Untuk performa yang lebih baik, beberapa visualisasi menggunakan sampling data
- Pastikan model sudah dilatih dengan data yang sesuai sebelum digunakan

## 🤝 Kontribusi

Jika menemukan bug atau ingin menambahkan fitur, silakan buat issue atau pull request.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik.

