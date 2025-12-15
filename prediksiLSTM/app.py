import streamlit as st
import pandas as pd
import altair as alt  # Pastikan altair diimpor
from pathlib import Path
import folium
from folium.plugins import HeatMap
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json

# Import streamlit_folium dengan error handling
FOLIUM_AVAILABLE = False
try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except (ImportError, Exception) as e:
    FOLIUM_AVAILABLE = False
    import streamlit.components.v1 as components
    
    # Fallback function menggunakan st.components.v1.html
    def st_folium(mymap, width=700, height=500, key=None, returned_objects=None):
        """Fallback function jika streamlit_folium tidak tersedia"""
        try:
            # Coba gunakan _repr_html_() untuk mendapatkan HTML
            html_string = mymap._repr_html_()
            components.html(html_string, width=width, height=height, scrolling=False)
        except Exception as e:
            st.error(f"⚠️ Error menampilkan peta: {e}")
            st.info("💡 **Solusi**: Pastikan streamlit-folium terinstall dengan benar. Jalankan: `pip install streamlit-folium`")
            st.info("📖 Lihat file `TROUBLESHOOTING_ST_FOLIUM.md` untuk panduan lengkap.")

# Tampilkan warning sekali di awal jika streamlit_folium tidak tersedia
if not FOLIUM_AVAILABLE:
    st.warning("⚠️ **Peringatan**: streamlit-folium tidak tersedia. Menggunakan mode fallback. Beberapa fitur interaktif mungkin terbatas.")
    with st.expander("ℹ️ Cara Memperbaiki", expanded=False):
        st.markdown("""
        **Langkah-langkah:**
        1. Install ulang: `pip install streamlit-folium`
        2. Restart aplikasi: Stop dan jalankan ulang `streamlit run app.py`
        3. Clear cache: Hapus folder `.streamlit/cache`
        4. Lihat file `TROUBLESHOOTING_ST_FOLIUM.md` untuk panduan lengkap
        """)

st.set_page_config(page_title="Peta Gempa Indonesia", layout="wide")

# Menentukan path file lokal
file_path = Path("data/katalog_gempa_v2clean.csv")

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

try:
    data = load_data(file_path)
    st.caption(f"Data dimuat: {len(data):,} baris")
except FileNotFoundError:
    st.error(f"File {file_path} tidak ditemukan. Pastikan path file benar.")
    st.stop()

# Pastikan kolom datetime terparse dengan benar
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
data = data.dropna(subset=['datetime'])  # Hapus data yang memiliki tanggal tidak valid

# Pastikan kolom latitude, longitude, dan magnitude ada dan valid
data = data.dropna(subset=['latitude', 'longitude', 'magnitude'])  # Hapus baris dengan nilai NaN pada kolom penting
data['magnitude'] = pd.to_numeric(data['magnitude'], errors='coerce')  # Pastikan magnitude adalah numerik

# Pastikan hanya data yang valid yang digunakan
data = data.dropna(subset=['magnitude'])  # Menghapus baris dengan nilai NaN pada magnitude

# Parse datetime agar ada fitur tahun/bulan
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month

# Cache fungsi pembuatan peta untuk performa lebih baik
@st.cache_data
def get_filtered_data_by_year(_data, year):
    """Cache filtered data berdasarkan tahun"""
    return _data[_data['year'] == year].copy()

@st.cache_data
def calculate_stats(_filtered_data):
    """
    Menghitung statistik deskriptif untuk data magnitudo gempa.
    
    Fungsi ini menggunakan pandas aggregation untuk menghitung 5 metrik statistik:
    - count: Jumlah total kejadian gempa (n)
    - min: Magnitudo minimum (gempa terkecil)
    - max: Magnitudo maksimum (gempa terbesar)
    - mean: Magnitudo rata-rata (mean/rerata)
    - median: Magnitudo median (nilai tengah)
    
    Parameter:
        _filtered_data: DataFrame yang sudah difilter berdasarkan tahun
    
    Returns:
        Series berisi statistik: ['count', 'min', 'max', 'mean', 'median']
    
    Contoh output:
        count     318.00
        min        2.15
        max        6.46
        mean       4.39
        median     4.54
    """
    return _filtered_data['magnitude'].agg(['count', 'min', 'max', 'mean', 'median'])

# ============================================================================
# METODOLOGI PREDIKSI LOKASI GEMPA TAHUN 2026
# ============================================================================
# Metode: Frequency-Based Hotspot Prediction dengan Spatial Binning
#
# Konsep Dasar:
# - Asumsi: Lokasi yang sering mengalami gempa di masa lalu cenderung 
#   mengalami gempa lagi di masa depan (asumsi stabilitas pola seismik)
# - Metode ini menggunakan analisis frekuensi dan spatial aggregation
#
# Langkah-langkah:
# 1. Spatial Binning: Membagi wilayah Indonesia menjadi grid 0.5° x 0.5°
# 2. Aggregation: Mengelompokkan semua gempa historis ke dalam grid
# 3. Frequency Counting: Menghitung frekuensi gempa per grid
# 4. Ranking: Mengurutkan grid berdasarkan frekuensi tertinggi
# 5. Selection: Memilih top-N hotspot dengan frekuensi tertinggi
# 6. Prediction: Lokasi hotspot ini diprediksi akan mengalami gempa di 2026
#
# Kelebihan Metode:
# - Sederhana dan interpretatif
# - Berbasis data historis yang nyata
# - Tidak memerlukan model ML kompleks
# - Cocok untuk identifikasi daerah rawan gempa
#
# Keterbatasan:
# - Tidak memprediksi waktu kejadian yang tepat
# - Tidak memprediksi magnitudo
# - Asumsi pola tetap stabil (tidak ada perubahan geologi drastis)
# ============================================================================

# ============================================================================
# FUNGSI LSTM UNTUK PREDIKSI HOTSPOT
# ============================================================================

def prepare_time_series_data(df: pd.DataFrame, bin_size=0.5):
    """
    Mempersiapkan data time series untuk LSTM.
    Membuat time series frekuensi gempa per grid per tahun.
    """
    if df.empty:
        return None, None
    
    # Spatial binning
    tmp = df.copy()
    tmp["lat_bin"] = (tmp["latitude"] / bin_size).round() * bin_size
    tmp["lon_bin"] = (tmp["longitude"] / bin_size).round() * bin_size
    
    # Buat time series: frekuensi gempa per grid per tahun
    time_series_data = []
    grid_list = []
    
    # Group by grid
    for (lat_bin, lon_bin), group in tmp.groupby(["lat_bin", "lon_bin"]):
        # Hitung frekuensi per tahun untuk grid ini
        yearly_counts = group.groupby("year").size().sort_index()
        
        # Pastikan ada data minimal 3 tahun untuk training
        if len(yearly_counts) >= 3:
            # Buat sequence lengkap dari 2008-2025
            years = range(2008, 2026)
            frequency_series = [yearly_counts.get(year, 0) for year in years]
            
            # Hanya tambahkan jika ada variasi data (tidak semua 0)
            if sum(frequency_series) > 0:
                time_series_data.append(frequency_series)
                grid_list.append((lat_bin, lon_bin, group["magnitude"].mean()))
    
    if not time_series_data:
        return None, None
    
    return np.array(time_series_data), grid_list

def create_sequences(data, sequence_length=5):
    """
    Membuat sequences untuk LSTM dari time series data.
    """
    X, y = [], []
    for series in data:
        for i in range(len(series) - sequence_length):
            X.append(series[i:i+sequence_length])
            y.append(series[i+sequence_length])
    return np.array(X), np.array(y)

@st.cache_resource
def build_lstm_model(input_shape, units=50):
    """
    Membangun model LSTM untuk prediksi frekuensi gempa.
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def save_lstm_model(model, filepath="models/lstm_earthquake_model.h5"):
    """
    Menyimpan model LSTM ke file.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)
    return filepath

def save_prediction_results(hotspot_points, method, filepath="models/prediction_results.json"):
    """
    Menyimpan hasil prediksi ke file JSON.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        'method': method,
        'hotspot_points': hotspot_points,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    return filepath

def load_prediction_results(filepath="models/prediction_results.json"):
    """
    Memuat hasil prediksi dari file JSON.
    """
    if Path(filepath).exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('hotspot_points', []), data.get('method', 'Unknown')
        except Exception as e:
            st.warning(f"Gagal memuat hasil prediksi: {e}")
            return None, None
    return None, None

def load_lstm_model(filepath="models/lstm_earthquake_model.h5"):
    """
    Memuat model LSTM dari file jika ada.
    Menangani masalah kompatibilitas dengan load compile=False dan compile ulang.
    """
    if Path(filepath).exists():
        try:
            # Coba load dengan compile=False untuk menghindari masalah deserialisasi metrics
            model = load_model(filepath, compile=False)
            
            # Compile ulang model dengan konfigurasi yang sama seperti saat training
            # Ini menghindari masalah kompatibilitas metrics
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            # Jika masih error, coba dengan custom_objects sebagai string
            try:
                # Gunakan string untuk metrics, bukan instance
                custom_objects = {
                    'mse': 'mse',
                    'mae': 'mae'
                }
                model = load_model(filepath, compile=False, custom_objects=custom_objects)
                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                return model
            except Exception as e2:
                # Jika masih gagal, coba load tanpa compile dan compile manual
                try:
                    model = load_model(filepath, compile=False)
                    # Compile dengan string metrics
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae']
                    )
                    return model
                except Exception as e3:
                    st.warning(f"Gagal memuat model: {e3}")
                    st.info("💡 **Saran**: Hapus model lama (`models/lstm_earthquake_model.h5`) dan training ulang untuk menghindari masalah kompatibilitas.")
                    return None
    return None

def estimate_training_time(time_series_data, epochs=30, batch_size=32):
    """
    Mengestimasi waktu training berdasarkan data yang ada.
    """
    if time_series_data is None or len(time_series_data) == 0:
        return None, None, None
    
    sequence_length = 5
    total_sequences = 0
    
    # Hitung jumlah sequences
    for series in time_series_data:
        total_sequences += max(0, len(series) - sequence_length)
    
    # Estimasi berdasarkan:
    # - Jumlah sequences
    # - Batch size
    # - Epochs
    # - Kompleksitas model (2 LSTM layers dengan 50 units)
    
    batches_per_epoch = max(1, total_sequences // batch_size)
    total_batches = batches_per_epoch * epochs
    
    # Estimasi waktu per batch (sekitar 0.1-0.5 detik tergantung hardware)
    # Untuk CPU: ~0.3-0.5 detik per batch
    # Untuk GPU: ~0.05-0.1 detik per batch
    time_per_batch_cpu = 0.4  # detik
    time_per_batch_gpu = 0.08  # detik
    
    # Cek apakah GPU tersedia
    try:
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    except:
        gpu_available = False
    
    if gpu_available:
        estimated_seconds = total_batches * time_per_batch_gpu
        device = "GPU"
    else:
        estimated_seconds = total_batches * time_per_batch_cpu
        device = "CPU"
    
    # Tambahkan overhead preprocessing (sekitar 10-20%)
    estimated_seconds = estimated_seconds * 1.15
    
    return {
        'total_grids': len(time_series_data),
        'total_sequences': total_sequences,
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches,
        'estimated_seconds': estimated_seconds,
        'device': device
    }

def train_lstm_model(time_series_data, epochs=50, batch_size=32):
    """
    Melatih model LSTM untuk prediksi frekuensi gempa.
    """
    if time_series_data is None or len(time_series_data) == 0:
        return None, None, None
    
    try:
        # Buat sequences dari data asli (belum dinormalisasi global)
        sequence_length = 5
        X_all, y_all = [], []
        
        for series in time_series_data:
            # Normalisasi per series
            series_2d = series.reshape(-1, 1)
            scaler_local = MinMaxScaler()
            scaled = scaler_local.fit_transform(series_2d).flatten()
            
            # Buat sequences
            for i in range(len(scaled) - sequence_length):
                X_all.append(scaled[i:i+sequence_length])
                y_all.append(scaled[i+sequence_length])
        
        if len(X_all) == 0:
            return None, None, None
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        # Reshape untuk LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Build dan train model
        model = build_lstm_model((sequence_length, 1), units=50)
        
        # Training dengan progress
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),
            validation_data=(X_test, y_test) if len(X_test) > 0 else None,
            verbose=0
        )
        
        # Return model dan None untuk scaler (karena kita normalisasi per series)
        return model, None, history
    except Exception as e:
        st.error(f"Error training LSTM: {e}")
        return None, None, None

def predict_with_lstm(model, scaler, time_series_data, grid_list, target_year=2026, top_k=15):
    """
    Memprediksi frekuensi gempa tahun 2026 menggunakan LSTM untuk setiap grid.
    """
    if model is None or time_series_data is None or len(time_series_data) == 0:
        return []
    
    predictions = []
    sequence_length = 5
    
    for idx, series in enumerate(time_series_data):
        if idx >= len(grid_list):
            continue
            
        lat_bin, lon_bin, mean_mag = grid_list[idx]
        
        # Normalisasi
        series_2d = series.reshape(-1, 1)
        scaler_local = MinMaxScaler()
        scaled = scaler_local.fit_transform(series_2d).flatten()
        
        # Ambil 5 tahun terakhir untuk prediksi
        if len(scaled) >= sequence_length:
            last_sequence = scaled[-sequence_length:].reshape(1, sequence_length, 1)
            
            try:
                # Prediksi
                pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
                
                # Denormalisasi
                pred_2d = np.array([[pred_scaled]])
                pred_denorm = scaler_local.inverse_transform(pred_2d)[0][0]
                pred_frequency = max(0, int(round(pred_denorm)))  # Pastikan non-negative
                
                predictions.append({
                    "lat": lat_bin,
                    "lon": lon_bin,
                    "count": pred_frequency,  # Prediksi frekuensi dari LSTM
                    "mag": mean_mag,
                    "year": target_year,
                    "month": None,
                })
            except Exception as e:
                # Skip jika error, gunakan frekuensi historis sebagai fallback
                predictions.append({
                    "lat": lat_bin,
                    "lon": lon_bin,
                    "count": int(series[-1]) if len(series) > 0 else 0,
                    "mag": mean_mag,
                    "year": target_year,
                    "month": None,
                })
    
    # Sort berdasarkan prediksi frekuensi tertinggi
    predictions.sort(key=lambda x: x["count"], reverse=True)
    
    return predictions[:top_k]

# Fungsi-fungsi helper (didefinisikan sekali, tidak dihitung ulang)
def build_hotspot_points(df: pd.DataFrame, bin_size=0.5, top_k=15):
    """
    Membangun titik hotspot prediksi berdasarkan frekuensi gempa historis.
    
    Metodologi:
    1. Spatial Binning: Membagi koordinat menjadi grid berukuran bin_size (0.5°)
       - 0.5° ≈ 55 km di khatulistiwa
    2. Grouping: Mengelompokkan semua gempa ke dalam grid yang sama
    3. Aggregation: Menghitung:
       - count: Jumlah gempa per grid (frekuensi)
       - mean_mag: Magnitudo rata-rata per grid
    4. Ranking: Mengurutkan grid berdasarkan frekuensi (count) tertinggi
    5. Selection: Memilih top_k grid dengan frekuensi tertinggi sebagai hotspot
    
    Parameter:
        df: DataFrame berisi data gempa historis (2008-2025)
        bin_size: Ukuran grid dalam derajat (default 0.5°)
        top_k: Jumlah hotspot teratas yang akan dipilih (default 15)
    
    Returns:
        List of dict berisi koordinat hotspot yang diprediksi untuk tahun 2026
    """
    if df.empty:
        return []
    
    # Langkah 1: Spatial Binning - Rounding koordinat ke grid terdekat
    tmp = df.copy()
    tmp["lat_bin"] = (tmp["latitude"] / bin_size).round() * bin_size
    tmp["lon_bin"] = (tmp["longitude"] / bin_size).round() * bin_size
    
    # Langkah 2 & 3: Grouping dan Aggregation
    agg = (
        tmp.groupby(["lat_bin", "lon_bin"])  # Kelompokkan berdasarkan grid
        .agg(
            count=("magnitude", "count"),      # Hitung frekuensi gempa
            mean_mag=("magnitude", "mean")     # Hitung magnitudo rata-rata
        )
        .reset_index()
        .sort_values("count", ascending=False)  # Urutkan berdasarkan frekuensi
        .head(top_k)  # Langkah 4 & 5: Pilih top-k hotspot
    )
    
    # Langkah 6: Mapping ke tahun prediksi 2026
    # Catatan: Bulan tidak relevan karena prediksi ini berbasis frekuensi historis
    # dan hanya memprediksi lokasi, bukan waktu spesifik kejadian
    target_year_pred = 2026
    return [
        {
            "lat": r.lat_bin,
            "lon": r.lon_bin,
            "count": int(r.count),        # Frekuensi gempa historis
            "mag": float(r.mean_mag),     # Magnitudo rata-rata historis
            "year": target_year_pred,     # Tahun prediksi
            "month": None,                # Tidak relevan untuk prediksi lokasi
        }
        for r in agg.itertuples()
    ]

def build_heatmap_points(df: pd.DataFrame, bin_size=0.5):
    if df.empty:
        return []
    tmp = df.copy()
    tmp["lat_bin"] = (tmp["latitude"] / bin_size).round() * bin_size
    tmp["lon_bin"] = (tmp["longitude"] / bin_size).round() * bin_size
    agg = (
        tmp.groupby(["lat_bin", "lon_bin"])
        .agg(count=("magnitude", "count"), mean_mag=("magnitude", "mean"))
        .reset_index()
    )
    max_count = max(agg["count"].max(), 1)
    # Bobot heatmap: gabungan count (dominan) dan mean magnitude
    agg["weight"] = 0.8 * (agg["count"] / max_count) + 0.2 * (agg["mean_mag"] / agg["mean_mag"].max())
    return agg[["lat_bin", "lon_bin", "weight"]].values.tolist()

# --- Prediksi lokasi 2026 menggunakan LSTM ---
st.subheader("Peta prediksi lokasi 2026 menggunakan LSTM (Long Short-Term Memory)")

# Penjelasan metodologi prediksi LSTM
with st.expander("📚 Penjelasan Metodologi Prediksi LSTM", expanded=False):
    st.markdown("""
    ### **Metode Prediksi: LSTM (Long Short-Term Memory) untuk Time Series**
    
    #### **1. Konsep Dasar**
    Sistem ini menggunakan **LSTM Neural Network** untuk menganalisis pola temporal frekuensi gempa 
    per grid dari tahun 2008-2025, kemudian memprediksi frekuensi gempa di tahun 2026.
    
    #### **2. Langkah-Langkah Metodologi**
    
    **A. Spatial Binning (Pembagian Grid)**
    - Wilayah Indonesia dibagi menjadi grid berukuran **0.5° × 0.5°**
    - 0.5° ≈ **55 km** di khatulistiwa
    
    **B. Time Series Preparation**
    - Untuk setiap grid, dibuat time series frekuensi gempa per tahun (2008-2025)
    - Contoh: Grid A → [5, 8, 12, 10, 15, 18, 20, 22, ...] (frekuensi per tahun)
    
    **C. LSTM Training**
    - Model LSTM dilatih untuk mempelajari pola temporal
    - Input: Sequence 5 tahun terakhir
    - Output: Prediksi frekuensi tahun berikutnya
    
    **D. Prediction & Ranking**
    - LSTM memprediksi frekuensi gempa tahun 2026 untuk setiap grid
    - Grid diurutkan berdasarkan prediksi frekuensi tertinggi
    - Dipilih top-N grid sebagai hotspot prediksi
    
    #### **3. Kelebihan LSTM**
    - ✅ Mampu menangkap pola temporal kompleks
    - ✅ Mempertimbangkan trend dan seasonality
    - ✅ Lebih akurat untuk prediksi berbasis time series
    
    #### **4. Apa yang Diprediksi?**
    - ✅ **Lokasi** hotspot berdasarkan prediksi frekuensi LSTM
    - ✅ **Frekuensi prediksi** untuk tahun 2026
    - ✅ **Magnitudo estimasi** berdasarkan rata-rata historis
    """)

bin_size = 0.5  # Ukuran grid 0.5 derajat
top_k = st.slider("Tampilkan top-N hotspot", min_value=5, max_value=50, value=15, step=5)

# Pilihan metode prediksi
method = st.radio(
    "Pilih Metode Prediksi:",
    ["LSTM (Deep Learning)", "Frequency-Based (Statistik)"],
    index=0,
    help="LSTM menggunakan neural network untuk analisis temporal, Frequency-Based menggunakan analisis statistik sederhana"
)

# Menggunakan semua data historis tanpa filter bulan
heat_source_pred = data.copy()

# Cek apakah ada hasil prediksi yang tersimpan
saved_prediction_path = "models/prediction_results.json"
saved_hotspot_points, saved_method = load_prediction_results(saved_prediction_path)

# Inisialisasi session state untuk regenerate
if 'regenerate_prediction' not in st.session_state:
    st.session_state.regenerate_prediction = False

# Tampilkan info jika ada hasil prediksi tersimpan
if saved_hotspot_points and saved_method == method:
    col_info1, col_info2 = st.columns([3, 1])
    with col_info1:
        st.info(f"💾 **Hasil prediksi tersimpan ditemukan!** ({len(saved_hotspot_points)} hotspot dari metode {saved_method}). Peta akan ditampilkan otomatis. Centang 'Regenerate' jika ingin generate ulang.")
    with col_info2:
        st.session_state.regenerate_prediction = st.checkbox("🔄 Regenerate", value=st.session_state.regenerate_prediction, help="Centang untuk generate ulang prediksi")

if method == "LSTM (Deep Learning)":
    # Persiapan data time series untuk LSTM
    with st.spinner("⏳ Mempersiapkan data time series untuk LSTM..."):
        time_series_data, grid_list = prepare_time_series_data(heat_source_pred, bin_size=bin_size)
    
    if time_series_data is None or len(time_series_data) == 0:
        st.warning("⚠️ Data tidak cukup untuk LSTM. Minimal diperlukan 3 tahun data per grid. Menggunakan metode Frequency-Based sebagai fallback.")
        hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
    else:
        st.success(f"✅ Data time series siap: {len(time_series_data)} grid dengan data temporal")
        
        # Estimasi waktu training
        epochs_default = 30
        batch_size = 32
        time_estimate = estimate_training_time(time_series_data, epochs=epochs_default, batch_size=batch_size)
        
        if time_estimate:
            est_minutes = int(time_estimate['estimated_seconds'] // 60)
            est_seconds = int(time_estimate['estimated_seconds'] % 60)
            
            with st.expander("⏱️ Estimasi Waktu Training", expanded=True):
                st.markdown(f"""
                **Berdasarkan data Anda:**
                - **Jumlah Grid**: {time_estimate['total_grids']:,} grid
                - **Total Sequences**: {time_estimate['total_sequences']:,} sequences
                - **Batches per Epoch**: {time_estimate['batches_per_epoch']:,}
                - **Device**: {time_estimate['device']}
                
                **Estimasi waktu untuk 30 epochs:**
                - **Perkiraan**: ~{est_minutes} menit {est_seconds} detik
                - **Range**: {max(1, est_minutes-1)}-{est_minutes+2} menit (tergantung hardware)
                
                **Catatan:**
                - Estimasi untuk {time_estimate['device']}
                - Waktu aktual bisa berbeda ±20% tergantung beban sistem
                - GPU akan lebih cepat jika tersedia
                """)
        
        # Cek apakah model sudah ada
        model_path = "models/lstm_earthquake_model.h5"
        saved_model = load_lstm_model(model_path)
        
        col_train1, col_train2, col_train3 = st.columns([2, 1, 1])
        
        with col_train1:
            if saved_model is not None:
                st.info("💾 **Model LSTM tersimpan ditemukan!** Model akan dimuat otomatis. Klik 'Train Ulang' jika ingin training baru.")
            else:
                st.info("💡 **Model belum ada.** Klik 'Train & Predict' untuk melatih model LSTM.")
        
        with col_train2:
            retrain = st.checkbox("🔄 Train Ulang", value=False, help="Centang untuk training ulang meskipun model sudah ada")
        
        with col_train3:
            if Path(model_path).exists():
                if st.button("🗑️ Hapus Model", help="Hapus model lama jika ada masalah kompatibilitas"):
                    try:
                        Path(model_path).unlink()
                        st.success("✅ Model berhasil dihapus! Silakan training ulang.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Gagal menghapus model: {e}")
        
        # Training LSTM
        epochs = st.slider("Epochs untuk training LSTM", min_value=10, max_value=100, value=30, step=10)
        
        # Update estimasi waktu berdasarkan epochs yang dipilih
        if time_estimate and epochs != epochs_default:
            scale_factor = epochs / epochs_default
            new_est_seconds = time_estimate['estimated_seconds'] * scale_factor
            new_est_minutes = int(new_est_seconds // 60)
            new_est_seconds_remain = int(new_est_seconds % 60)
            st.caption(f"⏱️ Estimasi waktu untuk {epochs} epochs: ~{new_est_minutes} menit {new_est_seconds_remain} detik")
        
        model = None
        history = None
        
        # Cek apakah menggunakan hasil prediksi tersimpan
        if saved_hotspot_points and saved_method == method and not (st.session_state.regenerate_prediction or retrain):
            hotspot_points = saved_hotspot_points
            st.success(f"✅ Menggunakan hasil prediksi tersimpan! ({len(hotspot_points)} hotspot)")
        # Gunakan model yang sudah ada atau training baru
        elif saved_model is not None and not retrain:
            model = saved_model
            st.success("✅ Menggunakan model LSTM yang sudah tersimpan!")
            
            # Prediksi langsung dengan model yang sudah ada
            with st.spinner("⏳ Memprediksi frekuensi gempa tahun 2026 dengan model tersimpan..."):
                hotspot_points = predict_with_lstm(
                    model, None, time_series_data, grid_list, 
                    target_year=2026, top_k=top_k
                )
            
            if hotspot_points:
                st.success(f"✅ Prediksi selesai! {len(hotspot_points)} hotspot ditemukan.")
                # Simpan hasil prediksi
                try:
                    save_prediction_results(hotspot_points, method, saved_prediction_path)
                    st.session_state.regenerate_prediction = False  # Reset regenerate flag
                except Exception as e:
                    st.warning(f"⚠️ Gagal menyimpan hasil prediksi: {e}")
            else:
                st.error("❌ Gagal melakukan prediksi. Menggunakan metode Frequency-Based.")
                hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
        
        # Training baru jika diminta atau model tidak ada
        if (saved_model is None or retrain) and st.button("🚀 Train & Predict dengan LSTM", type="primary"):
            with st.spinner(f"⏳ Training LSTM model ({epochs} epochs)... Ini mungkin memakan waktu beberapa menit..."):
                model, scaler, history = train_lstm_model(time_series_data, epochs=epochs)
            
            if model is not None:
                st.success("✅ Model LSTM berhasil dilatih!")
                
                # Simpan model
                try:
                    save_path = save_lstm_model(model, model_path)
                    st.success(f"💾 Model disimpan di: {save_path}")
                except Exception as e:
                    st.warning(f"⚠️ Gagal menyimpan model: {e}")
                
                # Prediksi
                with st.spinner("⏳ Memprediksi frekuensi gempa tahun 2026..."):
                    hotspot_points = predict_with_lstm(
                        model, scaler, time_series_data, grid_list, 
                        target_year=2026, top_k=top_k
                    )
                
                if hotspot_points:
                    st.success(f"✅ Prediksi selesai! {len(hotspot_points)} hotspot ditemukan.")
                    
                    # Simpan hasil prediksi
                    try:
                        save_prediction_results(hotspot_points, method, saved_prediction_path)
                        st.success("💾 Hasil prediksi disimpan! Peta akan tetap ditampilkan setelah refresh.")
                        st.session_state.regenerate_prediction = False  # Reset regenerate flag
                    except Exception as e:
                        st.warning(f"⚠️ Gagal menyimpan hasil prediksi: {e}")
                    
                    # Tampilkan grafik training history jika ada
                    if history is not None:
                        with st.expander("📈 Training History", expanded=False):
                            hist_df = pd.DataFrame({
                                'Epoch': range(1, len(history.history['loss']) + 1),
                                'Loss': history.history['loss'],
                            })
                            
                            # Tambahkan val_loss jika ada
                            if 'val_loss' in history.history and len(history.history['val_loss']) > 0:
                                hist_df['Val Loss'] = history.history['val_loss']
                                chart = alt.Chart(hist_df).mark_line().encode(
                                    x='Epoch',
                                    y=alt.Y('value:Q', scale=alt.Scale(type='log'), title='Loss'),
                                    color='key:N'
                                ).transform_fold(
                                    ['Loss', 'Val Loss'],
                                    as_=['key', 'value']
                                ).properties(width=600, height=300, title="Training History")
                            else:
                                chart = alt.Chart(hist_df).mark_line(color='red').encode(
                                    x='Epoch',
                                    y=alt.Y('Loss', scale=alt.Scale(type='log'), title='Loss'),
                                ).properties(width=600, height=300, title="Training Loss")
                            
                            st.altair_chart(chart, use_container_width=True)
                else:
                    st.error("❌ Gagal melakukan prediksi. Menggunakan metode Frequency-Based.")
                    hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
            else:
                st.error("❌ Gagal training model. Menggunakan metode Frequency-Based.")
                hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
        
        # Jika belum ada model dan belum training
        if saved_model is None and model is None:
            hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
else:
    # Metode Frequency-Based (original)
    # Cek apakah menggunakan hasil prediksi tersimpan
    if saved_hotspot_points and saved_method == method and not st.session_state.regenerate_prediction:
        hotspot_points = saved_hotspot_points
        st.success(f"✅ Menggunakan hasil prediksi tersimpan! ({len(hotspot_points)} hotspot)")
    else:
        hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
        # Simpan hasil prediksi
        try:
            save_prediction_results(hotspot_points, method, saved_prediction_path)
            st.session_state.regenerate_prediction = False  # Reset regenerate flag
        except Exception as e:
            st.warning(f"⚠️ Gagal menyimpan hasil prediksi: {e}")

# --- Heatmap estimasi lokasi (berbasis pola historis) ---
st.subheader("Heatmap estimasi kerawanan (berdasarkan pola 2008-2025)")
heat_year = st.selectbox("Tahun target (estimasi)", range(2008, 2027), index=list(range(2008, 2027)).index(2026))

# Menggunakan semua data historis tanpa filter bulan untuk heatmap
heat_source = data.copy()
heatmap_points = build_heatmap_points(heat_source)


# Membuat peta interaktif menggunakan Folium dengan optimasi performa
@st.cache_resource
def create_map_cached(_year, _data_count, _data_sum, _max_markers=5000):
    """
    Versi cached dari create_map untuk peta historis per tahun.
    Cache berdasarkan tahun, jumlah data, sum magnitude, dan max_markers.
    
    Parameter dengan prefix _ akan digunakan sebagai key cache.
    Data aktual akan diambil dari get_filtered_data_by_year yang sudah di-cache.
    """
    # Ambil data dari cache filtered_data berdasarkan tahun
    filtered_data = get_filtered_data_by_year(data, _year)
    
    # Buat peta dari data yang sudah di-cache
    return create_map(
        filtered_data,
        predicted_points=None,
        target_year=_year,
        heatmap_points=None,
        max_markers=_max_markers
    )

def add_grid_to_map(mymap, bin_size=0.5, bounds=None, max_grid_lines=50):
    """
    Menambahkan grid layer ke peta untuk visualisasi spatial binning.
    Dioptimasi untuk performa dengan membatasi jumlah grid lines.
    
    Parameters:
        mymap: Folium map object
        bin_size: Ukuran grid dalam derajat (default 0.5°)
        bounds: Tuple (min_lat, min_lon, max_lat, max_lon) untuk batas grid
                Jika None, akan menggunakan batas Indonesia
        max_grid_lines: Maksimal jumlah grid lines (default 50 untuk performa)
    """
    # Tentukan batas Indonesia jika tidak diberikan
    if bounds is None:
        min_lat, min_lon = -11.0, 95.0  # Batas barat-selatan Indonesia
        max_lat, max_lon = 6.0, 141.0   # Batas timur-utara Indonesia
    else:
        min_lat, min_lon, max_lat, max_lon = bounds
    
    # Hitung jumlah grid lines yang akan dibuat
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    num_lat_lines = int(lat_range / bin_size) + 1
    num_lon_lines = int(lon_range / bin_size) + 1
    total_lines = num_lat_lines + num_lon_lines
    
    # Jika terlalu banyak, batasi atau gunakan grid lebih besar
    if total_lines > max_grid_lines:
        # Gunakan FeatureGroup untuk performa lebih baik
        grid_group = folium.FeatureGroup(name="Grid", show=False)
        
        # Batasi hanya di area yang relevan (tambah padding kecil)
        padding = min(lat_range * 0.1, lon_range * 0.1, 2.0)  # Max 2 derajat padding
        min_lat_limited = min_lat - padding
        max_lat_limited = max_lat + padding
        min_lon_limited = min_lon - padding
        max_lon_limited = max_lon + padding
        
        # Buat grid lines dengan batasan
        # Garis lintang (latitude) - batasi jumlah
        lat = min_lat_limited
        lat_count = 0
        while lat <= max_lat_limited and lat_count < max_grid_lines // 2:
            folium.PolyLine(
                locations=[[lat, min_lon_limited], [lat, max_lon_limited]],
                color='#888888',
                weight=1,
                opacity=0.4,
                dashArray='5, 5'
            ).add_to(grid_group)
            lat += bin_size
            lat_count += 1
        
        # Garis bujur (longitude) - batasi jumlah
        lon = min_lon_limited
        lon_count = 0
        while lon <= max_lon_limited and lon_count < max_grid_lines // 2:
            folium.PolyLine(
                locations=[[min_lat_limited, lon], [max_lat_limited, lon]],
                color='#888888',
                weight=1,
                opacity=0.4,
                dashArray='5, 5'
            ).add_to(grid_group)
            lon += bin_size
            lon_count += 1
        
        grid_group.add_to(mymap)
    else:
        # Jika jumlah grid lines wajar, buat seperti biasa
        # Gunakan FeatureGroup untuk performa lebih baik
        grid_group = folium.FeatureGroup(name="Grid")
        
        # Garis lintang (latitude)
        lat = min_lat
        while lat <= max_lat:
            folium.PolyLine(
                locations=[[lat, min_lon], [lat, max_lon]],
                color='#888888',
                weight=1,
                opacity=0.4,
                dashArray='5, 5'
            ).add_to(grid_group)
            lat += bin_size
        
        # Garis bujur (longitude)
        lon = min_lon
        while lon <= max_lon:
            folium.PolyLine(
                locations=[[min_lat, lon], [max_lat, lon]],
                color='#888888',
                weight=1,
                opacity=0.4,
                dashArray='5, 5'
            ).add_to(grid_group)
            lon += bin_size
        
        grid_group.add_to(mymap)
    
    return mymap

def create_map(data, predicted_points=None, target_year=None, heatmap_points=None, max_markers=5000, show_grid=False, bin_size=0.5):
    """
    Membuat peta interaktif dengan optimasi performa.
    
    Optimasi yang diterapkan:
    - Limitasi jumlah marker maksimal
    - Pre-format datetime untuk performa lebih baik
    - Menggunakan itertuples() yang lebih cepat
    - Popup HTML yang efisien
    
    Parameters:
        show_grid: Boolean, apakah menampilkan grid (default False)
        bin_size: Ukuran grid dalam derajat jika show_grid=True (default 0.5°)
    """
    # Tentukan pusat peta (Indonesia)
    map_center = [-5.0, 115.0]
    zoom_start = 5
    
    mymap = folium.Map(location=map_center, zoom_start=zoom_start)
    
    # Tambahkan grid jika diminta (dioptimasi untuk performa)
    if show_grid:
        # Tentukan bounds dari data atau hotspot points (hanya area yang relevan)
        if predicted_points and len(predicted_points) > 0:
            lats = [p['lat'] for p in predicted_points]
            lons = [p['lon'] for p in predicted_points]
            # Tambah padding 2-3 grid untuk konteks visual (bukan seluruh Indonesia)
            padding = bin_size * 3
            bounds = (min(lats) - padding, min(lons) - padding, 
                     max(lats) + padding, max(lons) + padding)
        elif not data.empty:
            padding = bin_size * 3
            bounds = (data['latitude'].min() - padding, data['longitude'].min() - padding,
                     data['latitude'].max() + padding, data['longitude'].max() + padding)
        else:
            # Jika tidak ada data, gunakan area kecil di sekitar pusat peta
            bounds = (map_center[0] - 5, map_center[1] - 5, 
                     map_center[0] + 5, map_center[1] + 5)
        
        # Tambahkan grid dengan batasan maksimal 50 lines untuk performa
        mymap = add_grid_to_map(mymap, bin_size=bin_size, bounds=bounds, max_grid_lines=50)

    # Optimasi: Jika data terlalu banyak, batasi jumlah marker
    data_to_plot = data.copy()
    original_count = len(data_to_plot)
    if len(data_to_plot) > max_markers:
        # Ambil sample berdasarkan magnitudo terbesar untuk performa
        data_to_plot = data_to_plot.nlargest(max_markers, 'magnitude')
    
    # Optimasi: Pre-format datetime sekali untuk semua row (lebih cepat dari format di loop)
    if not data_to_plot.empty and 'datetime' in data_to_plot.columns:
        # Buat kolom formatted datetime sekali saja menggunakan vectorized operations
        data_to_plot = data_to_plot.copy()
        mask_notna = data_to_plot['datetime'].notna()
        data_to_plot.loc[mask_notna, 'waktu_str'] = data_to_plot.loc[mask_notna, 'datetime'].dt.strftime("%d %B %Y, %H:%M:%S WIB")
        data_to_plot.loc[~mask_notna, 'waktu_str'] = "Tidak tersedia"
        data_to_plot.loc[mask_notna, 'tanggal_str'] = data_to_plot.loc[mask_notna, 'datetime'].dt.strftime("%Y-%m-%d")
        data_to_plot.loc[~mask_notna, 'tanggal_str'] = "N/A"
        data_to_plot.loc[mask_notna, 'jam_str'] = data_to_plot.loc[mask_notna, 'datetime'].dt.strftime("%H:%M:%S")
        data_to_plot.loc[~mask_notna, 'jam_str'] = "N/A"
    
    # Optimasi: Gunakan itertuples() yang lebih cepat daripada iterrows()
    for row in data_to_plot.itertuples():
        # Ambil nilai dengan itertuples (lebih cepat)
        magnitude = getattr(row, 'magnitude', 0)
        latitude = getattr(row, 'latitude', 0)
        longitude = getattr(row, 'longitude', 0)
        location = getattr(row, 'location', 'N/A')
        
        # Gunakan pre-formatted datetime jika ada
        if hasattr(row, 'waktu_str'):
            waktu_str = row.waktu_str
            tanggal_str = row.tanggal_str
            jam_str = row.jam_str
        else:
            # Fallback jika pre-format tidak ada
            dt_val = getattr(row, 'datetime', None)
            if pd.notna(dt_val):
                waktu_str = dt_val.strftime("%d %B %Y, %H:%M:%S WIB")
                tanggal_str = dt_val.strftime("%Y-%m-%d")
                jam_str = dt_val.strftime("%H:%M:%S")
            else:
                waktu_str = "Tidak tersedia"
                tanggal_str = "N/A"
                jam_str = "N/A"
        
        # Buat popup dengan informasi lengkap (optimasi: gunakan f-string, minimalkan whitespace)
        popup_content = f"""<div style="font-family:Arial;min-width:200px;"><b>📍 Info Gempa</b><br><hr style="margin:3px 0;"><b>Waktu:</b> {waktu_str}<br><b>Tanggal:</b> {tanggal_str}<br><b>Jam:</b> {jam_str}<br><hr style="margin:3px 0;"><b>Magnitudo:</b> {magnitude:.2f}<br><b>Lokasi:</b> {location}<br><b>Koordinat:</b> ({latitude:.4f}, {longitude:.4f})</div>"""
        
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=5,  # Sedikit lebih kecil untuk performa
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=280),  # Sedikit lebih kecil
            tooltip=f"{magnitude:.1f}"  # Tooltip lebih sederhana
        ).add_to(mymap)
    
    # Tampilkan peringatan jika data dibatasi
    if len(data) > max_markers:
        st.info(f"⚠️ Data ditampilkan terbatas {max_markers:,} dari {len(data):,} gempa (menampilkan yang magnitudo terbesar untuk performa)")

    # Tambahkan marker prediksi hotspot (jika ada dan sesuai tahun yang dipilih)
    if predicted_points:
        for p in predicted_points:
            if target_year is None or p.get("year") == target_year:
                # Buat popup dengan informasi lengkap termasuk koordinat
                # Catatan: Bulan tidak ditampilkan karena prediksi ini hanya untuk lokasi, bukan waktu spesifik
                bulan_info = f"<b>Bulan:</b> {p['month']:02d}<br>" if p.get('month') is not None else ""
                popup_content_pred = f"""
                <div style="font-family: Arial; min-width: 200px;">
                    <b>🔮 Prediksi Hotspot Gempa</b><br>
                    <hr style="margin: 5px 0;">
                    <b>Tahun Prediksi:</b> {p['year']}<br>
                    {bulan_info}
                    <hr style="margin: 5px 0;">
                    <b>Koordinat:</b><br>
                    Latitude: {p['lat']:.4f}°<br>
                    Longitude: {p['lon']:.4f}°<br>
                    <hr style="margin: 5px 0;">
                    <b>Magnitudo Estimasi:</b> {p['mag']:.2f}<br>
                    <b>Frekuensi Historis:</b> {p.get('count', 'N/A')} kejadian<br>
                    <hr style="margin: 5px 0;">
                    <small><i>Prediksi lokasi hotspot berdasarkan pola frekuensi gempa 2008-2025<br>Tidak memprediksi waktu spesifik kejadian</i></small>
                </div>
                """
                
                folium.Marker(
                    location=[p["lat"], p["lon"]],
                    icon=folium.Icon(color="blue", icon="star", prefix="fa"),
                    popup=folium.Popup(popup_content_pred, max_width=300),
                    tooltip=f"Hotspot: {p['lat']:.4f}°, {p['lon']:.4f}° | Mag: {p['mag']:.2f}"
                ).add_to(mymap)

    return mymap

# --- Visualisasi historis (peta terpisah) ---
st.subheader("Peta historis 2008-2025")

# Debug info: Tampilkan proses yang berjalan
with st.expander("ℹ️ Info: Proses yang berjalan saat mengubah tahun", expanded=False):
    st.markdown("""
    **Ketika Anda mengubah tahun, proses berikut dijalankan:**
    1. ✅ **Filter data** - Memfilter data gempa berdasarkan tahun (cached untuk performa)
    2. ✅ **Hitung statistik** - Menghitung count, min, max, mean, median (cached)
    3. ✅ **Buat grafik** - Membuat chart distribusi magnitudo dengan Altair
    4. ⚠️ **Buat peta** - Ini proses paling berat:
       - Membuat objek peta Folium
       - Loop melalui semua gempa tahun tersebut
       - Format datetime untuk setiap gempa
       - Buat marker untuk setiap gempa (hingga 5000 marker)
       - Render popup HTML untuk setiap marker
    5. ✅ **Render peta** - Menampilkan peta ke browser dengan st_folium
    
    **Tips performa:**
    - Proses paling lambat adalah pembuatan marker di peta
    - Jika data > 5000 gempa, hanya 5000 gempa terbesar yang ditampilkan
    - Gunakan tahun dengan data lebih sedikit untuk loading lebih cepat
    """)

year = st.selectbox("Pilih Tahun (historis)", range(2008, 2026))

# Gunakan cached function untuk filter data
filtered_data = get_filtered_data_by_year(data, year)

if filtered_data.empty:
    st.warning("Tidak ada data historis untuk tahun yang dipilih.")
    mymap_hist = create_map(
        pd.DataFrame(), 
        predicted_points=None, 
        target_year=year, 
        heatmap_points=None,
        max_markers=5000
    )
else:
    # Statistik singkat - menggunakan cache
    stats = calculate_stats(filtered_data)
    
    # Tampilkan statistik dengan penjelasan
    with st.expander("📊 Penjelasan Statistik", expanded=False):
        st.markdown(f"""
        **Penjelasan Statistik Deskriptif Tahun {year}:**
        
        - **n = {int(stats['count'])}**: Jumlah total kejadian gempa dalam tahun {year}
        - **min = {stats['min']:.2f}**: Magnitudo gempa terkecil yang tercatat
        - **max = {stats['max']:.2f}**: Magnitudo gempa terbesar yang tercatat
        - **mean = {stats['mean']:.2f}**: Magnitudo rata-rata (rerata) semua gempa
        - **median = {stats['median']:.2f}**: Magnitudo median (nilai tengah setelah data diurutkan)
        
        **Interpretasi:**
        - Jika mean > median: Distribusi cenderung miring ke kanan (banyak gempa kecil, beberapa gempa besar)
        - Jika mean < median: Distribusi cenderung miring ke kiri (banyak gempa besar)
        - Jika mean ≈ median: Distribusi relatif simetris
        """)
    
    # Tampilkan statistik dengan penjelasan yang lebih jelas
    st.markdown("### 📊 Statistik Deskriptif")
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    
    with col_stat1:
        st.metric("Jumlah (n)", f"{int(stats['count']):,}", help="Total kejadian gempa di tahun ini")
    with col_stat2:
        st.metric("Min", f"{stats['min']:.2f}", help="Magnitudo gempa terkecil")
    with col_stat3:
        st.metric("Max", f"{stats['max']:.2f}", help="Magnitudo gempa terbesar")
    with col_stat4:
        st.metric("Rata-rata", f"{stats['mean']:.2f}", help="Magnitudo rata-rata (mean)")
    with col_stat5:
        st.metric("Median", f"{stats['median']:.2f}", help="Magnitudo median (nilai tengah)")
    
    # Tampilkan juga dalam format caption untuk konsistensi
    st.caption(
        f"**Ringkasan**: Data tahun {year}: n={int(stats['count'])}, "
        f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
        f"mean={stats['mean']:.2f}, median={stats['median']:.2f}"
    )
    
    # Penjelasan interpretasi
    if stats['mean'] < stats['median']:
        st.info(f"💡 **Interpretasi**: Mean ({stats['mean']:.2f}) < Median ({stats['median']:.2f}) → Distribusi miring ke kiri. Sebagian besar gempa memiliki magnitudo lebih besar dari rata-rata.")
    elif stats['mean'] > stats['median']:
        st.info(f"💡 **Interpretasi**: Mean ({stats['mean']:.2f}) > Median ({stats['median']:.2f}) → Distribusi miring ke kanan. Banyak gempa kecil dengan beberapa gempa besar.")
    else:
        st.info(f"💡 **Interpretasi**: Mean ({stats['mean']:.2f}) ≈ Median ({stats['median']:.2f}) → Distribusi relatif simetris.")

    # Grafik distribusi magnitudo per tahun - hanya render jika perlu
    with st.container():
        hist = (
            alt.Chart(filtered_data)
            .mark_bar(color="#F44336")
            .encode(
                x=alt.X("magnitude:Q", bin=alt.Bin(step=0.2), title="Magnitudo"),
                y=alt.Y("count()", title="Jumlah kejadian"),
                tooltip=[alt.Tooltip("count()", title="Jumlah"), alt.Tooltip("magnitude:Q", bin=True, title="Magnitudo")],
            )
            .properties(width=700, height=300, title=f"Distribusi Magnitudo Tahun {year}")
        )
        st.altair_chart(hist, use_container_width=True)

    # Peta historis - optimasi performa dengan mengurangi marker untuk data banyak
    # Untuk data sangat banyak, kurangi marker lebih agresif
    max_markers_adjusted = 5000
    if len(filtered_data) > 10000:
        max_markers_adjusted = 3000  # Kurangi marker untuk data sangat banyak
    elif len(filtered_data) > 5000:
        max_markers_adjusted = 4000
    
    # Gunakan cached map untuk performa lebih baik - peta akan di-cache per tahun
    # Hitung hash dari data untuk key cache
    data_count = len(filtered_data)
    data_sum = filtered_data['magnitude'].sum() if not filtered_data.empty else 0
    
    # Cek apakah ini pertama kali (belum di-cache) atau sudah di-cache
    # Streamlit akan otomatis cache, jadi pertama kali akan lebih lambat
    # Untuk deteksi cache, kita bisa menggunakan try-except atau langsung panggil
    # Karena st.cache_resource otomatis handle cache, kita tidak perlu deteksi manual
    
    # Tampilkan progress bar hanya jika data banyak (pertama kali akan lebih lambat)
    if data_count > 1000:
        with st.spinner(f"⏳ Memuat peta untuk {data_count:,} gempa tahun {year}..."):
            mymap_hist = create_map_cached(year, data_count, data_sum, max_markers_adjusted)
    else:
        mymap_hist = create_map_cached(year, data_count, data_sum, max_markers_adjusted)
    
    # Tampilkan info bahwa peta di-cache (akan cepat saat switch tahun)
    st.caption(f"💡 **Tip**: Peta tahun {year} di-cache untuk performa lebih baik. Switch tahun akan lebih cepat!")
    
    # Tampilkan info jika data dibatasi
    if len(filtered_data) > max_markers_adjusted:
        st.info(f"💡 **Tip Performa**: Data tahun {year} memiliki {len(filtered_data):,} gempa. Ditampilkan {max_markers_adjusted:,} gempa dengan magnitudo terbesar untuk loading lebih cepat.")

# Peta historis - di tengah halaman
col_map1, col_map2, col_map3 = st.columns([1, 3, 1])
with col_map2:
    st_folium(
        mymap_hist,
        width=900,
        height=600,
        key=f"map_hist_{year}",
        returned_objects=[],
    )

# --- Peta prediksi 2026 (terpisah) ---
st.subheader("Peta prediksi hotspot 2026 (berdasar pola historis)")
pred_data_df = pd.DataFrame(hotspot_points)
if pred_data_df.empty:
    st.warning("Tidak ada titik hotspot yang dapat dihitung.")
    mymap_pred = create_map(pd.DataFrame(columns=filtered_data.columns), predicted_points=None, target_year=year, heatmap_points=None)
else:
    # Penjelasan grid dengan expander
    with st.expander("ℹ️ Penjelasan: Apa itu Grid 0.5°?", expanded=False):
        st.markdown("""
        ### **Grid 0.5° - Penjelasan Spatial Binning**
        
        **Grid 0.5°** berarti wilayah Indonesia dibagi menjadi kotak-kotak (grid) dengan ukuran:
        - **0.5 derajat latitude** (lintang) × **0.5 derajat longitude** (bujur)
        - **≈ 55 km × 55 km** di khatulistiwa (untuk Indonesia)
        
        #### **Cara Kerja:**
        1. **Pembagian Grid**: Setiap koordinat gempa dibulatkan ke grid terdekat
           - Contoh: Gempa di (-6.2°, 106.8°) → masuk ke grid (-6.0°, 107.0°)
        
        2. **Pengelompokan**: Semua gempa dalam grid yang sama dikelompokkan
           - Grid (-6.0°, 107.0°) berisi semua gempa di area tersebut
        
        3. **Perhitungan**: Untuk setiap grid dihitung:
           - **Frekuensi**: Berapa banyak gempa terjadi di grid tersebut
           - **Magnitudo rata-rata**: Rata-rata magnitudo gempa di grid tersebut
        
        #### **Mengapa 0.5°?**
        - ✅ **Ukuran optimal**: Tidak terlalu kecil (detail berlebihan) atau terlalu besar (kurang detail)
        - ✅ **Cocok untuk Indonesia**: ~55 km cukup untuk identifikasi daerah rawan
        - ✅ **Balance**: Keseimbangan antara akurasi dan efisiensi komputasi
        
        #### **Contoh Visual:**
        ```
        Grid 0.5° × 0.5° di Indonesia:
        
        ┌─────────┬─────────┬─────────┐
        │ Grid 1  │ Grid 2  │ Grid 3  │
        │ 0.5°×0.5°│ 0.5°×0.5°│ 0.5°×0.5°│
        │ ≈55×55km│ ≈55×55km│ ≈55×55km│
        ├─────────┼─────────┼─────────┤
        │ Grid 4  │ Grid 5  │ Grid 6  │
        │ 0.5°×0.5°│ 0.5°×0.5°│ 0.5°×0.5°│
        └─────────┴─────────┴─────────┘
        ```
        
        **Catatan**: Ukuran grid bisa diubah, tetapi 0.5° adalah pilihan yang optimal untuk analisis gempa Indonesia.
        """)
    
    st.caption(f"Menampilkan {len(pred_data_df)} hotspot teratas (grid {bin_size}° ≈ 55 km × 55 km).")
    
    # Toggle untuk menampilkan/menyembunyikan grid
    col_grid1, col_grid2 = st.columns([3, 1])
    with col_grid1:
        st.caption("💡")
    with col_grid2:
        show_grid = st.checkbox("📐 Tampilkan Grid", value=True, 
                                help="Tampilkan grid 0.5° × 0.5° untuk memvisualisasikan spatial binning. Grid ini menunjukkan bagaimana wilayah dibagi menjadi kotak-kotak untuk analisis.")
    
    # Buat peta prediksi dengan spinner untuk feedback
    with st.spinner("⏳ Membuat peta prediksi..."):
        mymap_pred = create_map(
            pd.DataFrame(columns=filtered_data.columns),  # tidak pakai data historis di peta prediksi
            predicted_points=hotspot_points,
            target_year=2026,
            heatmap_points=heatmap_points,
            show_grid=show_grid,
            bin_size=bin_size
        )

# Peta prediksi - di tengah halaman
col_pred1, col_pred2, col_pred3 = st.columns([1, 3, 1])
with col_pred2:
    st_folium(
        mymap_pred,
        width=900,
        height=600,
        key="map_pred_2026",
        returned_objects=[],
    )
