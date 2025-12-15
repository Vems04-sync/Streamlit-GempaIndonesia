import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import folium
from folium.plugins import HeatMap
import tensorflow as tf
import json
import altair as alt
import urllib.request
import os

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

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Data Gempa - DEC & Anomaly Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    """Memuat data gempa dari CSV"""
    try:
        df = pd.read_csv("data/katalog_gempa_v2.csv")
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values('datetime').reset_index(drop=True)
        df['year'] = df['datetime'].dt.year
        
        # Preprocessing untuk clustering (sesuai source code)
        df_clustering = df[['year', 'magnitude', 'depth', 'latitude', 'longitude']].copy()
        df_clustering['magnitude'] = df_clustering['magnitude'].interpolate()
        df_clustering['depth'] = df_clustering['depth'].interpolate()
        
        return df, df_clustering
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Fungsi untuk membuat scaler dari data (jika file .pkl tidak ada)
def create_scaler_from_data(df_features):
    """Membuat scaler dari data jika file .pkl tidak ada"""
    from sklearn.preprocessing import StandardScaler
    
    try:
        # Preprocess data
        df_dec_input = df_features[['magnitude', 'depth', 'latitude', 'longitude']].copy()
        df_dec_input = df_dec_input.interpolate()
        df_dec_input.drop_duplicates(inplace=True)
        
        # Buat dan fit scaler
        scaler_dec = StandardScaler()
        scaler_dec.fit(df_dec_input)
        
        return scaler_dec
    except Exception as e:
        return None

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_models():
    """Memuat model autoencoder, encoder, dan scaler"""
    import os
    error_messages = []
    
    try:
        # Load autoencoder model
        autoencoder_path = 'model/autoencoder_model.h5'
        if not os.path.exists(autoencoder_path):
            error_messages.append(f"File tidak ditemukan: {autoencoder_path}")
            return None, None, None, error_messages
        
        try:
            # Coba load dengan compile=False untuk menghindari masalah kompatibilitas metrics
            # Ini adalah solusi untuk error "Could not deserialize 'keras.metrics.mse'"
            autoencoder = load_model(autoencoder_path, compile=False)
        except Exception as e1:
            # Jika masih error, coba dengan custom_objects untuk metrics
            try:
                import tensorflow as tf
                # Buat custom_objects untuk metrics yang mungkin tidak bisa di-deserialize
                custom_objects = {}
                autoencoder = load_model(autoencoder_path, compile=False, custom_objects=custom_objects)
            except Exception as e2:
                # Jika masih error, coba dengan safe_mode=False (untuk TensorFlow 2.13+)
                try:
                    autoencoder = load_model(autoencoder_path, compile=False, safe_mode=False)
                except Exception as e3:
                    # Jika semua gagal, tampilkan error
                    error_msg = f"Error loading autoencoder model. Tried multiple methods but all failed.\nLast error: {str(e3)}"
                    error_messages.append(error_msg)
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    return None, None, None, error_messages
        
        # Load atau buat scaler
        scaler_dec = None
        scaler_path = 'scaler_dec.pkl'
        if os.path.exists(scaler_path):
            try:
                scaler_dec = joblib.load(scaler_path)
            except Exception as e:
                error_msg = f"Error loading scaler_dec.pkl: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
        else:
            error_messages.append(f"File tidak ditemukan: {scaler_path}")
        
        # Membuat encoder model dari autoencoder
        encoder_model = None
        encoder_path = 'encoder_model.h5'
        if os.path.exists(encoder_path):
            try:
                # Coba load dengan compile=False untuk menghindari masalah kompatibilitas
                encoder_model = load_model(encoder_path, compile=False)
            except Exception as e1:
                try:
                    # Coba dengan custom_objects
                    import tensorflow as tf
                    custom_objects = {
                        'mse': tf.keras.metrics.MeanSquaredError(),
                        'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                    }
                    encoder_model = load_model(encoder_path, compile=False, custom_objects=custom_objects)
                except Exception as e2:
                    try:
                        encoder_model = load_model(encoder_path, compile=False, safe_mode=False)
                    except Exception as e3:
                        error_msg = f"Error loading encoder_model.h5, akan dibuat dari autoencoder: {str(e3)}"
                        error_messages.append(error_msg)
                        print(error_msg)
        
        if encoder_model is None:
            try:
                # Buat encoder dari autoencoder
                input_layer = autoencoder.input
                # Latent layer adalah layer ke-5 (index 4) dalam autoencoder
                latent_layer = autoencoder.layers[4].output
                encoder_model = Model(inputs=input_layer, outputs=latent_layer)
            except Exception as e:
                error_msg = f"Error creating encoder model: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
                return scaler_dec, autoencoder, None, error_messages
        
        return scaler_dec, autoencoder, encoder_model, error_messages
    except Exception as e:
        error_msg = f"Unexpected error loading models: {str(e)}"
        error_messages.append(error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None, error_messages

# Fungsi untuk menghitung threshold anomali secara dinamis
@st.cache_data
def calculate_threshold(_scaler_dec, _autoencoder, df_features):
    """Menghitung threshold anomali menggunakan mean + 3*std"""
    try:
        # Preprocess data
        df_dec_input = df_features[['magnitude', 'depth', 'latitude', 'longitude']].copy()
        df_dec_input = df_dec_input.interpolate()
        df_dec_input.drop_duplicates(inplace=True)
        
        # Scale data
        X_dec = _scaler_dec.transform(df_dec_input)
        
        # Rekonstruksi data
        X_dec_pred = _autoencoder.predict(X_dec, verbose=0)
        
        # Hitung error MSE per sampel
        reconstruction_error = np.mean(np.square(X_dec - X_dec_pred), axis=1)
        
        # Threshold = mean + 3*std
        threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)
        
        return threshold
    except Exception as e:
        st.warning(f"Error calculating threshold: {e}")
        return 0.0001606588873608411  # Default threshold

# Fungsi untuk melakukan DEC clustering per tahun
@st.cache_data
def perform_dec_clustering(_scaler_dec, _encoder_model, df_clustering, n_clusters=2):
    """Melakukan DEC clustering per tahun"""
    try:
        # Preprocess data untuk DEC (sesuai source code)
        df_dec_input = df_clustering[['magnitude', 'depth', 'latitude', 'longitude']].copy()
        df_dec_input = df_dec_input.interpolate()
        
        # Simpan index sebelum drop_duplicates untuk alignment
        df_dec_input_with_meta = df_clustering[['year', 'magnitude', 'depth', 'latitude', 'longitude']].copy()
        df_dec_input_with_meta = df_dec_input_with_meta.interpolate()
        df_dec_input_with_meta.drop_duplicates(subset=['magnitude', 'depth', 'latitude', 'longitude'], inplace=True)
        
        # Scale data (hanya features, tanpa year)
        features_for_scaling = df_dec_input_with_meta[['magnitude', 'depth', 'latitude', 'longitude']]
        X_dec = _scaler_dec.transform(features_for_scaling)
        
        # Ekstrak latent space
        Z = _encoder_model.predict(X_dec, verbose=0)
        
        # Buat DataFrame latent features
        latent_col_names = [f'latent_feature_{i}' for i in range(Z.shape[1])]
        df_Z = pd.DataFrame(Z, columns=latent_col_names)
        
        # Gabungkan dengan year, latitude, longitude, magnitude, depth (indices sudah align karena drop_duplicates dilakukan pada DataFrame yang sama)
        df_latent_features = pd.concat(
            [df_dec_input_with_meta[['year', 'magnitude', 'depth', 'latitude', 'longitude']].reset_index(drop=True), df_Z.reset_index(drop=True)],
            axis=1
        )
        
        # Clustering per tahun
        df_clustered_by_year = {}
        years = df_latent_features['year'].unique()
        
        for year in years:
            df_year_latent = df_latent_features[df_latent_features['year'] == year].copy()
            features_for_clustering = df_year_latent[latent_col_names]
            
            if len(features_for_clustering) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(features_for_clustering)
                df_year_latent['cluster_label'] = cluster_labels
                df_clustered_by_year[year] = df_year_latent
        
        return df_clustered_by_year, latent_col_names
    except Exception as e:
        st.error(f"Error performing DEC clustering: {e}")
        return {}, []

# Fungsi untuk menghitung silhouette score per tahun
@st.cache_data
def calculate_silhouette_scores(df_clustered_by_year, latent_col_names):
    """Menghitung silhouette score untuk setiap tahun"""
    yearly_silhouette_scores = {}
    
    for year, yearly_df in df_clustered_by_year.items():
        features_for_silhouette = yearly_df[latent_col_names]
        cluster_labels = yearly_df['cluster_label']
        
        if len(np.unique(cluster_labels)) > 1 and len(cluster_labels) > 1:
            silhouette_avg = silhouette_score(features_for_silhouette, cluster_labels)
            yearly_silhouette_scores[year] = silhouette_avg
        else:
            yearly_silhouette_scores[year] = None
    
    return yearly_silhouette_scores

# Sidebar untuk navigasi
st.sidebar.title("🌍 Menu Navigasi")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Dashboard", "EDA - Exploratory Data Analysis", "DEC Clustering", "Deteksi Anomali", "Data Exploration", "Prediksi Gempa"]
)

# Memuat data
df, df_clustering = load_data()

if df is None or df_clustering is None:
    st.error("Tidak dapat memuat data. Pastikan file data/katalog_gempa_v2.csv ada.")
    st.stop()

# Memuat model
scaler_dec, autoencoder, encoder_model, load_errors = load_models()

# Jika scaler tidak ada tapi autoencoder ada, buat scaler on-the-fly
# (Hanya dilakukan sekali, hasilnya di-cache)
if scaler_dec is None and autoencoder is not None and df_clustering is not None:
    scaler_dec = create_scaler_from_data(df_clustering)
    if scaler_dec is not None:
        # Simpan scaler untuk penggunaan berikutnya (opsional, tidak blocking)
        try:
            joblib.dump(scaler_dec, 'scaler_dec.pkl')
        except:
            pass  # Jika tidak bisa save, tidak masalah, tetap bisa digunakan

# Hitung threshold secara dinamis jika model tersedia
if scaler_dec is not None and autoencoder is not None:
    threshold = calculate_threshold(scaler_dec, autoencoder, df_clustering)
else:
    threshold = 0.0001606588873608411  # Default threshold

# ============================================================================
# FUNGSI-FUNGSI UNTUK PREDIKSI GEMPA (LSTM)
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

def save_lstm_model(model, filepath="model/lstm_earthquake_model.h5"):
    """
    Menyimpan model LSTM ke file.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)
    return filepath

def save_prediction_results(hotspot_points, method, filepath="model/prediction_results.json"):
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

def load_prediction_results(filepath="model/prediction_results.json"):
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

def download_file_from_google_drive(file_id, destination_path):
    """
    Mengunduh file dari Google Drive menggunakan file ID.
    Menangani file besar yang memerlukan konfirmasi virus scan.
    
    Args:
        file_id: ID file dari Google Drive (dari URL: /d/FILE_ID/view)
        destination_path: Path lokal untuk menyimpan file
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # URL untuk mengunduh file dari Google Drive
        # Coba dengan confirm=t untuk file besar
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        
        # Unduh file
        urllib.request.urlretrieve(url, destination_path)
        
        # Verifikasi file berhasil diunduh (cek ukuran file)
        if os.path.exists(destination_path):
            file_size = os.path.getsize(destination_path)
            # Jika file terlalu kecil (< 100 bytes), mungkin HTML error page
            if file_size < 100:
                # Coba metode alternatif tanpa confirm
                url_alt = f"https://drive.google.com/uc?export=download&id={file_id}"
                urllib.request.urlretrieve(url_alt, destination_path)
                file_size = os.path.getsize(destination_path)
                if file_size < 100:
                    st.error("File yang diunduh terlalu kecil, mungkin terjadi error")
                    return False
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Error mengunduh file dari Google Drive: {e}")
        return False

@st.cache_resource
def load_rf_temporal_model_from_drive():
    """
    Memuat model earthquake_rf_temporal.pkl dari Google Drive.
    Model di-cache untuk menghindari download berulang.
    
    Model akan diunduh dari:
    https://drive.google.com/file/d/1BoYnBU_rt4ZBluDn9mK-Y1j53_Gpmo6B/view?usp=sharing
    
    Returns:
        model: Model yang dimuat dari Google Drive, atau None jika gagal
    
    Contoh penggunaan:
        rf_model = load_rf_temporal_model_from_drive()
        if rf_model is not None:
            predictions = rf_model.predict(X_test)
    """
    # File ID dari Google Drive URL
    # URL: https://drive.google.com/file/d/1BoYnBU_rt4ZBluDn9mK-Y1j53_Gpmo6B/view?usp=sharing
    file_id = "1BoYnBU_rt4ZBluDn9mK-Y1j53_Gpmo6B"
    
    # Buat direktori model jika belum ada
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    # Path lokal untuk menyimpan file
    local_path = model_dir / "earthquake_rf_temporal.pkl"
    
    # Jika file sudah ada, langsung load
    if local_path.exists():
        try:
            model = joblib.load(local_path)
            return model
        except Exception as e:
            st.warning(f"File ada tapi gagal dimuat: {e}. Mencoba download ulang...")
            local_path.unlink()  # Hapus file yang corrupt
    
    # Jika file tidak ada, download dari Google Drive
    with st.spinner("⏳ Mengunduh model dari Google Drive..."):
        if download_file_from_google_drive(file_id, str(local_path)):
            try:
                model = joblib.load(local_path)
                st.success("✅ Model berhasil diunduh dan dimuat dari Google Drive!")
                return model
            except Exception as e:
                st.error(f"Gagal memuat model setelah download: {e}")
                return None
        else:
            return None

def load_lstm_model(filepath="model/lstm_earthquake_model.h5"):
    """
    Memuat model LSTM dari file jika ada.
    """
    if Path(filepath).exists():
        try:
            model = load_model(filepath, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            try:
                model = load_model(filepath, compile=False, safe_mode=False)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                return model
            except Exception as e2:
                st.warning(f"Gagal memuat model: {e2}")
                return None
    return None

def estimate_training_time(time_series_data, epochs=30, batch_size=32):
    """
    Mengestimasi waktu training berdasarkan data yang ada.
    """
    if time_series_data is None or len(time_series_data) == 0:
        return None
    
    sequence_length = 5
    total_sequences = 0
    
    for series in time_series_data:
        total_sequences += max(0, len(series) - sequence_length)
    
    batches_per_epoch = max(1, total_sequences // batch_size)
    total_batches = batches_per_epoch * epochs
    
    time_per_batch_cpu = 0.4
    time_per_batch_gpu = 0.08
    
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
        sequence_length = 5
        X_all, y_all = [], []
        
        for series in time_series_data:
            series_2d = series.reshape(-1, 1)
            scaler_local = MinMaxScaler()
            scaled = scaler_local.fit_transform(series_2d).flatten()
            
            for i in range(len(scaled) - sequence_length):
                X_all.append(scaled[i:i+sequence_length])
                y_all.append(scaled[i+sequence_length])
        
        if len(X_all) == 0:
            return None, None, None
        
        X = np.array(X_all)
        y = np.array(y_all)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        model = build_lstm_model((sequence_length, 1), units=50)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),
            validation_data=(X_test, y_test) if len(X_test) > 0 else None,
            verbose=0
        )
        
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
        
        # Hitung total frekuensi historis 2008-2025 (semua tahun kecuali 2026)
        # series berisi frekuensi per tahun dari 2008-2025 (18 tahun)
        historical_count = int(sum(series))  # Total frekuensi 2008-2025
        
        series_2d = series.reshape(-1, 1)
        scaler_local = MinMaxScaler()
        scaled = scaler_local.fit_transform(series_2d).flatten()
        
        if len(scaled) >= sequence_length:
            last_sequence = scaled[-sequence_length:].reshape(1, sequence_length, 1)
            
            try:
                pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
                pred_2d = np.array([[pred_scaled]])
                pred_denorm = scaler_local.inverse_transform(pred_2d)[0][0]
                pred_frequency = max(0, int(round(pred_denorm)))
                
                predictions.append({
                    "lat": lat_bin,
                    "lon": lon_bin,
                    "count": pred_frequency,  # Prediksi frekuensi 2026
                    "historical_count": historical_count,  # Total frekuensi historis 2008-2025
                    "mag": mean_mag,
                    "year": target_year,
                    "month": None,
                })
            except Exception as e:
                predictions.append({
                    "lat": lat_bin,
                    "lon": lon_bin,
                    "count": int(series[-1]) if len(series) > 0 else 0,
                    "historical_count": historical_count,
                    "mag": mean_mag,
                    "year": target_year,
                    "month": None,
                })
    
    # Sort berdasarkan prediksi frekuensi tertinggi
    predictions.sort(key=lambda x: x["count"], reverse=True)
    return predictions[:top_k]

def build_hotspot_points(df: pd.DataFrame, bin_size=0.5, top_k=15):
    """
    Membangun titik hotspot prediksi berdasarkan frekuensi gempa historis.
    
    Metodologi (sesuai prediksiLSTM/app.py):
    1. Spatial Binning: Membagi koordinat menjadi grid berukuran bin_size (0.5°)
    2. Grouping: Mengelompokkan semua gempa ke dalam grid yang sama
    3. Aggregation: Menghitung count (frekuensi) dan mean_mag (magnitudo rata-rata)
    4. Ranking: Mengurutkan grid berdasarkan frekuensi tertinggi
    5. Selection: Memilih top_k grid dengan frekuensi tertinggi sebagai hotspot
    
    Catatan: Data yang dikirim ke fungsi ini seharusnya sudah difilter tahun 2008-2025
    sebelum dipanggil, jadi tidak perlu filter tahun lagi di sini.
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
    target_year_pred = 2026
    return [
        {
            "lat": r.lat_bin,
            "lon": r.lon_bin,
            "count": int(r.count),        # Frekuensi gempa historis
            "historical_count": int(r.count),  # Sama dengan count untuk konsistensi
            "mag": float(r.mean_mag),     # Magnitudo rata-rata historis
            "year": target_year_pred,     # Tahun prediksi
            "month": None,                # Tidak relevan untuk prediksi lokasi
        }
        for r in agg.itertuples()
    ]

def build_heatmap_points(df: pd.DataFrame, bin_size=0.5):
    """Membangun titik heatmap untuk visualisasi kerawanan."""
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
    agg["weight"] = 0.8 * (agg["count"] / max_count) + 0.2 * (agg["mean_mag"] / agg["mean_mag"].max())
    return agg[["lat_bin", "lon_bin", "weight"]].values.tolist()

def prepare_prediction_chart_data(df: pd.DataFrame, hotspot_points=None, method="Frequency-Based"):
    """
    Mempersiapkan data untuk plot time series frekuensi gempa 2008-2026.
    
    Returns:
        DataFrame dengan kolom: year, frequency, type (Historical/Predicted)
    """
    # Filter data 2008-2025
    if 'year' in df.columns:
        df_historical = df[(df['year'] >= 2008) & (df['year'] <= 2025)].copy()
    else:
        df_historical = df.copy()
    
    # Hitung frekuensi per tahun (2008-2025)
    if not df_historical.empty and 'year' in df_historical.columns:
        yearly_freq = df_historical.groupby('year').size().reset_index(name='frequency')
        yearly_freq.columns = ['year', 'frequency']
        yearly_freq['type'] = 'Historical'
    else:
        yearly_freq = pd.DataFrame(columns=['year', 'frequency', 'type'])
    
    # Buat DataFrame lengkap dari 2008-2026
    all_years = pd.DataFrame({'year': range(2008, 2027)})
    chart_data = all_years.merge(yearly_freq, on='year', how='left')
    chart_data['frequency'] = chart_data['frequency'].fillna(0).astype(int)
    chart_data['type'] = chart_data['type'].fillna('Historical')
    
    # Hitung prediksi 2026
    # Catatan: hotspot_points hanya menunjukkan top-k hotspot, bukan total gempa
    # Untuk chart, kita estimasi total gempa 2026 menggunakan metode yang sesuai
    if not yearly_freq.empty:
        if method == "LSTM (Deep Learning)":
            # Untuk LSTM: estimasi menggunakan trend dari data historis
            # Gunakan rata-rata 3-5 tahun terakhir dengan sedikit adjustment berdasarkan trend
            if len(yearly_freq) >= 5:
                recent_years = yearly_freq.tail(5)['frequency']
                # Hitung trend (linear regression sederhana)
                years_recent = np.arange(len(recent_years))
                if len(recent_years) > 1 and recent_years.std() > 0:
                    trend = np.polyfit(years_recent, recent_years.values, 1)[0]
                    pred_2026 = int(round(recent_years.mean() + trend))
                else:
                    pred_2026 = int(round(recent_years.mean()))
            else:
                pred_2026 = int(round(yearly_freq['frequency'].mean()))
        else:
            # Frequency-Based: prediksi menggunakan rata-rata frekuensi 5 tahun terakhir
            if len(yearly_freq) >= 5:
                last_5_years = yearly_freq.tail(5)['frequency'].mean()
                pred_2026 = int(round(last_5_years))
            else:
                pred_2026 = int(round(yearly_freq['frequency'].mean()))
    else:
        pred_2026 = 0
    
    # Tambahkan prediksi 2026 (jika belum ada di chart_data atau replace yang ada)
    if 2026 in chart_data['year'].values:
        chart_data.loc[chart_data['year'] == 2026, 'frequency'] = pred_2026
        chart_data.loc[chart_data['year'] == 2026, 'type'] = 'Predicted'
    else:
        pred_row = pd.DataFrame({
            'year': [2026],
            'frequency': [pred_2026],
            'type': ['Predicted']
        })
        chart_data = pd.concat([chart_data, pred_row], ignore_index=True)
    
    return chart_data

def add_grid_to_map(mymap, bin_size=0.5, bounds=None, max_grid_lines=50):
    """Menambahkan grid layer ke peta untuk visualisasi spatial binning."""
    if bounds is None:
        min_lat, min_lon = -11.0, 95.0
        max_lat, max_lon = 6.0, 141.0
    else:
        min_lat, min_lon, max_lat, max_lon = bounds
    
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    num_lat_lines = int(lat_range / bin_size) + 1
    num_lon_lines = int(lon_range / bin_size) + 1
    total_lines = num_lat_lines + num_lon_lines
    
    if total_lines > max_grid_lines:
        grid_group = folium.FeatureGroup(name="Grid", show=True)
        padding = min(lat_range * 0.1, lon_range * 0.1, 2.0)
        min_lat_limited = min_lat - padding
        max_lat_limited = max_lat + padding
        min_lon_limited = min_lon - padding
        max_lon_limited = max_lon + padding
        
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
        grid_group = folium.FeatureGroup(name="Grid")
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

def create_prediction_map(data, predicted_points=None, target_year=None, heatmap_points=None, max_markers=5000, show_grid=False, bin_size=0.5):
    """Membuat peta interaktif untuk prediksi gempa."""
    map_center = [-5.0, 115.0]
    zoom_start = 5
    
    mymap = folium.Map(location=map_center, zoom_start=zoom_start)
    
    if show_grid:
        # Tentukan bounds untuk grid
        if predicted_points and len(predicted_points) > 0:
            lats = [p['lat'] for p in predicted_points]
            lons = [p['lon'] for p in predicted_points]
            padding = bin_size * 5  # Tambah padding lebih besar
            bounds = (min(lats) - padding, min(lons) - padding, 
                     max(lats) + padding, max(lons) + padding)
        elif not data.empty and len(data) > 0:
            padding = bin_size * 5
            bounds = (data['latitude'].min() - padding, data['longitude'].min() - padding,
                     data['latitude'].max() + padding, data['longitude'].max() + padding)
        else:
            # Gunakan bounds Indonesia jika tidak ada data
            bounds = (-11.0, 95.0, 6.0, 141.0)
        
        mymap = add_grid_to_map(mymap, bin_size=bin_size, bounds=bounds, max_grid_lines=100)

    data_to_plot = data.copy()
    if len(data_to_plot) > max_markers:
        data_to_plot = data_to_plot.nlargest(max_markers, 'magnitude')
    
    if not data_to_plot.empty and 'datetime' in data_to_plot.columns:
        data_to_plot = data_to_plot.copy()
        mask_notna = data_to_plot['datetime'].notna()
        data_to_plot.loc[mask_notna, 'waktu_str'] = data_to_plot.loc[mask_notna, 'datetime'].dt.strftime("%d %B %Y, %H:%M:%S WIB")
        data_to_plot.loc[~mask_notna, 'waktu_str'] = "Tidak tersedia"
        data_to_plot.loc[mask_notna, 'tanggal_str'] = data_to_plot.loc[mask_notna, 'datetime'].dt.strftime("%Y-%m-%d")
        data_to_plot.loc[~mask_notna, 'tanggal_str'] = "N/A"
        data_to_plot.loc[mask_notna, 'jam_str'] = data_to_plot.loc[mask_notna, 'datetime'].dt.strftime("%H:%M:%S")
        data_to_plot.loc[~mask_notna, 'jam_str'] = "N/A"
    
    for row in data_to_plot.itertuples():
        magnitude = getattr(row, 'magnitude', 0)
        latitude = getattr(row, 'latitude', 0)
        longitude = getattr(row, 'longitude', 0)
        location = getattr(row, 'location', 'N/A')
        
        if hasattr(row, 'waktu_str'):
            waktu_str = row.waktu_str
            tanggal_str = row.tanggal_str
            jam_str = row.jam_str
        else:
            dt_val = getattr(row, 'datetime', None)
            if pd.notna(dt_val):
                waktu_str = dt_val.strftime("%d %B %Y, %H:%M:%S WIB")
                tanggal_str = dt_val.strftime("%Y-%m-%d")
                jam_str = dt_val.strftime("%H:%M:%S")
            else:
                waktu_str = "Tidak tersedia"
                tanggal_str = "N/A"
                jam_str = "N/A"
        
        popup_content = f"""<div style="font-family:Arial;min-width:200px;"><b>📍 Info Gempa</b><br><hr style="margin:3px 0;"><b>Waktu:</b> {waktu_str}<br><b>Tanggal:</b> {tanggal_str}<br><b>Jam:</b> {jam_str}<br><hr style="margin:3px 0;"><b>Magnitudo:</b> {magnitude:.2f}<br><b>Lokasi:</b> {location}<br><b>Koordinat:</b> ({latitude:.4f}, {longitude:.4f})</div>"""
        
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=280),
            tooltip=f"{magnitude:.1f}"
        ).add_to(mymap)
    
    if predicted_points:
        for p in predicted_points:
            if target_year is None or p.get("year") == target_year:
                bulan_info = f"<b>Bulan:</b> {p['month']:02d}<br>" if p.get('month') is not None else ""
                
                # Tentukan apakah ini prediksi LSTM atau Frequency-Based
                # Sesuai prediksiLSTM/app.py: untuk LSTM, count = prediksi 2026, historical_count = total historis
                # Untuk Frequency-Based, count = historical_count = frekuensi historis
                historical_count = p.get('historical_count', None)
                pred_count = p.get('count', 0)
                
                # Jika historical_count ada dan berbeda dengan count, berarti ini LSTM
                if historical_count is not None and historical_count != pred_count:
                    # Ini prediksi LSTM - tampilkan kedua informasi
                    freq_info = f"""
                    <b>Frekuensi Historis (2008-2025):</b> {historical_count} kejadian<br>
                    <b>Prediksi Frekuensi 2026:</b> {pred_count} kejadian<br>
                    """
                else:
                    # Ini Frequency-Based - count = historical_count = frekuensi historis
                    # Sesuai prediksiLSTM/app.py: hanya tampilkan "Frekuensi Historis"
                    freq_info = f"""
                    <b>Frekuensi Historis:</b> {pred_count} kejadian<br>
                    """
                
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
                    {freq_info}
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
    
    # Tambahkan heatmap jika disediakan
    if heatmap_points and len(heatmap_points) > 0:
        HeatMap(heatmap_points, radius=15, blur=10, max_zoom=1).add_to(mymap)

    return mymap

@st.cache_data
def get_filtered_data_by_year_prediction(_data, year):
    """Cache filtered data berdasarkan tahun untuk prediksi."""
    return _data[_data['year'] == year].copy()

@st.cache_data
def calculate_stats_prediction(_filtered_data):
    """Menghitung statistik deskriptif untuk data magnitudo gempa."""
    return _filtered_data['magnitude'].agg(['count', 'min', 'max', 'mean', 'median'])

# ==================== DASHBOARD ====================
if page == "Dashboard":
    st.markdown('<h1 class="main-header">📊 Dashboard Analisis Data Gempa</h1>', unsafe_allow_html=True)
    
    # Statistik utama
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Gempa", f"{len(df):,}")
    
    with col2:
        st.metric("Rentang Tahun", f"{df['year'].min()} - {df['year'].max()}")
    
    with col3:
        avg_magnitude = df['magnitude'].mean()
        st.metric("Rata-rata Magnitudo", f"{avg_magnitude:.2f}")
    
    with col4:
        avg_depth = df['depth'].mean()
        st.metric("Rata-rata Kedalaman", f"{avg_depth:.1f} km")
    
    st.divider()
    
    # Grafik ringkasan
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Magnitudo per Tahun")
        yearly_mag = df.groupby('year')['magnitude'].mean().reset_index()
        fig = px.line(yearly_mag, x='year', y='magnitude', 
                     title='Rata-rata Magnitudo per Tahun',
                     labels={'year': 'Tahun', 'magnitude': 'Magnitudo'})
        fig.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribusi Magnitudo")
        fig = px.histogram(df, x='magnitude', nbins=50,
                          title='Distribusi Magnitudo Gempa',
                          labels={'magnitude': 'Magnitudo', 'count': 'Frekuensi'})
        fig.update_traces(marker_color='#ff7f0e')
        st.plotly_chart(fig, use_container_width=True)
    
    # Peta sebaran gempa
    st.subheader("🗺️ Peta Sebaran Lokasi Gempa")
    
    # Sampling data untuk performa (ambil 10% data)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Hapus baris dengan NaN pada kolom yang diperlukan untuk visualisasi
    df_sample_clean = df_sample.dropna(subset=['magnitude', 'latitude', 'longitude'])
    
    if len(df_sample_clean) > 0:
        fig = px.scatter_mapbox(
            df_sample_clean,
            lat="latitude",
            lon="longitude",
            color="magnitude",
            size="magnitude",
            hover_data=["depth", "datetime"],
            color_continuous_scale="Viridis",
            zoom=4,
            height=600,
            mapbox_style="open-street-map"
        )
        fig.update_layout(title="Sebaran Lokasi Gempa (Sample)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada data yang valid untuk visualisasi peta (semua data memiliki NaN).")

# ==================== EDA ====================
elif page == "EDA - Exploratory Data Analysis":
    st.markdown('<h1 class="main-header">📈 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    # Pastikan data tersedia
    if df is None or len(df) == 0:
        st.error("Data tidak tersedia. Silakan refresh halaman.")
        st.stop()
    
    # Validasi kolom yang diperlukan
    required_columns = ['magnitude', 'depth', 'latitude', 'longitude', 'datetime', 'year']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_columns)}")
        st.stop()
    
    # Tab untuk berbagai visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribusi Magnitudo", 
        "Distribusi Kedalaman",
        "Sebaran Lokasi",
        "Magnitudo vs Waktu",
        "Korelasi Depth vs Magnitude"
    ])
    
    with tab1:
        try:
            st.subheader("Distribusi Magnitudo Gempa")
            # Hapus NaN untuk visualisasi
            df_clean_mag = df.dropna(subset=['magnitude'])
            if len(df_clean_mag) > 0:
                fig = px.histogram(df_clean_mag, x='magnitude', nbins=50,
                                  title='Distribusi Magnitudo Gempa',
                                  labels={'magnitude': 'Magnitudo', 'count': 'Frekuensi'},
                                  histnorm='probability density')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada data yang valid untuk visualisasi.")
            
            st.write("**Statistik Deskriptif:**")
            st.dataframe(df['magnitude'].describe())
        except Exception as e:
            st.error(f"Error di tab Distribusi Magnitudo: {str(e)}")
            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())
    
    with tab2:
        try:
            st.subheader("Distribusi Kedalaman Gempa")
            # Hapus NaN untuk visualisasi
            df_clean_depth = df.dropna(subset=['depth'])
            if len(df_clean_depth) > 0:
                fig = px.histogram(df_clean_depth, x='depth', nbins=50,
                                  title='Distribusi Kedalaman Gempa',
                                  labels={'depth': 'Kedalaman (km)', 'count': 'Frekuensi'},
                                  color_discrete_sequence=['orange'],
                                  histnorm='probability density')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada data yang valid untuk visualisasi.")
            
            st.write("**Statistik Deskriptif:**")
            st.dataframe(df['depth'].describe())
        except Exception as e:
            st.error(f"Error di tab Distribusi Kedalaman: {str(e)}")
            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())
    
    with tab3:
        try:
            st.subheader("Sebaran Lokasi Gempa")
            
            # Filter berdasarkan tahun
            year_range = st.slider(
                "Pilih Rentang Tahun",
                min_value=int(df['year'].min()),
                max_value=int(df['year'].max()),
                value=(int(df['year'].min()), int(df['year'].max()))
            )
            
            df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
            df_sample = df_filtered.sample(n=min(5000, len(df_filtered)), random_state=42)
            
            # Hapus baris dengan NaN pada kolom yang diperlukan
            df_sample_clean = df_sample.dropna(subset=['magnitude', 'latitude', 'longitude'])
            
            if len(df_sample_clean) > 0:
                fig = px.scatter(
                    df_sample_clean,
                    x='longitude',
                    y='latitude',
                    color='magnitude',
                    size='magnitude',
                    hover_data=['depth', 'datetime'],
                    title='Sebaran Lokasi Gempa',
                    labels={'longitude': 'Longitude', 'latitude': 'Latitude'},
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada data yang valid untuk visualisasi (semua data memiliki NaN).")
        except Exception as e:
            st.error(f"Error di tab Sebaran Lokasi: {str(e)}")
            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())
    
    with tab4:
        try:
            st.subheader("Magnitudo Gempa terhadap Waktu")
            
            # Pastikan kolom datetime ada dan valid
            df_clean_time = df.dropna(subset=['datetime', 'magnitude']).copy()
            if len(df_clean_time) == 0:
                st.warning("Tidak ada data yang valid untuk visualisasi waktu.")
            else:
                # Pastikan datetime adalah datetime type
                try:
                    # Coba convert ke datetime jika belum
                    if not str(df_clean_time['datetime'].dtype).startswith('datetime'):
                        df_clean_time['datetime'] = pd.to_datetime(df_clean_time['datetime'], errors='coerce')
                    df_clean_time = df_clean_time.dropna(subset=['datetime'])
                except Exception as e:
                    st.warning(f"Error memproses datetime: {e}")
                    df_clean_time = pd.DataFrame()  # Set empty jika error
                
                if len(df_clean_time) > 0:
                    # Agregasi per bulan untuk performa
                    # Buat copy untuk menghindari modifikasi DataFrame yang di-cache
                    df_temp = df_clean_time.copy()
                    df_temp['year_month'] = df_temp['datetime'].dt.to_period('M').astype(str)
                    monthly_avg = df_temp.groupby('year_month')['magnitude'].mean().reset_index()
                    monthly_avg['year_month'] = pd.to_datetime(monthly_avg['year_month'], errors='coerce')
                    monthly_avg = monthly_avg.dropna(subset=['year_month'])
                    
                    if len(monthly_avg) > 0:
                        fig = px.line(
                            monthly_avg,
                            x='year_month',
                            y='magnitude',
                            title='Rata-rata Magnitudo Gempa per Bulan',
                            labels={'year_month': 'Tahun', 'magnitude': 'Magnitudo'}
                        )
                        fig.update_traces(line_width=2)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Tidak ada data yang valid untuk visualisasi waktu setelah agregasi.")
                else:
                    st.warning("Tidak ada data datetime yang valid.")
        except Exception as e:
            st.error(f"Error di tab Magnitudo vs Waktu: {str(e)}")
            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())
    
    with tab5:
        try:
            st.subheader("Korelasi Kedalaman vs Magnitudo")
            
            df_sample = df.sample(n=min(10000, len(df)), random_state=42)
            
            # Hapus baris dengan NaN pada kolom yang diperlukan
            df_sample_clean = df_sample.dropna(subset=['depth', 'magnitude'])
            
            if len(df_sample_clean) > 0:
                fig = px.scatter(
                    df_sample_clean,
                    x='depth',
                    y='magnitude',
                    title='Korelasi Kedalaman vs Magnitudo',
                    labels={'depth': 'Kedalaman (km)', 'magnitude': 'Magnitudo'},
                    opacity=0.3,
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Korelasi
                correlation = df_sample_clean[['depth', 'magnitude']].corr().iloc[0, 1]
                st.metric("Koefisien Korelasi", f"{correlation:.4f}")
            else:
                st.warning("Tidak ada data yang valid untuk visualisasi (semua data memiliki NaN).")
        except Exception as e:
            st.error(f"Error di tab Korelasi Depth vs Magnitude: {str(e)}")
            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())

# ==================== DEC CLUSTERING ====================
elif page == "DEC Clustering":
    st.markdown('<h1 class="main-header">🔬 Deep Embedded Clustering (DEC)</h1>', unsafe_allow_html=True)
    
    # Debug info
    import os
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Debug Info")
    st.sidebar.write(f"encoder_model: {'✅ Ada' if encoder_model is not None else '❌ None'}")
    st.sidebar.write(f"scaler_dec: {'✅ Ada' if scaler_dec is not None else '❌ None'}")
    st.sidebar.write(f"autoencoder: {'✅ Ada' if autoencoder is not None else '❌ None'}")
    
    # Tampilkan error jika ada
    if load_errors:
        with st.sidebar.expander("⚠️ Error Details"):
            for error in load_errors:
                st.sidebar.text(error)
    
    # Tombol untuk clear cache
    if st.sidebar.button("🔄 Clear Cache & Reload Models"):
        st.cache_resource.clear()
        st.rerun()
    
    if encoder_model is None or scaler_dec is None:
        st.error("⚠️ Model atau scaler tidak ditemukan!")
        
        # Cek file yang ada dan tidak ada
        missing_files = []
        existing_files = []
        
        if not os.path.exists('scaler_dec.pkl'):
            missing_files.append("scaler_dec.pkl")
        else:
            existing_files.append("scaler_dec.pkl")
        
        if not os.path.exists('model/autoencoder_model.h5'):
            missing_files.append("model/autoencoder_model.h5")
        else:
            existing_files.append("model/autoencoder_model.h5")
        
        if os.path.exists('encoder_model.h5'):
            existing_files.append("encoder_model.h5")
        else:
            st.info("ℹ️ encoder_model.h5 tidak ditemukan, akan dibuat dari autoencoder")
        
        if missing_files:
            st.warning("**File yang tidak ditemukan:**")
            for file in missing_files:
                st.write(f"  - ❌ {file}")
        
        if existing_files:
            st.success("**File yang sudah ada:**")
            for file in existing_files:
                st.write(f"  - ✅ {file}")
        
        # Tampilkan error messages jika ada
        if load_errors:
            st.error("**Error saat memuat model:**")
            for error in load_errors:
                st.code(error, language=None)
        
        # Coba reload jika file ada tapi model None
        if existing_files and (encoder_model is None or scaler_dec is None):
            st.info("💡 File sudah ada, tetapi model belum dimuat. Coba klik tombol 'Clear Cache & Reload Models' di sidebar.")
            st.info("💡 Jika ada error di atas, perbaiki masalah tersebut terlebih dahulu.")
        
        st.divider()
        st.subheader("📝 Penjelasan Format File")
        
        st.markdown("""
        **Perbedaan Format:**
        
        - **`autoencoder_model.h5`** (Format HDF5):
          - Model neural network dari Keras/TensorFlow
          - Digunakan untuk rekonstruksi dan ekstraksi fitur
          - ✅ File ini sudah ada
        
        - **`scaler_dec.pkl`** (Format Pickle/Joblib):
          - Preprocessing scaler dari scikit-learn (StandardScaler)
          - Digunakan untuk menormalisasi data sebelum masuk ke model
          - ❌ File ini belum ada
        
        **Mengapa Harus .pkl?**
        - Format `.pkl` adalah standar untuk menyimpan objek Python (scaler, model sklearn, dll)
        - Format `.h5` hanya untuk model Keras/TensorFlow
        - Scaler dan Model adalah dua komponen berbeda yang tidak bisa saling menggantikan
        - Scaler diperlukan untuk preprocessing data sebelum masuk ke model
        
        **Alur Kerja:**
        ```
        Data → Scaler (.pkl) → Normalisasi → Model (.h5) → Hasil
        ```
        """)
        
        st.divider()
        st.subheader("🔧 Cara Membuat File yang Hilang")
        
        st.markdown("""
        **Opsi 1: Menggunakan Script Helper (Paling Mudah)**
        
        Jalankan script berikut di terminal/command prompt:
        ```bash
        python generate_scaler.py
        ```
        
        Script ini akan membuat file `scaler_dec.pkl` secara otomatis.
        
        ---
        
        **Opsi 2: Membuat Scaler Manual**
        
        Jika Anda ingin membuat scaler secara manual, jalankan kode berikut:
        ```python
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Load data
        df = pd.read_csv("data/katalog_gempa_v2.csv")
        
        # Preprocess
        df_features = df[['magnitude', 'depth', 'latitude', 'longitude']].copy()
        df_features = df_features.interpolate()
        df_features.drop_duplicates(inplace=True)
        
        # Create and fit scaler
        scaler_dec = StandardScaler()
        scaler_dec.fit_transform(df_features)
        
        # Save scaler
        joblib.dump(scaler_dec, 'scaler_dec.pkl')
        print("Scaler berhasil disimpan!")
        ```
        
        ---
        
        **Opsi 3: Menjalankan Source Code Lengkap**
        
        Jika kedua file belum ada, jalankan file `salinan_dari_dec_update_revisi.py`:
        
        1. Buka file `salinan_dari_dec_update_revisi.py`
        2. Jalankan semua bagian hingga selesai
        3. Pastikan file berikut dihasilkan:
           - `scaler_dec.pkl` (dari bagian "PERSIAPAN MODELING DEC")
           - `model/autoencoder_model.h5` (dari bagian "TRAINING AUTOENCODER")
        """)
        
        st.info("💡 **Tip:** Setelah file dibuat, refresh halaman ini untuk memuat model.")
        st.stop()
    
    st.write("""
    Halaman ini menampilkan hasil Deep Embedded Clustering (DEC) pada data gempa.
    DEC menggunakan Autoencoder untuk mengekstrak representasi latent space, kemudian 
    melakukan K-Means clustering pada latent features tersebut per tahun.
    """)
    
    # Parameter clustering
    st.subheader("⚙️ Parameter Clustering")
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.number_input(
            "Jumlah Cluster",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            help="Jumlah cluster untuk K-Means clustering"
        )
    
    with col2:
        selected_year = st.selectbox(
            "Pilih Tahun untuk Visualisasi",
            options=sorted(df_clustering['year'].unique()),
            help="Pilih tahun untuk melihat visualisasi clustering"
        )
    
    if st.button('🔄 Jalankan DEC Clustering', type="primary", use_container_width=True):
        with st.spinner('Memproses DEC clustering... Ini mungkin memakan waktu beberapa saat...'):
            # Perform clustering
            df_clustered_by_year, latent_col_names = perform_dec_clustering(
                scaler_dec, encoder_model, df_clustering, n_clusters=n_clusters
            )
            
            if df_clustered_by_year:
                st.success(f"✅ Clustering selesai untuk {len(df_clustered_by_year)} tahun!")
                
                # Simpan ke session state
                st.session_state['df_clustered_by_year'] = df_clustered_by_year
                st.session_state['latent_col_names'] = latent_col_names
                
                # Hitung silhouette scores
                silhouette_scores = calculate_silhouette_scores(df_clustered_by_year, latent_col_names)
                st.session_state['silhouette_scores'] = silhouette_scores
                
                # Tampilkan informasi cluster dominan per tahun
                st.subheader("📊 Cluster Dominan per Tahun")
                cluster_info = []
                for year, yearly_df in df_clustered_by_year.items():
                    cluster_counts = yearly_df['cluster_label'].value_counts()
                    if not cluster_counts.empty:
                        dominant_cluster = cluster_counts.idxmax()
                        dominant_count = cluster_counts.max()
                        total = len(yearly_df)
                        percentage = (dominant_count / total) * 100
                        cluster_info.append({
                            'Tahun': year,
                            'Cluster Dominan': dominant_cluster,
                            'Jumlah': dominant_count,
                            'Total': total,
                            'Persentase (%)': f"{percentage:.2f}"
                        })
                
                if cluster_info:
                    df_cluster_info = pd.DataFrame(cluster_info)
                    st.dataframe(df_cluster_info, use_container_width=True)
            else:
                st.error("Gagal melakukan clustering. Periksa data dan model.")
    
    # Visualisasi jika clustering sudah dilakukan
    if 'df_clustered_by_year' in st.session_state and st.session_state['df_clustered_by_year']:
        df_clustered_by_year = st.session_state['df_clustered_by_year']
        latent_col_names = st.session_state['latent_col_names']
        
        # Tab untuk visualisasi
        tab1, tab2, tab3 = st.tabs(["🗺️ Peta Clustering", "📈 Silhouette Scores", "📋 Data Clustering"])
        
        with tab1:
            st.subheader(f"Peta Sebaran Cluster - Tahun {selected_year}")
            
            if selected_year in df_clustered_by_year:
                yearly_df = df_clustered_by_year[selected_year]
                
                # Warna cluster yang lebih kontras dan terlihat jelas
                # Cluster 0: Biru tua, Cluster 1: Merah terang, dll
                color_map = {
                    0: "#1f77b4",  # Biru tua (lebih terang dari biru biasa)
                    1: "#ff4444",  # Merah terang (lebih terang dari merah biasa)
                    2: "#2ca02c",  # Hijau
                    3: "#ff7f0e",  # Orange
                    4: "#9467bd",  # Ungu
                    5: "#8c564b",  # Coklat
                    6: "#e377c2",  # Pink
                    7: "#7f7f7f",  # Abu-abu
                    8: "#bcbd22",  # Kuning-hijau
                    9: "#17becf"   # Cyan
                }
                
                # Buat figure dengan density heatmap dan scatter
                fig = go.Figure()
                
                # Ambil unique clusters
                unique_clusters = sorted(yearly_df['cluster_label'].unique())
                
                # Tambahkan density heatmap untuk setiap cluster
                for cluster_id in unique_clusters:
                    cluster_data = yearly_df[yearly_df['cluster_label'] == cluster_id]
                    
                    if len(cluster_data) > 0:
                        try:
                            # Coba gunakan Densitymapbox jika tersedia
                            fig.add_trace(
                                go.Densitymapbox(
                                    lat=cluster_data['latitude'],
                                    lon=cluster_data['longitude'],
                                    z=[1] * len(cluster_data),  # Density value
                                    radius=25,  # Radius untuk smoothing
                                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, color_map.get(cluster_id, '#000000')]],
                                    showscale=False,
                                    name=f'Cluster {cluster_id} Density',
                                    opacity=0.5,
                                    hoverinfo='skip'
                                )
                            )
                        except:
                            # Fallback: gunakan scatter dengan size besar untuk efek heatmap
                            cluster_color = color_map.get(cluster_id, '#000000')
                            # Konversi hex ke rgba dengan opacity
                            rgb = tuple(int(cluster_color[i:i+2], 16) for i in (1, 3, 5))
                            rgba_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)'
                            
                            fig.add_trace(
                                go.Scattermapbox(
                                    lat=cluster_data['latitude'],
                                    lon=cluster_data['longitude'],
                                    mode='markers',
                                    marker=dict(
                                        size=30,  # Size besar untuk efek heatmap
                                        color=rgba_color,
                                        opacity=0.3,
                                        symbol='circle'
                                    ),
                                    name=f'Cluster {cluster_id} Heatmap',
                                    hoverinfo='skip',
                                    showlegend=False
                                )
                            )
                
                # Tambahkan scatter points untuk setiap cluster (di atas heatmap)
                for cluster_id in unique_clusters:
                    cluster_data = yearly_df[yearly_df['cluster_label'] == cluster_id]
                    
                    if len(cluster_data) > 0:
                        # Siapkan hover_data
                        hover_text = []
                        for idx, row in cluster_data.iterrows():
                            text = f"Cluster: {row['cluster_label']}<br>"
                            if 'magnitude' in row:
                                text += f"Magnitude: {row['magnitude']:.2f}<br>"
                            if 'depth' in row:
                                text += f"Depth: {row['depth']:.1f} km<br>"
                            text += f"Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}"
                            hover_text.append(text)
                        
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=cluster_data['latitude'],
                                lon=cluster_data['longitude'],
                                mode='markers',
                                marker=dict(
                                    size=12,  # Size lebih besar agar lebih terlihat
                                    color=color_map.get(cluster_id, '#000000'),
                                    opacity=0.95,  # Opacity tinggi agar jelas terlihat
                                    symbol='circle'
                                ),
                                name=f'Cluster {cluster_id}',
                                text=hover_text,
                                hoverinfo='text',
                                showlegend=True
                            )
                        )
                
                # Update layout
                fig.update_layout(
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(
                            lat=yearly_df['latitude'].mean(),
                            lon=yearly_df['longitude'].mean()
                        ),
                        zoom=4.5
                    ),
                    title=f"Sebaran Cluster & Heatmap Density Gempa - Tahun {selected_year}",
                    height=700,
                    legend=dict(
                        title="Cluster",
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=0
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistik cluster untuk tahun terpilih
                st.subheader(f"Statistik Cluster - Tahun {selected_year}")
                
                # Buat statistik cluster
                stats_dict = {'latitude': 'count'}
                
                # Tambahkan magnitude jika ada
                if 'magnitude' in yearly_df.columns:
                    stats_dict['magnitude'] = 'mean'
                
                # Tambahkan depth jika ada
                if 'depth' in yearly_df.columns:
                    stats_dict['depth'] = 'mean'
                
                cluster_stats = yearly_df.groupby('cluster_label').agg(stats_dict)
                cluster_stats = cluster_stats.rename(columns={'latitude': 'Jumlah Data'})
                
                # Rename kolom untuk lebih jelas
                if 'magnitude' in cluster_stats.columns:
                    cluster_stats = cluster_stats.rename(columns={'magnitude': 'Rata-rata Magnitudo'})
                if 'depth' in cluster_stats.columns:
                    cluster_stats = cluster_stats.rename(columns={'depth': 'Rata-rata Kedalaman (km)'})
                
                st.dataframe(cluster_stats, use_container_width=True)
            else:
                st.warning(f"Data untuk tahun {selected_year} tidak tersedia.")
        
        with tab2:
            st.subheader("Silhouette Score per Tahun")
            
            if 'silhouette_scores' in st.session_state:
                silhouette_scores = st.session_state['silhouette_scores']
                
                # Filter scores yang tidak None
                valid_scores = {k: v for k, v in silhouette_scores.items() if v is not None}
                
                if valid_scores:
                    # Buat DataFrame untuk visualisasi
                    df_silhouette = pd.DataFrame({
                        'Tahun': list(valid_scores.keys()),
                        'Silhouette Score': list(valid_scores.values())
                    }).sort_values('Tahun')
                    
                    # Visualisasi
                    fig = px.bar(
                        df_silhouette,
                        x='Tahun',
                        y='Silhouette Score',
                        title='Silhouette Score per Tahun',
                        labels={'Silhouette Score': 'Silhouette Score', 'Tahun': 'Tahun'},
                        color='Silhouette Score',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabel
                    st.dataframe(df_silhouette, use_container_width=True)
                    
                    # Statistik
                    avg_score = df_silhouette['Silhouette Score'].mean()
                    max_score = df_silhouette['Silhouette Score'].max()
                    min_score = df_silhouette['Silhouette Score'].min()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rata-rata Silhouette Score", f"{avg_score:.4f}")
                    with col2:
                        st.metric("Maksimum", f"{max_score:.4f}")
                    with col3:
                        st.metric("Minimum", f"{min_score:.4f}")
                else:
                    st.warning("Tidak ada silhouette score yang dapat dihitung.")
            else:
                st.info("Jalankan clustering terlebih dahulu untuk melihat silhouette scores.")
        
        with tab3:
            st.subheader("Data Hasil Clustering")
            
            # Pilih tahun untuk ditampilkan
            year_to_show = st.selectbox(
                "Pilih Tahun",
                options=sorted(df_clustered_by_year.keys()),
                key="year_selector"
            )
            
            if year_to_show in df_clustered_by_year:
                yearly_df = df_clustered_by_year[year_to_show]
                
                st.write(f"**Data Clustering Tahun {year_to_show}**")
                st.write(f"Total data: {len(yearly_df):,}")
                
                # Tampilkan data
                columns_to_display = ['year', 'magnitude', 'depth', 'latitude', 'longitude', 'cluster_label'] + latent_col_names[:5]
                available_columns = [col for col in columns_to_display if col in yearly_df.columns]
                
                st.dataframe(
                    yearly_df[available_columns],
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = yearly_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Data Clustering Tahun {year_to_show}",
                    data=csv,
                    file_name=f"dec_clustering_{year_to_show}.csv",
                    mime="text/csv"
                )
    
    # Informasi tentang DEC
    st.divider()
    with st.expander("ℹ️ Informasi tentang Deep Embedded Clustering (DEC)"):
        st.write("""
        **Deep Embedded Clustering (DEC):**
        - DEC menggunakan Autoencoder untuk mengekstrak representasi latent space dari data
        - Latent space adalah representasi terkompresi yang menangkap pola penting dalam data
        - K-Means clustering dilakukan pada latent features, bukan pada data asli
        - Clustering dilakukan per tahun untuk melihat perubahan pola seiring waktu
        
        **Proses:**
        1. Data di-preprocess dan di-scale menggunakan StandardScaler
        2. Autoencoder mengekstrak latent space (10 dimensi)
        3. K-Means clustering dilakukan pada latent features per tahun
        4. Hasil clustering divisualisasikan pada peta geografis
        
        **Silhouette Score:**
        - Mengukur seberapa baik data dikelompokkan ke dalam cluster
        - Nilai berkisar dari -1 hingga 1
        - Nilai lebih tinggi menunjukkan clustering yang lebih baik
        """)

# ==================== DETEKSI ANOMALI ====================
elif page == "Deteksi Anomali":
    st.markdown('<h1 class="main-header">🔍 Deteksi Anomali dengan Autoencoder</h1>', unsafe_allow_html=True)
    
    if autoencoder is None or scaler_dec is None:
        st.error("⚠️ Model atau scaler tidak ditemukan. Pastikan file berikut ada:")
        st.code("""
        - scaler_dec.pkl
        - model/autoencoder_model.h5
        """)
        st.info("Jalankan notebook untuk menghasilkan file-file tersebut terlebih dahulu.")
        st.stop()
    
    st.write("Aplikasi ini mendeteksi anomali pada data gempa berdasarkan rekonstruksi error dari Autoencoder.")
    
    # Input dari pengguna
    st.subheader("📝 Input Parameter Gempa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        magnitude = st.number_input(
            'Magnitude', 
            min_value=0.0, 
            max_value=10.0, 
            value=3.0, 
            step=0.1,
            help="Magnitudo gempa (skala Richter)"
        )
        depth = st.number_input(
            'Depth (km)', 
            min_value=0, 
            max_value=700, 
            value=10, 
            step=1,
            help="Kedalaman gempa dalam kilometer"
        )
    
    with col2:
        latitude = st.number_input(
            'Latitude', 
            min_value=-90.0, 
            max_value=90.0, 
            value=-5.0, 
            step=0.01,
            help="Koordinat lintang (garis lintang)"
        )
        longitude = st.number_input(
            'Longitude', 
            min_value=-180.0, 
            max_value=180.0, 
            value=110.0, 
            step=0.01,
            help="Koordinat bujur (garis bujur)"
        )
    
    if st.button('🔍 Deteksi Anomali', type="primary", use_container_width=True):
        with st.spinner('Memproses deteksi anomali...'):
            # Membuat DataFrame dari input pengguna
            input_data = pd.DataFrame([[magnitude, depth, latitude, longitude]],
                                    columns=['magnitude', 'depth', 'latitude', 'longitude'])
            
            # Scaling input data
            scaled_input = scaler_dec.transform(input_data)
            
            # Prediksi rekonstruksi dari autoencoder
            reconstructed_input = autoencoder.predict(scaled_input, verbose=0)
            
            # Hitung reconstruction error (MSE)
            reconstruction_error = np.mean(np.square(scaled_input - reconstructed_input), axis=1)[0]
            
            # Tampilkan hasil
            st.divider()
            st.subheader("📊 Hasil Deteksi:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Reconstruction Error", f"{reconstruction_error:.8f}")
                st.metric("Threshold Anomali", f"{threshold:.8f}")
            
            with col2:
                if reconstruction_error > threshold:
                    st.error('⚠️ **ANOMALI DETECTED!**')
                    st.write("Data ini terdeteksi sebagai **ANOMALI** berdasarkan reconstruction error yang tinggi.")
                else:
                    st.success('✅ **NORMAL**')
                    st.write("Data ini terdeteksi sebagai **NORMAL**.")
            
            # Visualisasi
            st.subheader("📈 Visualisasi Reconstruction Error")
            
            # Buat bar chart untuk perbandingan
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Reconstruction Error', 'Threshold'],
                y=[reconstruction_error, threshold],
                marker_color=['red' if reconstruction_error > threshold else 'green', 'blue'],
                text=[f'{reconstruction_error:.8f}', f'{threshold:.8f}'],
                textposition='auto'
            ))
            fig.update_layout(
                title='Perbandingan Reconstruction Error dengan Threshold',
                yaxis_title='Error Value',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Batch anomaly detection
    st.divider()
    st.subheader("🔍 Deteksi Anomali Batch (Seluruh Data)")
    
    if st.button('📊 Analisis Anomali pada Seluruh Data', type="secondary", use_container_width=True):
        if autoencoder is None or scaler_dec is None:
            st.error("Model tidak tersedia untuk analisis batch.")
        else:
            with st.spinner('Memproses deteksi anomali pada seluruh data...'):
                # Preprocess data
                df_dec_input = df_clustering[['magnitude', 'depth', 'latitude', 'longitude']].copy()
                df_dec_input = df_dec_input.interpolate()
                df_dec_input.drop_duplicates(inplace=True)
                
                # Scale data
                X_dec = scaler_dec.transform(df_dec_input)
                
                # Rekonstruksi
                X_dec_pred = autoencoder.predict(X_dec, verbose=0)
                
                # Hitung reconstruction error
                reconstruction_error = np.mean(np.square(X_dec - X_dec_pred), axis=1)
                
                # Tentukan anomali
                anomaly_flags = reconstruction_error > threshold
                
                # Gabungkan hasil
                df_results = df_dec_input.copy()
                df_results['reconstruction_error'] = reconstruction_error
                df_results['is_anomaly'] = anomaly_flags.astype(int)
                
                # Statistik
                n_normal = len(df_results[df_results['is_anomaly'] == 0])
                n_anomaly = len(df_results[df_results['is_anomaly'] == 1])
                pct_anomaly = (n_anomaly / len(df_results)) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Normal", f"{n_normal:,}")
                with col2:
                    st.metric("Data Anomali", f"{n_anomaly:,}")
                with col3:
                    st.metric("Persentase Anomali", f"{pct_anomaly:.2f}%")
                
                # Visualisasi distribusi reconstruction error
                st.subheader("📈 Distribusi Reconstruction Error")
                fig = px.histogram(
                    df_results,
                    x='reconstruction_error',
                    nbins=50,
                    title='Distribusi Reconstruction Error',
                    labels={'reconstruction_error': 'Reconstruction Error', 'count': 'Frekuensi'}
                )
                fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                            annotation_text=f"Threshold: {threshold:.8f}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Top 10 anomali
                st.subheader("🔴 Top 10 Anomali dengan Error Terbesar")
                df_top_anomaly = df_results.sort_values(by='reconstruction_error', ascending=False).head(10)
                st.dataframe(df_top_anomaly, use_container_width=True)
                
                # Download hasil
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil Deteksi Anomali",
                    data=csv,
                    file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Informasi tentang threshold
    st.divider()
    with st.expander("ℹ️ Informasi tentang Threshold"):
        st.write(f"""
        **Threshold Anomali:**
        - Threshold ditentukan menggunakan metode **mean + 3 × standard deviation**
        - Data dengan reconstruction error di atas threshold dianggap sebagai anomali
        - Threshold saat ini: **{threshold:.8f}**
        
        **Cara Kerja:**
        1. Autoencoder mencoba merekonstruksi data input
        2. Reconstruction error dihitung sebagai Mean Squared Error (MSE)
        3. Jika error > threshold → **ANOMALI**
        4. Jika error ≤ threshold → **NORMAL**
        """)

# ==================== PREDIKSI GEMPA ====================
elif page == "Prediksi Gempa":
    st.markdown('<h1 class="main-header">🔮 Prediksi Lokasi Gempa 2026</h1>', unsafe_allow_html=True)
    
    # Load data untuk prediksi (gunakan data yang sudah dimuat)
    if df is None:
        st.error("Data tidak tersedia. Silakan refresh halaman.")
        st.stop()
    
    # Pastikan data memiliki kolom yang diperlukan
    prediction_data = df.copy()
    if 'datetime' not in prediction_data.columns:
        st.error("Data tidak memiliki kolom 'datetime'. Pastikan format data benar.")
        st.stop()
    
    # Pastikan kolom datetime terparse dengan benar
    prediction_data['datetime'] = pd.to_datetime(prediction_data['datetime'], errors='coerce')
    prediction_data = prediction_data.dropna(subset=['datetime'])
    prediction_data = prediction_data.dropna(subset=['latitude', 'longitude', 'magnitude'])
    prediction_data['magnitude'] = pd.to_numeric(prediction_data['magnitude'], errors='coerce')
    prediction_data = prediction_data.dropna(subset=['magnitude'])
    
    # Parse datetime agar ada fitur tahun/bulan
    prediction_data['year'] = prediction_data['datetime'].dt.year
    prediction_data['month'] = prediction_data['datetime'].dt.month
    
    st.write("""
    Halaman ini menampilkan prediksi lokasi hotspot gempa untuk tahun 2026 menggunakan metode LSTM (Long Short-Term Memory) 
    atau Frequency-Based Analysis. Sistem ini menganalisis pola temporal dan spasial dari data gempa historis 2008-2025.
    """)
    
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
    # Pastikan menggunakan SEMUA data yang valid untuk perhitungan frekuensi historis
    heat_source_pred = prediction_data.copy()
    
    # Debug: Tampilkan info data yang digunakan (opsional, bisa di-comment jika tidak perlu)
    # st.caption(f"📊 Data untuk prediksi: {len(heat_source_pred):,} baris, tahun {heat_source_pred['year'].min()}-{heat_source_pred['year'].max()}")
    
    # Cek apakah ada hasil prediksi yang tersimpan
    saved_prediction_path = "model/prediction_results.json"
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
            
            # Cek apakah model sudah ada
            model_path = "model/lstm_earthquake_model.h5"
            saved_model = load_lstm_model(model_path)
            
            if saved_model is not None:
                st.info("💾 **Model LSTM tersimpan ditemukan!** Model akan dimuat otomatis.")
            else:
                st.info("💡 **Model belum ada.** Menggunakan metode Frequency-Based sebagai fallback.")
            
            model = None
            history = None
            
            # Cek apakah menggunakan hasil prediksi tersimpan
            hotspot_points = None
            use_saved = saved_hotspot_points and saved_method == method and not st.session_state.regenerate_prediction
            
            if use_saved:
                # Filter sesuai top_k dari slider
                if len(saved_hotspot_points) >= top_k:
                    hotspot_points = saved_hotspot_points[:top_k]
                    if len(saved_hotspot_points) > top_k:
                        st.success(f"✅ Menggunakan hasil prediksi tersimpan! Menampilkan {len(hotspot_points)} hotspot teratas dari {len(saved_hotspot_points)} hotspot tersimpan.")
                    else:
                        st.success(f"✅ Menggunakan hasil prediksi tersimpan! ({len(hotspot_points)} hotspot)")
                else:
                    # Hasil tersimpan tidak cukup, perlu regenerate
                    st.info(f"ℹ️ Hasil prediksi tersimpan hanya memiliki {len(saved_hotspot_points)} hotspot, sedangkan slider meminta {top_k}. Regenerating...")
                    hotspot_points = None  # Force regenerate
            
            # Jika tidak menggunakan hasil tersimpan atau perlu regenerate, gunakan model
            if hotspot_points is None:
                # Gunakan model yang sudah ada
                if saved_model is not None:
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
                            st.session_state.regenerate_prediction = False
                        except Exception as e:
                            st.warning(f"⚠️ Gagal menyimpan hasil prediksi: {e}")
                    else:
                        st.error("❌ Gagal melakukan prediksi. Menggunakan metode Frequency-Based.")
                        hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
                # Jika belum ada model, gunakan metode Frequency-Based
                elif saved_model is None:
                    hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
                    st.info("ℹ️ Model LSTM belum tersedia. Menggunakan metode Frequency-Based untuk prediksi.")
    else:
        # Metode Frequency-Based (original)
        # Cek apakah menggunakan hasil prediksi tersimpan
        hotspot_points = None
        if saved_hotspot_points and saved_method == method and not st.session_state.regenerate_prediction:
            # Filter sesuai top_k dari slider
            if len(saved_hotspot_points) >= top_k:
                hotspot_points = saved_hotspot_points[:top_k]
                if len(saved_hotspot_points) > top_k:
                    st.success(f"✅ Menggunakan hasil prediksi tersimpan! Menampilkan {len(hotspot_points)} hotspot teratas dari {len(saved_hotspot_points)} hotspot tersimpan.")
                else:
                    st.success(f"✅ Menggunakan hasil prediksi tersimpan! ({len(hotspot_points)} hotspot)")
            else:
                # Hasil tersimpan tidak cukup, perlu regenerate
                st.info(f"ℹ️ Hasil prediksi tersimpan hanya memiliki {len(saved_hotspot_points)} hotspot, sedangkan slider meminta {top_k}. Regenerating...")
                hotspot_points = None  # Force regenerate
        
        # Jika tidak menggunakan hasil tersimpan atau perlu regenerate, hitung ulang
        if hotspot_points is None:
            hotspot_points = build_hotspot_points(heat_source_pred, bin_size=bin_size, top_k=top_k)
            # Simpan hasil prediksi
            try:
                save_prediction_results(hotspot_points, method, saved_prediction_path)
                st.session_state.regenerate_prediction = False
            except Exception as e:
                st.warning(f"⚠️ Gagal menyimpan hasil prediksi: {e}")
    
    # --- Plot Time Series Frekuensi Gempa 2008-2026 ---
    st.subheader("📈 Plot Frekuensi Gempa 2008-2026")
    
    # Siapkan data untuk chart
    chart_data = prepare_prediction_chart_data(prediction_data, hotspot_points, method)
    
    if not chart_data.empty:
        # Buat plot menggunakan Altair
        chart = alt.Chart(chart_data).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X('year:O', title='Tahun', axis=alt.Axis(format='d')),
            y=alt.Y('frequency:Q', title='Frekuensi Gempa'),
            color=alt.Color('type:N', 
                           scale=alt.Scale(domain=['Historical', 'Predicted'], 
                                          range=['#1f77b4', '#ff7f0e']),
                           legend=alt.Legend(title='Jenis Data')),
            strokeDash=alt.condition(
                alt.datum.type == 'Predicted',
                alt.value([5, 5]),  # Dashed line untuk prediksi
                alt.value([0])      # Solid line untuk historis
            )
        ).properties(
            width=800,
            height=400,
            title='Trend Frekuensi Gempa 2008-2026 (Historis + Prediksi)'
        )
        
        # Tambahkan titik untuk prediksi dengan ukuran lebih besar
        pred_points = alt.Chart(chart_data[chart_data['type'] == 'Predicted']).mark_circle(
            size=150,
            color='#ff7f0e'
        ).encode(
            x='year:O',
            y='frequency:Q',
            tooltip=[alt.Tooltip('year:O', title='Tahun'),
                    alt.Tooltip('frequency:Q', title='Frekuensi Prediksi', format='.0f')]
        )
        
        # Gabungkan chart
        final_chart = (chart + pred_points).resolve_scale(color='independent')
        
        st.altair_chart(final_chart, use_container_width=True)
        
        # Tampilkan statistik ringkas
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        historical_data = chart_data[chart_data['type'] == 'Historical']
        predicted_data = chart_data[chart_data['type'] == 'Predicted']
        
        with col_stat1:
            if not historical_data.empty:
                total_historical = historical_data['frequency'].sum()
                st.metric("Total Gempa 2008-2025", f"{total_historical:,}")
        
        with col_stat2:
            if not historical_data.empty:
                avg_historical = historical_data['frequency'].mean()
                st.metric("Rata-rata per Tahun", f"{avg_historical:.1f}")
        
        with col_stat3:
            if not historical_data.empty:
                max_year = historical_data.loc[historical_data['frequency'].idxmax(), 'year']
                max_freq = historical_data['frequency'].max()
                st.metric(f"Tahun Tertinggi ({max_year})", f"{max_freq:,}")
        
        with col_stat4:
            if not predicted_data.empty:
                pred_2026_value = predicted_data[predicted_data['year'] == 2026]['frequency'].values[0]
                if not historical_data.empty:
                    last_year_value = historical_data[historical_data['year'] == 2025]['frequency'].values[0] if 2025 in historical_data['year'].values else 0
                    change = pred_2026_value - last_year_value
                    change_pct = (change / last_year_value * 100) if last_year_value > 0 else 0
                    st.metric("Prediksi 2026", f"{pred_2026_value:,}", 
                             delta=f"{change:+,} ({change_pct:+.1f}%)")
    
    # --- Heatmap estimasi lokasi (berbasis pola historis) ---
    st.subheader("Heatmap estimasi kerawanan (berdasarkan pola 2008-2025)")
    heat_year = st.selectbox("Tahun target (estimasi)", range(2008, 2027), index=list(range(2008, 2027)).index(2026))
    
    # Menggunakan semua data historis tanpa filter bulan untuk heatmap
    heat_source = prediction_data.copy()
    heatmap_points = build_heatmap_points(heat_source)
    
    # --- Peta prediksi 2026 (terpisah) ---
    st.subheader("Peta prediksi hotspot 2026 (berdasar pola historis)")
    pred_data_df = pd.DataFrame(hotspot_points)
    if pred_data_df.empty:
        st.warning("Tidak ada titik hotspot yang dapat dihitung.")
        show_grid = st.checkbox("📐 Tampilkan Grid", value=True, 
                                help="Tampilkan grid 0.5° × 0.5° untuk memvisualisasikan spatial binning.")
        mymap_pred = create_prediction_map(
            pd.DataFrame(columns=prediction_data.columns), 
            predicted_points=None, 
            target_year=2026, 
            heatmap_points=None,
            show_grid=show_grid,
            bin_size=bin_size
        )
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
            mymap_pred = create_prediction_map(
                pd.DataFrame(columns=prediction_data.columns),
                predicted_points=hotspot_points,
                target_year=2026,
                heatmap_points=None,
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

# ==================== DATA EXPLORATION ====================
elif page == "Data Exploration":
    st.markdown('<h1 class="main-header">🔎 Data Exploration</h1>', unsafe_allow_html=True)
    
    # Filter data
    st.subheader("🔍 Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_filter = st.multiselect(
            "Pilih Tahun",
            options=sorted(df['year'].unique()),
            default=sorted(df['year'].unique())
        )
    
    with col2:
        min_mag = float(df['magnitude'].min())
        max_mag = float(df['magnitude'].max())
        mag_range = st.slider(
            "Rentang Magnitudo",
            min_value=min_mag,
            max_value=max_mag,
            value=(min_mag, max_mag)
        )
    
    with col3:
        min_depth = int(df['depth'].min())
        max_depth = int(df['depth'].max())
        depth_range = st.slider(
            "Rentang Kedalaman (km)",
            min_value=min_depth,
            max_value=max_depth,
            value=(min_depth, max_depth)
        )
    
    # Filter data
    df_filtered = df[
        (df['year'].isin(year_filter)) &
        (df['magnitude'] >= mag_range[0]) &
        (df['magnitude'] <= mag_range[1]) &
        (df['depth'] >= depth_range[0]) &
        (df['depth'] <= depth_range[1])
    ]
    
    st.write(f"**Total data setelah filter: {len(df_filtered):,}**")
    
    # Tampilkan data
    st.subheader("📋 Data Table")
    
    # Pilih kolom yang ditampilkan
    columns_to_show = st.multiselect(
        "Pilih Kolom untuk Ditampilkan",
        options=df_filtered.columns.tolist(),
        default=['datetime', 'magnitude', 'depth', 'latitude', 'longitude', 'location']
    )
    
    if columns_to_show:
        st.dataframe(
            df_filtered[columns_to_show],
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df_filtered[columns_to_show].to_csv(index=False)
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name=f"earthquake_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Statistik
    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df_filtered[['magnitude', 'depth', 'latitude', 'longitude']].describe())

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### 📝 Informasi
**Aplikasi Analisis Data Gempa**
- Deep Embedded Clustering (DEC)
- Anomaly Detection dengan Autoencoder
- Exploratory Data Analysis
- Visualisasi Geografis Clustering
""")

if scaler_dec is not None and autoencoder is not None:
    st.sidebar.success("✅ Model tersedia")
    st.sidebar.info(f"Threshold: {threshold:.8f}")
else:
    st.sidebar.warning("⚠️ Model belum dimuat")

