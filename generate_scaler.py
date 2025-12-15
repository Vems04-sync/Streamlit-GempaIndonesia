"""
Script untuk menghasilkan file scaler_dec.pkl yang diperlukan untuk aplikasi Streamlit.
Jalankan script ini sebelum menjalankan streamlit_app.py jika file scaler_dec.pkl belum ada.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_scaler():
    """Menghasilkan scaler_dec.pkl dari data gempa"""
    
    print("=" * 60)
    print("GENERATE SCALER DEC")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Memuat data dari data/katalog_gempa_v2.csv...")
    try:
        df = pd.read_csv("data/katalog_gempa_v2.csv")
        print(f"   ✅ Data berhasil dimuat: {len(df)} baris")
    except FileNotFoundError:
        print("   ❌ Error: File data/katalog_gempa_v2.csv tidak ditemukan!")
        return False
    except Exception as e:
        print(f"   ❌ Error memuat data: {e}")
        return False
    
    # 2. Preprocessing
    print("\n2. Melakukan preprocessing data...")
    try:
        # Ambil kolom yang diperlukan
        df_features = df[['magnitude', 'depth', 'latitude', 'longitude']].copy()
        
        # Interpolasi nilai hilang
        df_features = df_features.interpolate()
        
        # Hapus data duplikat
        df_features.drop_duplicates(inplace=True)
        
        print(f"   ✅ Preprocessing selesai: {len(df_features)} baris setelah preprocessing")
    except Exception as e:
        print(f"   ❌ Error preprocessing: {e}")
        return False
    
    # 3. Buat dan fit scaler
    print("\n3. Membuat StandardScaler...")
    try:
        scaler_dec = StandardScaler()
        X_dec = scaler_dec.fit_transform(df_features)
        print(f"   ✅ Scaler berhasil dibuat dan di-fit")
        print(f"   ✅ Shape data setelah scaling: {X_dec.shape}")
    except Exception as e:
        print(f"   ❌ Error membuat scaler: {e}")
        return False
    
    # 4. Simpan scaler
    print("\n4. Menyimpan scaler ke scaler_dec.pkl...")
    try:
        joblib.dump(scaler_dec, 'scaler_dec.pkl')
        print("   ✅ Scaler berhasil disimpan ke scaler_dec.pkl")
    except Exception as e:
        print(f"   ❌ Error menyimpan scaler: {e}")
        return False
    
    # 5. Verifikasi
    print("\n5. Verifikasi file...")
    if os.path.exists('scaler_dec.pkl'):
        file_size = os.path.getsize('scaler_dec.pkl')
        print(f"   ✅ File scaler_dec.pkl ada (ukuran: {file_size} bytes)")
        return True
    else:
        print("   ❌ File scaler_dec.pkl tidak ditemukan setelah penyimpanan!")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCRIPT GENERATE SCALER DEC")
    print("=" * 60)
    print("\nScript ini akan menghasilkan file scaler_dec.pkl yang diperlukan")
    print("untuk aplikasi Streamlit.\n")
    
    success = generate_scaler()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUKSES! File scaler_dec.pkl berhasil dibuat.")
        print("\nSekarang Anda dapat menjalankan aplikasi Streamlit dengan:")
        print("   streamlit run streamlit_app.py")
    else:
        print("❌ GAGAL! Terjadi error saat membuat scaler.")
        print("\nPastikan:")
        print("   1. File data/katalog_gempa_v2.csv ada")
        print("   2. Semua dependencies terinstall (pandas, scikit-learn, joblib)")
        print("   3. Jalankan: pip install -r requirements.txt")
    print("=" * 60 + "\n")

