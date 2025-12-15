"""
Script untuk memeriksa apakah semua file yang diperlukan untuk menjalankan aplikasi Streamlit sudah ada.
"""

import os
from pathlib import Path

def check_files():
    """Memeriksa keberadaan file-file yang diperlukan"""
    
    required_files = {
        "Data": [
            "data/katalog_gempa_v2.csv"
        ],
        "Model": [
            "model/autoencoder_model.h5",
            "scaler_dec.pkl"
        ]
    }
    
    missing_files = []
    existing_files = []
    
    print("=" * 60)
    print("PEMERIKSAAN FILE YANG DIPERLUKAN")
    print("=" * 60)
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file_path in files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
                existing_files.append(file_path)
            else:
                print(f"  ❌ {file_path} - TIDAK DITEMUKAN")
                missing_files.append(file_path)
    
    print("\n" + "=" * 60)
    
    if missing_files:
        print(f"\n⚠️  Total file yang hilang: {len(missing_files)}")
        print("\nFile yang perlu dibuat:")
        for file in missing_files:
            print(f"  - {file}")
        
        print("\n💡 Cara membuat file yang hilang:")
        print("   1. Buka notebook: Salinan_dari_DEC_Update_Revisi.ipynb")
        print("   2. Jalankan semua cell hingga selesai")
        print("   3. Pastikan file berikut dihasilkan:")
        print("      - scaler_dec.pkl (dari Cell 23)")
        print("      - model/autoencoder_model.h5 (dari Cell 27)")
        
        return False
    else:
        print("\n✅ Semua file yang diperlukan sudah ada!")
        print("   Anda dapat menjalankan aplikasi dengan:")
        print("   streamlit run streamlit_app.py")
        return True

if __name__ == "__main__":
    check_files()

