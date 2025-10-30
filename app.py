import streamlit as st
import pandas as pd
import joblib

# ==============================================================================
# KONFIGURASI HALAMAN DAN MEMUAT MODEL
# ==============================================================================

# Atur konfigurasi halaman (Judul di tab browser dan ikon)
st.set_page_config(page_title="Segmentasi Pelanggan", layout="wide")

# Fungsi untuk memuat model (menggunakan cache agar lebih cepat)
@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return kmeans_model, scaler

# Memuat model dan data
try:
    kmeans_model, scaler = load_models()
    rfm_data = pd.read_csv('rfm_data_final.csv')
except FileNotFoundError:
    st.error("File model atau data tidak ditemukan. Pastikan file .pkl dan .csv ada di folder yang sama.")
    st.stop()


# ==============================================================================
# UI - JUDUL DAN PENJELASAN
# ==============================================================================

st.title("Dasbor Segmentasi Pelanggan E-commerce")
st.subheader("Final Project Kelompok 12 : Nasa, Fiqa, Novia")
st.write("""
Proyek ini menggunakan Analisis RFM dan K-Means Clustering untuk membagi pelanggan
ke dalam 4 segmen berbeda berdasarkan perilaku pembelian mereka.
""")

# ==============================================================================
# UI - MENAMPILKAN HASIL ANALISIS
# ==============================================================================

st.header("Hasil Analisis Segmen Pelanggan")

# Definisikan interpretasi Setiap Segmen Pelanggan
segment_interpretation = {
    3: {
        "Nama": "Pelanggan VVIP (Mega Stars)",
        "Deskripsi": "Segmen super elit. Meskipun jumlahnya sangat kecil (hanya 4 pelanggan), kontribusi mereka luar biasa. Mereka baru saja berbelanja (Recency sangat rendah, ~3.5 hari), sangat sering (Frequency sangat tinggi, ~212 kali), dan nilai belanjanya masif (Monetary ekstrim, ~£436k). Pelanggan ini kemungkinan besar adalah reseller atau pelanggan B2B (Business-to-Business).",
	    "Rekomendasi Strategi": "1. Hubungan Personal: Jangan andalkan email otomatis. Segmen ini harus ditangani langsung oleh manajer akun atau pemilik bisnis. \n 2. Layanan Prioritas: Memberikan layanan pelanggan prioritas, diskon volume, dan penawaran khusus yang dirancang hanya untuk mereka. Tujuannya adalah retensi maksimal."
    },
    2: {
        "Nama": "Pelanggan Juara (Champions)",
        "Deskripsi": "Kelompok pelanggan terbaik setelah VVIP. Jumlahnya juga kecil (35 pelanggan). Mereka aktif (Recency rendah, ~26 hari), sering berbelanja (Frequency tinggi, ~104 kali), dan menghabiskan sangat banyak uang (Monetary sangat tinggi, ~£83k).",
        "Rekomendasi Strategi": "1. Program Loyalitas: Berikan mereka status keanggotaan tertinggi dalam program loyalitas Anda. \n 2. Apresiasi: Kirimkan hadiah atau akses eksklusif ke produk baru. Pastikan mereka merasa sangat dihargai."
    },
    1: {
        "Nama": "Pelanggan Setia (Loyal Customers)",
        "Deskripsi": "Tulang punggung dari basis pelanggan dan merupakan segmen terbesar yang aktif (3841 pelanggan). Mereka memiliki nilai yang solid di semua metrik: Recency cukup baik (~ 67 hari), frekuensi belanja lumayan (~ 7 kali), dan total belanja yang baik (~ £3k).",
        "Rekomendasi Strategi": "1. Kembangkan (Nurture): Tujuannya adalah mendorong mereka untuk berbelanja lebih sering dan lebih banyak agar bisa \"naik kelas\" menjadi \"Pelanggan Juara\". \n 2. Marketing Tertarget: Mengirimkan rekomendasi produk yang dipersonalisasi, tawarkan sistem poin, dan berikan diskon untuk pembelian berikutnya."
    },
    0: {
        "Nama": "Pelanggan Hilang (Lost Customers)",
        "Deskripsi": "Segmen pelanggan yang sudah tidak aktif. Recency mereka sangat tinggi (~463 hari atau lebih dari setahun), yang menandakan mereka sudah lama sekali tidak kembali. Frekuensi dan nilai belanja mereka juga rendah.",
        "Rekomendasi Strategi": "1. Upaya Minimal: Jangan habiskan banyak anggaran pemasaran untuk segmen ini. \n 2. Kampanye Win-back Otomatis: Cukup kirimkan satu atau dua kali email \"win-back\" otomatis dengan diskon yang sangat besar. Jika tidak ada respons, fokuskan sumber daya Anda pada segmen yang lebih aktif."
    }
}

# Menampilkan interpretasi dan strategi segmen
for cluster_id, info in segment_interpretation.items():
    with st.expander(f"**Cluster {cluster_id}: {info['Nama']}**"):
        
        st.write("**Deskripsi:**")
        st.write(info['Deskripsi'])

        st.write("**Rekomendasi Strategi:**")
        st.write(info.get("Rekomendasi Strategi", "N/A"))

st.dataframe(rfm_data.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster':'Jumlah Pelanggan'}).round(2))


# ==============================================================================
# UI - FITUR INTERAKTIF (PREDIKSI LIVE)
# ==============================================================================

st.header("Cek Segmen Pelanggan (Live)")

# Input dari pengguna
customer_id_input = st.number_input("Masukkan Customer ID:", step=1, value=0)

if customer_id_input != 0:
    try:
        # Cari data pelanggan di DataFrame RFM
        customer_data = rfm_data[rfm_data['Customer ID'] == customer_id_input]
        
        if not customer_data.empty:
            # Ambil nilai RFM
            rfm_values = customer_data[['Recency', 'Frequency', 'MonetaryValue']]
            
            # 1. Lakukan Penskalaan (menggunakan scaler yang sudah disimpan)
            rfm_scaled = scaler.transform(rfm_values)
            
            # 2. Lakukan Prediksi (menggunakan model yang sudah disimpan)
            cluster_prediction = kmeans_model.predict(rfm_scaled)
            
            # 3. Tampilkan hasil
            predicted_cluster = cluster_prediction[0]
            segment_info = segment_interpretation[predicted_cluster]
            
            st.success(f"Pelanggan {customer_id_input} termasuk dalam segmen:")
            st.subheader(f"**Cluster {predicted_cluster}: {segment_info['Nama']}**")    

            st.write("**Deskripsi:**")
            st.write(segment_info['Deskripsi'])

            st.write("**Rekomendasi Strategi:**")
            st.write(segment_info.get("Rekomendasi Strategi", "N/A"))
            
        else:
            st.warning("Customer ID tidak ditemukan dalam data RFM. Pelanggan ini mungkin tidak memiliki transaksi yang valid.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")