import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Bitcoin EDA Journal", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('bitcoin_cleaned.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

try:
    data = load_data()

    # --- SIDEBAR ---
    st.sidebar.title("🧭 Navigasi Jurnal")
    page = st.sidebar.radio("Pilih Bagian Jurnal:", 
                            ["Part 1: Cleaning", "Part 2: Visual Inspection", 
                             "Part 3: Decomposition", "Part 4: Stationarity Check"])

    st.title("₿ Bitcoin Time Series EDA Journal")
    st.markdown("---")

    if page == "Part 1: Cleaning":
        st.header("Jurnal Part 1: The First Encounter")
        st.write("Menampilkan 5 data pertama setelah proses cleaning dan resampling harian.")
        st.dataframe(data.head())
        st.success("Data telah bersih dari NaN dan menggunakan index datetime.")

    elif page == "Part 2: Visual Inspection":
        st.header("Jurnal Part 2: Visual Inspection")
        period = st.select_slider("Pilih Smoothing Period:", options=["Harian", "Mingguan", "Bulanan"])
        
        mapping = {"Harian": "D", "Mingguan": "W", "Bulanan": "M"}
        resampled = data['Close'].resample(mapping[period]).mean()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['Close'], alpha=0.3, label="Harian")
        ax.plot(resampled.index, resampled, color='red', label=f"Rata-rata {period}")
        ax.legend()
        st.pyplot(fig)
        st.write("Narasi: Semakin besar periode rata-rata, semakin jelas tren jangka panjang Bitcoin terlihat.")

    elif page == "Part 3: Decomposition":
        st.header("Jurnal Part 3: Decomposing the Pattern")
        if st.button("Bedah Komponen"):
            # Multiplicative karena fluktuasi harga Bitcoin bersifat proporsional terhadap nilainya
            result = seasonal_decompose(data['Close'], model='multiplicative', period=365)
            fig = result.plot()
            fig.set_size_inches(10, 8)
            st.pyplot(fig)
            st.info("Trend menunjukkan kenaikan eksponensial, sementara Seasonal menunjukkan siklus tahunan.")

    elif page == "Part 4: Stationarity Check":
        st.header("Jurnal Part 4: Statistical Health Check")
        window = st.slider("Window Size (Hari):", 30, 365, 180)
        
        rolmean = data['Close'].rolling(window=window).mean()
        rolstd = data['Close'].rolling(window=window).std()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Close'], alpha=0.3, label="Harga")
        ax.plot(rolmean, color='red', label="Rolling Mean")
        ax.plot(rolstd, color='black', label="Rolling Std")
        ax.legend()
        st.pyplot(fig)
        st.error("Kesimpulan: Rata-rata yang terus naik membuktikan data TIDAK STASIONER.")

except Exception as e:
    st.error(f"Error: Pastikan file 'bitcoin_cleaned.csv' ada di folder yang sama. Detail: {e}")