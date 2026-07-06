import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. SAYFA AYARLARI VE GÖRÜNÜM
# ==========================================
st.set_page_config(page_title="Dinamik Karar Destek Sistemi", page_icon="🚀", layout="wide")

st.title("🚀 Dinamik Tedarik Zinciri Karar Destek Sistemi (KDS)")
st.markdown("Makine Öğrenmesi, Monte Carlo Simülasyonu ve İnteraktif MCDM Optimizasyonu")
st.divider()

# ==========================================
# 2. VERİ YÜKLEME VE MODEL EĞİTİMİ (ÖNBELLEKLİ)
# ==========================================
PATH = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv(PATH, encoding='ISO-8859-1')
        df.columns = [col.strip() for col in df.columns]
        df['is_late'] = np.where(df['Delivery Status'].str.strip() == 'Late delivery', 1, 0)
        shipping_map = {
            'Standard Class': 'Standart Teslimat', 
            'Second Class': 'İkinci Sınıf Teslimat', 
            'First Class': 'Birinci Sınıf Teslimat', 
            'Same Day': 'Aynı Gün Teslimat'
        }
        df['Shipping Mode'] = df['Shipping Mode'].str.strip().map(shipping_map).fillna(df['Shipping Mode'])
        return df
    except Exception as e:
        return None

@st.cache_resource
def train_fast_model(df):
    features = ['Shipping Mode', 'Order Region', 'Category Name', 'Sales per customer']
    target = 'is_late'
    X = df[features].fillna('Bilinmiyor')
    y = df[target]
    cat_features = ['Shipping Mode', 'Order Region', 'Category Name']
    model = CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1, cat_features=cat_features, verbose=False)
    model.fit(X, y)
    return model

with st.spinner("Yapay Zeka Modelleri Eğitiliyor..."):
    df = load_and_preprocess_data()
    if df is not None:
        model = train_fast_model(df)
    else:
        st.error(f"Veri seti bulunamadı. Lütfen yolu kontrol edin: {PATH}")
        st.stop()

# ==========================================
# 3. KULLANICI GİRDİLERİ (SOL MENÜ)
# ==========================================
st.sidebar.header("📦 1. Sipariş Parametreleri")
selected_region = st.sidebar.selectbox("Sipariş Bölgesi", df['Order Region'].dropna().unique(), index=0)
selected_category = st.sidebar.selectbox("Ürün Kategorisi", df['Category Name'].dropna().unique(), index=0)
order_amount = st.sidebar.slider("Sipariş Tutarı ($)", min_value=10.0, max_value=2000.0, value=399.0, step=10.0)

st.sidebar.divider()
st.sidebar.header("⚙️ 2. Simülasyon Ayarları")
num_iterations = st.sidebar.slider("Monte Carlo İterasyon Sayısı", 1000, 100000, 50000, step=10000)
min_penalty = st.sidebar.slider("Minimum SLA Cezası (%)", 1, 10, 5)
max_penalty = st.sidebar.slider("Maksimum SLA Cezası (%)", 15, 50, 30)

st.sidebar.divider()
st.sidebar.header("⚖️ 3. MCDM Ağırlıkları")
st.sidebar.caption("Algoritmanın hangi kritere daha çok önem vereceğini belirleyin.")
w_cost = st.sidebar.slider("Beklenen Risk Maliyeti Ağırlığı", 0.0, 1.0, 0.35, step=0.05)
w_prob = st.sidebar.slider("Gecikme Olasılığı Ağırlığı", 0.0, 1.0, 0.25, step=0.05)
w_var = st.sidebar.slider("VaR (%95) Ağırlığı", 0.0, 1.0, 0.20, step=0.05)
w_max = st.sidebar.slider("Maksimum Zarar Ağırlığı", 0.0, 1.0, 0.10, step=0.05)
w_base = st.sidebar.slider("Peşin Taşıma Ücreti Ağırlığı", 0.0, 1.0, 0.10, step=0.05)

total_weight = w_cost + w_prob + w_var + w_max + w_base
if total_weight == 0:
    st.sidebar.error("Tüm ağırlıklar 0 olamaz!")
    st.stop()

weights = {
    'Beklenen Risk Maliyeti ($)': w_cost / total_weight, 
    'Gecikme Olasılığı': w_prob / total_weight, 
    'VaR (%95) ($)': w_var / total_weight, 
    'Maksimum Zarar ($)': w_max / total_weight, 
    'Taşıma Ücreti ($)': w_base / total_weight
}

# ==========================================
# 4. ANALİZ VE SİMÜLASYON MOTORU
# ==========================================
shipping_modes = ['Standart Teslimat', 'İkinci Sınıf Teslimat', 'Birinci Sınıf Teslimat', 'Aynı Gün Teslimat']
base_costs = {'Standart Teslimat': 10, 'İkinci Sınıf Teslimat': 20, 'Birinci Sınıf Teslimat': 40, 'Aynı Gün Teslimat': 80}

results = []
simulations = {}

for mode in shipping_modes:
    input_data = pd.DataFrame({
        'Shipping Mode': [mode], 'Order Region': [selected_region],
        'Category Name': [selected_category], 'Sales per customer': [order_amount]
    })
    
    prob_late = model.predict_proba(input_data)[0][1]
    
    np.random.seed(42)
    penalty_dist = np.random.triangular(order_amount * (min_penalty/100), 
                                        order_amount * ((min_penalty+max_penalty)/200), 
                                        order_amount * (max_penalty/100), 
                                        num_iterations)
    is_late_sim = np.random.binomial(1, prob_late, num_iterations)
    financial_loss_sim = is_late_sim * penalty_dist
    
    expected_cost = np.mean(financial_loss_sim)
    var_95 = np.percentile(financial_loss_sim, 95)
    max_loss = np.max(financial_loss_sim)
    simulations[mode] = financial_loss_sim
    
    results.append({
        'Nakliye Modu': mode, 'Gecikme Olasılığı': prob_late,
        'Beklenen Risk Maliyeti ($)': expected_cost, 'VaR (%95) ($)': var_95,
        'Maksimum Zarar ($)': max_loss, 'Taşıma Ücreti ($)': base_costs[mode]
    })

results_df = pd.DataFrame(results)

# Dinamik MCDM Hesaplama (SAW)
mcdm_df = results_df.copy()
for col, weight in weights.items():
    min_val = mcdm_df[col].min()
    max_val = mcdm_df[col].max()
    if max_val != min_val:
        mcdm_df[col] = ((max_val - mcdm_df[col]) / (max_val - min_val)) * weight * 100
    else:
        mcdm_df[col] = weight * 100

results_df['Optimizasyon Skoru'] = mcdm_df[list(weights.keys())].sum(axis=1).round(1)
results_df = results_df.sort_values(by='Optimizasyon Skoru', ascending=False).reset_index(drop=True)

# ==========================================
# 5. ARAYÜZ ÇIKTILARI VE GÖRSELLEŞTİRME
# ==========================================
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🏆 Sistem Önerisi")
    best_mode = results_df.iloc[0]['Nakliye Modu']
    best_score = results_df.iloc[0]['Optimizasyon Skoru']
    st.success(f"**En Optimal Rota:** {best_mode} \n\n **Skor:** {best_score} / 100")
    
    display_df = results_df[['Nakliye Modu', 'Gecikme Olasılığı', 'VaR (%95) ($)', 'Optimizasyon Skoru']].copy()
    display_df['Gecikme Olasılığı'] = display_df['Gecikme Olasılığı'].map("{:.1%}".format)
    display_df['VaR (%95) ($)'] = display_df['VaR (%95) ($)' ].map("${:.2f}".format)
    st.dataframe(display_df, use_container_width=True)

with col2:
    st.subheader("📊 Optimizasyon Skorları Karşılaştırması")
    fig1 = px.bar(results_df, x='Nakliye Modu', y='Optimizasyon Skoru', 
                  color='Optimizasyon Skoru', color_continuous_scale='Viridis',
                  text='Optimizasyon Skoru')
    fig1.update_traces(textposition='outside')
    fig1.update_layout(yaxis=dict(range=[0, 110]), showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# Monte Carlo Bölümü
st.subheader("🎲 İnteraktif Monte Carlo Risk Simülasyonu")
mode_to_plot = st.selectbox("Dağılımını Görmek İstediğiniz Nakliye Modunu Seçin", shipping_modes, index=0)

sim_data = simulations[mode_to_plot]
positive_sim_data = sim_data[sim_data > 0]

if len(positive_sim_data) > 0:
    fig2 = px.histogram(x=positive_sim_data, nbins=50, 
                        labels={'x': 'Ek Risk Maliyeti ($)', 'y': 'Frekans'},
                        color_discrete_sequence=['#fca34d'], opacity=0.7)
    
    exp_cost = results_df[results_df['Nakliye Modu'] == mode_to_plot]['Beklenen Risk Maliyeti ($)'].values[0]
    var_val = results_df[results_df['Nakliye Modu'] == mode_to_plot]['VaR (%95) ($)'].values[0]
    
    fig2.add_vline(x=exp_cost, line_dash="dash", line_color="blue", 
                   annotation_text=f"Beklenen (${exp_cost:.2f})", annotation_position="top right")
    fig2.add_vline(x=var_val, line_dash="dash", line_color="red", 
                   annotation_text=f"%95 VaR (${var_val:.2f})", annotation_position="top right")
    
    fig2.update_layout(title_text=f"<b>{mode_to_plot}</b> İçin Gecikme Kaynaklı Finansal Risk Dağılımı",
                       bargap=0.05, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Bu nakliye modunda hiç gecikme öngörülmemiştir.")

st.divider()

# ==========================================
# 6. YENİ: RAPORLAMA VE ÇIKTI YÖNETİMİ
# ==========================================
st.subheader("📋 Raporlama ve Çıktı Yönetimi")

# Dinamik Metin Tabanlı Yönetici Raporu Oluşturma
current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
report_content = f"""======================================================================
     TEDARİK ZİNCİRİ ENTEGRE KARAR DESTEK SİSTEMİ (KDS) RAPORU
======================================================================
Rapor Üretim Tarihi: {current_time}

[1] SENARYO VE GİRDİ PARAMETRELERİ
----------------------------------------------------------------------
- Sipariş Bölgesi    : {selected_region}
- Ürün Kategorisi   : {selected_category}
- Sipariş Net Tutarı : ${order_amount:.2f}
- Simülasyon Derinliği: {num_iterations} İterasyon
- SLA Ceza Sınırları : %{min_penalty} - %{max_penalty} (Üçgensel Dağılım)

[2] ALGORİTMİK AĞIRLIKLANDIRMA KRİTERLERİ (MCDM)
----------------------------------------------------------------------
- Beklenen Risk Maliyeti Ağırlığı : %{weights['Beklenen Risk Maliyeti ($)']*100:.1f}
- Gecikme Olasılığı Ağırlığı      : %{weights['Gecikme Olasılığı']*100:.1f}
- Riske Maruz Değer (VaR %95) Ağ. : %{weights['VaR (%95) ($)']*100:.1f}
- Maksimum Olası Zarar Ağırlığı   : %{weights['Maksimum Zarar ($)']*100:.1f}
- Peşin Taşıma Ücreti Ağırlığı    : %{weights['Taşıma Ücreti ($)']*100:.1f}

[3] OPTİMİZASYON VE SİSTEM ÖNERİSİ SONUÇLARI
----------------------------------------------------------------------
🏆 EN OPTİMAL ROTASYON SEÇENEĞİ: {best_mode}
🎯 SİSTEM BAŞARI SKORU          : {best_score} / 100

[4] DETAYLI PERFORMANS VE RİSK MATRİSİ
----------------------------------------------------------------------
"""
for idx, row in results_df.iterrows():
    report_content += f"""Nakliye Modu: {row['Nakliye Modu']}
  - MCDM Öneri Skoru           : {row['Optimizasyon Skoru']} / 100
  - Yapay Zeka Gecikme İhtimali : {row['Gecikme Olasılığı']:.1%}
  - Beklenen Risk Maliyeti     : ${row['Beklenen Risk Maliyeti ($)']:.2f}
  - Riske Maruz Değer (VaR %95): ${row['VaR (%95) ($)']:.2f}
  - Maksimum Senaryo Zararı    : ${row['Maksimum Zarar ($)']:.2f}
  - Peşin Sabit Taşıma Ücreti  : ${row['Taşıma Ücreti ($)']:.2f}
----------------------------------------------------------------------
"""

report_content += """
[5] PRESCRIPTIVE (KURALCI) STRATEJİK YÖNETİM NOTU
----------------------------------------------------------------------
Bu sonuç, peşin lojistik maliyetleri ile stokastik gecikme risklerini 
birlikte optimize etmiştir. Düşük puan alan modların tercih edilmesi, 
SLA ihlalleri sebebiyle kâr marjında erozyona yol açacaktır. En yüksek 
skora sahip olan modun seçilmesi operasyonel emniyet için zorunludur.

======================================================================
     Rapor sonudur. KDS Altyapısı Tarafından Otomatik Üretilmiştir.
======================================================================
"""

# Çıktı Butonları Yan Yana Yerleştiriliyor
out_col1, out_col2 = st.columns(2)

with out_col1:
    st.markdown("#### 📄 Yönetici Özet Raporu")
    st.caption("Tüm simülasyon ve optimizasyon kararlarını içeren resmi metin çıktısını indirin.")
    st.download_button(
        label="📥 Yönetici Raporunu (.TXT) İndir",
        data=report_content,
        file_name=f"KDS_Yonetici_Raporu_{selected_region.replace(' ', '_')}.txt",
        mime="text/plain"
    )

with out_col2:
    st.markdown("#### 📊 Ham Veri Matrisi")
    st.caption("Modellerin hesapladığı tüm sayısal sonuçları ham Excel/CSV formatında kaydedin.")
    
    # Veriyi CSV formatına dönüştürme
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tüm Risk Verilerini (.CSV) İndir",
        data=csv_data,
        file_name=f"KDS_Risk_Matrisi_{selected_region.replace(' ', '_')}.csv",
        mime="text/csv"
    )

st.caption("Bu Karar Destek Sistemi, Makine Öğrenmesi algoritmaları kullanılarak geliştirilmiştir.")