import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# =========================================================
# 1. GÖRSELLEŞTİRME AYARLARI
# =========================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

print("SİSTEM BAŞLATILIYOR: Entegre Lojistik Karar Destek Modeli (V2)\n")

# =========================================================
# 2. VERİ YÜKLEME VE KÖKTEN TEMİZLİK
# =========================================================
print("1/4: Veri seti yükleniyor ve temizleniyor...")
path = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')
df.columns = [col.strip() for col in df.columns]

df['is_late'] = np.where(df['Delivery Status'].str.strip() == 'Late delivery', 1, 0)
df['order_date_dt'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
df['order_month'] = df['order_date_dt'].dt.month
df['order_weekday'] = df['order_date_dt'].dt.weekday

drop_columns = ['Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 'Product Image', 'Product Description', 'Customer Street', 'Order Zipcode', 'order date (DateOrders)', 'shipping date (DateOrders)', 'order_date_dt']
df_cleaned = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

for col in ['Shipping Mode', 'Order Region', 'Category Name', 'Market']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.title()

shipping_map = {'Standard Class': 'Standart Teslimat', 'Second Class': 'İkinci Sınıf Teslimat', 'First Class': 'Birinci Sınıf Teslimat', 'Same Day': 'Aynı Gün Teslimat'}
df_cleaned['Shipping Mode'] = df_cleaned['Shipping Mode'].map(shipping_map).fillna(df_cleaned['Shipping Mode'])

# =========================================================
# 3. ÖZELLİK SEÇİMİ VE VERİ BÖLME
# =========================================================
print("2/4: Veri yapay zeka algoritmasına hazırlanıyor...")
features = ['Shipping Mode', 'Order Region', 'Category Name', 'Market', 
            'Order Item Quantity', 'Product Price', 'Order Profit Per Order', 
            'order_month', 'order_weekday']

X = df_cleaned[features].copy()
y = df_cleaned['is_late']

X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 4. MAKİNE ÖĞRENMESİ (CATBOOST) EĞİTİMİ
# =========================================================
print("3/4: CatBoost Modeli lojistik örüntülerini öğreniyor...")
best_model = CatBoostClassifier(iterations=150, learning_rate=0.1, depth=8, verbose=0, random_state=42)
best_model.fit(X_train_scaled, y_train)

# =========================================================
# 5. VEKTÖRİZE MONTE CARLO SİMÜLASYONU VE MCDM
# =========================================================
print("4/4: 100.000 İterasyonlu Vektörize Simülasyon ve MCDM Çalıştırılıyor...\n")

sample_index = 42 
sample_raw = X.iloc[sample_index].copy()

profit_col = next(col for col in X_test.columns if 'profit' in col.lower())
price_col = next(col for col in X_test.columns if 'price' in col.lower())
product_price = X_test.iloc[sample_index][price_col]

shipping_modes = ['Standart Teslimat', 'İkinci Sınıf Teslimat', 'Birinci Sınıf Teslimat', 'Aynı Gün Teslimat']
alternatives_df = pd.DataFrame([sample_raw] * len(shipping_modes))
alternatives_df['Shipping Mode'] = shipping_modes

alt_encoded = pd.get_dummies(alternatives_df, drop_first=True)
alt_encoded = alt_encoded.reindex(columns=X_encoded.columns, fill_value=0)
alt_scaled = scaler.transform(alt_encoded)
alt_probabilities = best_model.predict_proba(alt_scaled)[:, 1]

np.random.seed(42)
num_simulations = 100000 
results_list = []

mode_params = {
    'Standart Teslimat': [10, 0.05, 0.10, 0.15],
    'İkinci Sınıf Teslimat': [25, 0.08, 0.12, 0.20],
    'Birinci Sınıf Teslimat': [45, 0.10, 0.15, 0.25],
    'Aynı Gün Teslimat': [80, 0.15, 0.25, 0.40] 
}

for i, mode in enumerate(shipping_modes):
    prob = alt_probabilities[i]
    base_c, min_p, mode_p, max_p = mode_params[mode]
    
    is_delayed = np.random.rand(num_simulations) < prob
    penalty_rates = np.random.triangular(min_p, mode_p, max_p, num_simulations)
    sim_costs = np.where(is_delayed, penalty_rates * product_price, 0.0)
    
    expected_risk_cost = np.mean(sim_costs)
    var_95 = np.percentile(sim_costs, 95)
    max_loss = np.max(sim_costs)
    total_expected_cost = base_c + expected_risk_cost
    
    results_list.append({
        'Nakliye Türü': mode,
        'Gecikme Olasılığı (%)': prob * 100,
        'Taşıma Ücreti ($)': base_c,
        'Maksimum Olası Zarar ($)': max_loss,
        'Maksimum Risk VaR ($)': var_95,
        'Toplam Beklenen Maliyet ($)': total_expected_cost
    })

mcdm_df = pd.DataFrame(results_list)

weights = {
    'Toplam Beklenen Maliyet ($)': 0.35, 
    'Gecikme Olasılığı (%)': 0.25,       
    'Maksimum Risk VaR ($)': 0.20,
    'Maksimum Olası Zarar ($)': 0.10,
    'Taşıma Ücreti ($)': 0.10
}

normalized_df = pd.DataFrame()
normalized_df['Nakliye Türü'] = mcdm_df['Nakliye Türü']

for col in weights.keys():
    max_val = mcdm_df[col].max()
    min_val = mcdm_df[col].min()
    if max_val == min_val:
        normalized_df[col] = 1.0
    else:
        normalized_df[col] = (max_val - mcdm_df[col]) / (max_val - min_val)

mcdm_df['Sistem Skoru (100)'] = 0
for col, weight in weights.items():
    mcdm_df['Sistem Skoru (100)'] += normalized_df[col] * weight * 100

mcdm_df = mcdm_df.sort_values(by='Sistem Skoru (100)', ascending=False).reset_index(drop=True)

pd.set_option('display.float_format', '{:.2f}'.format)
print("--- DETAYLI MCDM KARAR MATRİSİ (100.000 Senaryo) ---")
print(mcdm_df[['Nakliye Türü', 'Gecikme Olasılığı (%)', 'Toplam Beklenen Maliyet ($)', 'Maksimum Risk VaR ($)', 'Sistem Skoru (100)']].to_string(index=False))

# =========================================================
# 6. GÖRSELLEŞTİRME (Bar Grafiği + Radar Grafiği)
# =========================================================
fig = plt.figure(figsize=(16, 7))

ax1 = plt.subplot(1, 2, 1)
sns.barplot(data=mcdm_df, x='Nakliye Türü', y='Sistem Skoru (100)', hue='Nakliye Türü', palette='viridis', order=mcdm_df['Nakliye Türü'], legend=False, ax=ax1)
ax1.set_title('Nakliye Alternatifleri: Nihai Karar Skorları', fontweight='bold', pad=15)
ax1.set_ylabel('Sistem Öneri Skoru', fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylim(0, 110)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontweight='bold', xytext=(0, 5), textcoords='offset points')

categories = list(weights.keys())
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax2 = plt.subplot(1, 2, 2, polar=True)
ax2.set_theta_offset(pi / 2)
ax2.set_theta_direction(-1)
plt.xticks(angles[:-1], ['Top. Maliyet', 'Gecikme %', 'VaR', 'Maks. Zarar', 'Taşıma Ücreti'], color='black', size=10, fontweight='bold')
ax2.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20", "40", "60", "80", "100"], color="grey", size=8)
plt.ylim(0, 1.05)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

top_2_modes = mcdm_df['Nakliye Türü'].head(2).tolist()
for i, mode in enumerate(top_2_modes):
    values = normalized_df[normalized_df['Nakliye Türü'] == mode][categories].values.flatten().tolist()
    values += values[:1]
    ax2.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=mode)
    ax2.fill(angles, values, color=colors[i], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('En İyi 2 Nakliye Türünün Kriter Bazlı Radar Analizi\n(Dışa Yakınlık = Daha İyi Puan)', fontweight='bold', pad=20)

plt.tight_layout()
plt.show()