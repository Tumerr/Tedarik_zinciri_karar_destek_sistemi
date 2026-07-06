import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. GÖRSELLEŞTİRME AYARLARI
# =========================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

# =========================================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME (KÖKTEN TEMİZLİK)
# =========================================================
print("1/4: Veri seti yükleniyor ve ön işlemler yapılıyor...")
path = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')
df.columns = [col.strip() for col in df.columns]

# Hedef Değişkenin Oluşturulması (is_late: Gecikti mi?)
df['is_late'] = np.where(df['Delivery Status'].str.strip() == 'Late delivery', 1, 0)

# Tarih Değişkenlerinin Türetilmesi
df['order_date_dt'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
df['order_month'] = df['order_date_dt'].dt.month
df['order_weekday'] = df['order_date_dt'].dt.weekday

# Gereksiz ve Kişisel Verilerin Atılması
drop_columns = [
    'Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname',
    'Product Image', 'Product Description', 'Customer Street', 'Order Zipcode',
    'order date (DateOrders)', 'shipping date (DateOrders)', 'order_date_dt'
]
df_cleaned = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

# Gizli Boşlukların Temizlenmesi ve Formatın Standartlaştırılması
for col in ['Shipping Mode', 'Order Region', 'Category Name', 'Market']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.title()

# Genişletilmiş Kategorik Veri Türkçeleştirme Sözlükleri
shipping_map = {'Standard Class': 'Standart Teslimat', 'Second Class': 'İkinci Sınıf Teslimat', 'First Class': 'Birinci Sınıf Teslimat', 'Same Day': 'Aynı Gün Teslimat'}
region_map = {
    'Central America': 'Orta Amerika', 'Western Europe': 'Batı Avrupa', 'South America': 'Güney Amerika', 
    'Oceania': 'Okyanusya', 'Northern Europe': 'Kuzey Avrupa', 'Southeast Asia': 'Güneydoğu Asya', 
    'Southern Europe': 'Güney Avrupa', 'Caribbean': 'Karayipler', 'West Of Usa': 'ABD Batı', 
    'Eastern Asia': 'Doğu Asya', 'East Of Usa': 'ABD Doğu', 'Us Center': 'ABD Merkez', 
    'South Of Usa': 'ABD Güney', 'South Asia': 'Güney Asya', 'Central Asia': 'Orta Asya', 
    'North Africa': 'Kuzey Afrika', 'Central Africa': 'Orta Afrika', 'West Africa': 'Batı Afrika', 
    'Southern Africa': 'Güney Afrika', 'East Africa': 'Doğu Afrika', 'Canada': 'Kanada'
}
category_map = {
    'Cleats': 'Kramponlar', "Men'S Footwear": 'Erkek Ayakkabı', "Women'S Apparel": 'Kadın Giyim', 
    'Indoor/Outdoor Games': 'İç/Dış Mekan Oyunları', 'Fishing': 'Balıkçılık', 'Water Sports': 'Su Sporları', 
    'Camping & Hiking': 'Kamp ve Doğa Yürüyüşü', 'Cardio Equipment': 'Kardiyo Ekipmanları', 
    'Shop By Sport': 'Spora Göre Alışveriş', 'Electronics': 'Elektronik', 'Computers': 'Bilgisayarlar', 
    'Accessories': 'Aksesuarlar', 'Golf Apparel': 'Golf Giyim', 'Children\'S Clothing': 'Çocuk Giyim', 
    'Toys': 'Oyuncaklar', 'Video Games': 'Video Oyunları', 'Health And Beauty': 'Sağlık ve Güzellik', 
    'Pet Supplies': 'Evcil Hayvan Ürünleri', 'Cameras': 'Kameralar', 'Books': 'Kitaplar', 
    'Hardware': 'Donanım', 'Music': 'Müzik', 'Crafts': 'El Sanatları', 'Golf Shoes': 'Golf Ayakkabıları', 
    'Lacrosse': 'Lakros', 'Tennis & Racquet': 'Tenis ve Raket', "Girls' Apparel": 'Kız Çocuk Giyim', 
    "Boys' Apparel": 'Erkek Çocuk Giyim'
}

df_cleaned['Shipping Mode'] = df_cleaned['Shipping Mode'].map(shipping_map).fillna(df_cleaned['Shipping Mode'])
df_cleaned['Order Region'] = df_cleaned['Order Region'].map(region_map).fillna(df_cleaned['Order Region'])
df_cleaned['Category Name'] = df_cleaned['Category Name'].map(category_map).fillna(df_cleaned['Category Name'])

print(f"Veri başarıyla hazırlandı. Toplam Satır: {df_cleaned.shape[0]}\n")

# =========================================================
# 3. ÖZELLİK SEÇİMİ VE BÖLME (TRAIN-TEST SPLIT)
# =========================================================
print("2/4: Özellikler seçiliyor ve veri eğitim/test olarak bölünüyor...")
features = ['Shipping Mode', 'Order Region', 'Category Name', 'Market', 
            'Order Item Quantity', 'Product Price', 'Order Profit Per Order', 
            'order_month', 'order_weekday']

X = df_cleaned[features].copy()
y = df_cleaned['is_late']

# Sayısal değerleri simülasyonda kullanabilmek adına indeksleri koruyarak kodlama yapıyoruz
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 4. MAKİNE ÖĞRENMESİ MODELLERİNİN EĞİTİLMESİ
# =========================================================
print("3/4: Makine öğrenmesi modelleri eğitiliyor (Lütfen bekleyin)...")
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=150, learning_rate=0.1, depth=8, verbose=0, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"✅ {name} başarıyla eğitildi.")

# =========================================================
# 5. MONTE CARLO SİMÜLASYONU (ENTEGRE KARAR DESTEK SİSTEMİ)
# =========================================================
print("\n4/4: Seçilen sipariş için Monte Carlo Risk Simülasyonu başlatılıyor...")

best_model = models["CatBoost"]
sample_index = 42  # Analiz edilmek üzere test setinden seçilen örnek sipariş indeksi
sample_order = X_test_scaled[sample_index].reshape(1, -1)

# Modelden spesifik siparişe ait gecikme olasılığının çekilmesi
delay_probability = best_model.predict_proba(sample_order)[0][1]

# Orijinal finansal değerlerin doğrulanması
order_profit = X_test.iloc[sample_index]['Order Profit Per Order']
product_price = X_test.iloc[sample_index]['Product Price']

print(f"\n[Sipariş Profili] Anlık Kar: ${order_profit:.2f} | Ürün Satış Fiyatı: ${product_price:.2f}")
print(f"[Model Öngörüsü] Yapay Zeka Gecikme Olasılığı: %{delay_probability*100:.2f}")

num_simulations = 10000
simulated_costs = []

min_penalty_rate = 0.05
max_penalty_rate = 0.20

np.random.seed(42)
for _ in range(num_simulations):
    # Olasılıksal olarak gecikme durumunun simüle edilmesi
    is_delayed = np.random.rand() < delay_probability
    
    if is_delayed:
        # Gecikme gerçekleştiğinde lojistik süreçlerdeki belirsizliği temsil eden üçgensel dağılım
        penalty_rate = np.random.triangular(min_penalty_rate, 0.10, max_penalty_rate)
        extra_cost = product_price * penalty_rate
    else:
        extra_cost = 0.0
        
    simulated_costs.append(extra_cost)

expected_extra_cost = np.mean(simulated_costs)
var_95 = np.percentile(simulated_costs, 95)

print(f"\n[Simülasyon Sonucu] Beklenen Ortalama Ek Risk Maliyeti: ${expected_extra_cost:.2f}")
print(f"[Risk Analizi] %95 Güven Düzeyinde Riske Maruz Değer (VaR): En Kötü Senaryoda ${var_95:.2f} Ek Maliyet")

# Simülasyon Sonuç Grafiği (Tamamen Türkçe)
plt.figure(figsize=(10, 6))
sns.histplot(simulated_costs, bins=50, color='darkorange', kde=True, stat="density")

plt.axvline(expected_extra_cost, color='blue', linestyle='dashed', linewidth=2, label=f'Beklenen Maliyet (${expected_extra_cost:.2f})')
plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'%95 VaR Risk Seviyesi (${var_95:.2f})')

plt.title('Monte Carlo Simülasyonu: Gecikme Kaynaklı Ek Maliyet Dağılımı', fontweight='bold', pad=15)
plt.xlabel('Gecikme Kaynaklı Operasyonel Ek Maliyet ($)', fontweight='bold')
plt.ylabel('Olasılık Yoğunluğu', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

print("Çok Kriterli Karar Verme (MCDM) Modeli Çalıştırılıyor...\n")

# ---------------------------------------------------------
# 1. ALTERNATİF SENARYOLARIN ÜRETİLMESİ
# ---------------------------------------------------------
# Orijinal siparişin ham (encode edilmemiş) özelliklerini alıyoruz
sample_raw = X.iloc[sample_index].copy()

# Sisteme sunacağımız 4 alternatif nakliye stratejisi
shipping_modes = ['Standart Teslimat', 'İkinci Sınıf Teslimat', 'Birinci Sınıf Teslimat', 'Aynı Gün Teslimat']

# Her strateji için siparişi kopyalayıp nakliye türünü değiştiriyoruz
alternatives_df = pd.DataFrame([sample_raw] * len(shipping_modes))
alternatives_df['Shipping Mode'] = shipping_modes

# Alternatifleri modelin anlayacağı formata (One-Hot Encoding) çeviriyoruz
# Orijinal eğitim setiyle sütun uyumsuzluğu olmaması için reindex yapıyoruz
alt_encoded = pd.get_dummies(alternatives_df, drop_first=True)
alt_encoded = alt_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Veriyi ölçeklendiriyoruz
alt_scaled = scaler.transform(alt_encoded)

# ---------------------------------------------------------
# 2. YAPAY ZEKA TAHMİNLERİ VE MONTE CARLO (Her Alternatif İçin)
# ---------------------------------------------------------
best_model = models["CatBoost"]
alt_probabilities = best_model.predict_proba(alt_scaled)[:, 1]

np.random.seed(42)
num_simulations = 5000
results_list = []

for i, mode in enumerate(shipping_modes):
    prob = alt_probabilities[i]
    sim_costs = []
    
    # Her nakliye türünün kendine has içsel bir baz maliyeti olduğunu varsayıyoruz (MCDM Kriteri)
    # Standart ucuzdur ama gecikme riski fazladır. Aynı Gün pahalıdır ama kesindir.
    base_cost = {'Standart Teslimat': 10, 'İkinci Sınıf Teslimat': 25, 'Birinci Sınıf Teslimat': 45, 'Aynı Gün Teslimat': 80}
    
    for _ in range(num_simulations):
        is_delayed = np.random.rand() < prob
        if is_delayed:
            penalty = np.random.triangular(0.05, 0.10, 0.20) * product_price
        else:
            penalty = 0.0
        sim_costs.append(penalty)
        
    expected_risk_cost = np.mean(sim_costs)
    var_95 = np.percentile(sim_costs, 95)
    
    # Toplam Beklenen Maliyet = Taşıma Ücreti + Risk (Gecikme) Maliyeti
    total_expected_cost = base_cost[mode] + expected_risk_cost
    
    results_list.append({
        'Nakliye Türü': mode,
        'Gecikme Olasılığı (%)': prob * 100,
        'Taşıma Ücreti ($)': base_cost[mode],
        'Risk Maliyeti ($)': expected_risk_cost,
        'Maksimum Risk VaR ($)': var_95,
        'Toplam Beklenen Maliyet ($)': total_expected_cost
    })

mcdm_df = pd.DataFrame(results_list)

# ---------------------------------------------------------
# 3. MCDM - PUANLAMA VE SIRALAMA (Ağırlıklı Karar Modeli)
# ---------------------------------------------------------
# Yöneticinin Karar Ağırlıkları (Toplamı 1.0 olmalı)
weights = {
    'Gecikme Olasılığı (%)': 0.30,       # Risk olasılığını minimize et
    'Toplam Beklenen Maliyet ($)': 0.50, # Cepten çıkacak toplam parayı minimize et
    'Maksimum Risk VaR ($)': 0.20        # En kötü senaryodan kaçın
}

# Min-Max Normalizasyonu (Maliyetleri ve riskleri minimize etmeye çalıştığımız için ters çeviriyoruz)
# Formül: (Max - Değer) / (Max - Min)
normalized_df = pd.DataFrame()
normalized_df['Nakliye Türü'] = mcdm_df['Nakliye Türü']

for col in weights.keys():
    max_val = mcdm_df[col].max()
    min_val = mcdm_df[col].min()
    if max_val == min_val:
        normalized_df[col] = 1.0
    else:
        normalized_df[col] = (max_val - mcdm_df[col]) / (max_val - min_val)

# Ağırlıklı Toplam Skoru Hesaplama (Skor 100'e ne kadar yakınsa o kadar iyi)
mcdm_df['Karar Skoru (100 Üzerinden)'] = 0
for col, weight in weights.items():
    mcdm_df['Karar Skoru (100 Üzerinden)'] += normalized_df[col] * weight * 100

mcdm_df = mcdm_df.sort_values(by='Karar Skoru (100 Üzerinden)', ascending=False).reset_index(drop=True)

print("--- ALTERNATİF NAKLİYE TÜRLERİ KARAR MATRİSİ ---")
print(mcdm_df[['Nakliye Türü', 'Gecikme Olasılığı (%)', 'Toplam Beklenen Maliyet ($)', 'Karar Skoru (100 Üzerinden)']].to_string(index=False))
print(f"\n🏆 SİSTEM ÖNERİSİ: Bu sipariş için en optimal ve güvenli seçenek '{mcdm_df.iloc[0]['Nakliye Türü']}' olarak belirlenmiştir.")

# ---------------------------------------------------------
# 4. KARAR DESTEK SİSTEMİ GÖRSELLEŞTİRMESİ
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

ax = sns.barplot(data=mcdm_df, x='Nakliye Türü', y='Karar Skoru (100 Üzerinden)', palette='coolwarm_r', order=mcdm_df['Nakliye Türü'])

plt.title('MCDM Algoritması: Nakliye Türleri İçin Optimizasyon Skorları', fontweight='bold', pad=15)
plt.xlabel('Nakliye Alternatifleri', fontweight='bold')
plt.ylabel('Sistem Öneri Skoru (100 = En İyi)', fontweight='bold')
plt.ylim(0, 110)

# Skorları çubukların üzerine yazdırma
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f} Puan', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()