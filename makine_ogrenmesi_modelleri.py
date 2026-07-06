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
# 2. VERİ YÜKLEME VE ÖN İŞLEME (df_cleaned OLUŞTURMA)
# =========================================================
print("Veri yükleniyor ve ön işlemler yapılıyor...")
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

# Gizli Boşlukların Temizlenmesi
for col in ['Shipping Mode', 'Order Region', 'Category Name']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.title()

# Kategorik Verileri Türkçeleştirme Sözlükleri
shipping_map = {'Standard Class': 'Standart Teslimat', 'Second Class': 'İkinci Sınıf Teslimat', 'First Class': 'Birinci Sınıf Teslimat', 'Same Day': 'Aynı Gün Teslimat'}
region_map = {'Central America': 'Orta Amerika', 'Western Europe': 'Batı Avrupa', 'South America': 'Güney Amerika', 'Oceania': 'Okyanusya', 'Northern Europe': 'Kuzey Avrupa', 'Southeast Asia': 'Güneydoğu Asya', 'Southern Europe': 'Güney Avrupa', 'Caribbean': 'Karayipler', 'West Of Usa': 'ABD Batı', 'Eastern Asia': 'Doğu Asya', 'East Of Usa': 'ABD Doğu', 'Us Center': 'ABD Merkez', 'South Of Usa': 'ABD Güney', 'South Asia': 'Güney Asya', 'Central Asia': 'Orta Asya', 'North Africa': 'Kuzey Afrika', 'Central Africa': 'Orta Afrika', 'West Africa': 'Batı Afrika', 'Southern Africa': 'Güney Afrika', 'East Africa': 'Doğu Afrika', 'Canada': 'Kanada'}
category_map = {'Cleats': 'Kramponlar', "Men'S Footwear": 'Erkek Ayakkabı', "Women'S Apparel": 'Kadın Giyim', 'Indoor/Outdoor Games': 'İç/Dış Mekan Oyunları', 'Fishing': 'Balıkçılık', 'Water Sports': 'Su Sporları', 'Camping & Hiking': 'Kamp ve Doğa Yürüyüşü', 'Cardio Equipment': 'Kardiyo Ekipmanları', 'Shop By Sport': 'Spora Göre Alışveriş', 'Electronics': 'Elektronik', 'Computers': 'Bilgisayarlar', 'Accessories': 'Aksesuarlar', 'Golf Apparel': 'Golf Giyim', 'Children\'S Clothing': 'Çocuk Giyim', 'Toys': 'Oyuncaklar', 'Video Games': 'Video Oyunları', 'Health And Beauty': 'Sağlık ve Güzellik', 'Pet Supplies': 'Evcil Hayvan Ürünleri', 'Cameras': 'Kameralar', 'Books': 'Kitaplar', 'Hardware': 'Donanım', 'Music': 'Müzik', 'Crafts': 'El Sanatları', 'Golf Shoes': 'Golf Ayakkabıları', 'Lacrosse': 'Lakros', 'Tennis & Racquet': 'Tenis ve Raket', "Girls' Apparel": 'Kız Çocuk Giyim', "Boys' Apparel": 'Erkek Çocuk Giyim'}

df_cleaned['Shipping Mode'] = df_cleaned['Shipping Mode'].map(shipping_map).fillna(df_cleaned['Shipping Mode'])
df_cleaned['Order Region'] = df_cleaned['Order Region'].map(region_map).fillna(df_cleaned['Order Region'])
df_cleaned['Category Name'] = df_cleaned['Category Name'].map(category_map).fillna(df_cleaned['Category Name'])

print(f"Veri başarıyla hazırlandı. Toplam Satır: {df_cleaned.shape[0]}\n")

# =========================================================
# 3. ÖZELLİK SEÇİMİ VE ENCODING
# =========================================================
features = ['Shipping Mode', 'Order Region', 'Category Name', 'Market', 
            'Order Item Quantity', 'Product Price', 'Order Profit Per Order', 
            'order_month', 'order_weekday']

# Sadece modele girecek özellikleri seçiyoruz
X = df_cleaned[features].copy()
y = df_cleaned['is_late']

# Kategorik değişkenleri (Metinleri) 0 ve 1'lere dönüştürüyoruz (One-Hot Encoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# =========================================================
# 4. TRAIN - TEST AYRIMI VE ÖLÇEKLENDİRME (SCALING)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Modeller eğitiliyor, lütfen bekleyin...\n")

# =========================================================
# 5. MODELLERİN TANIMLANMASI VE EĞİTİLMESİ
# =========================================================
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=150, learning_rate=0.1, depth=8, verbose=0, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': roc_auc})
    print(f"✅ {name} eğitimi tamamlandı.")

results_df = pd.DataFrame(results)

# =========================================================
# 6. SONUÇLARIN GÖRSELLEŞTİRİLEREK KARŞILAŞTIRILMASI
# =========================================================
results_melted = results_df.melt(id_vars='Model', var_name='Metrik', value_name='Skor')

plt.figure(figsize=(12, 7))
ax = sns.barplot(data=results_melted, x='Model', y='Skor', hue='Metrik', palette='Set2')

plt.title('Makine Öğrenmesi Algoritmalarının Performans Karşılaştırması', fontweight='bold', pad=15)
plt.ylabel('Skor (0 - 1 arası)', fontweight='bold')
plt.xlabel('Algoritma', fontweight='bold')
plt.ylim(0.4, 1.05)

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

plt.legend(loc='lower right', title='Değerlendirme Metriği')
plt.tight_layout()
plt.show()

print("\n--- Model Performans Tablosu ---")
print(results_df.sort_values(by='F1-Score', ascending=False).to_string(index=False))