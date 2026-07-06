import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. VERİ YÜKLEME VE TEMEL DÜZENLEMELER
# ---------------------------------------------------------
path = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')

# Sütun isimlerindeki gereksiz boşlukları temizleme
df.columns = [col.strip() for col in df.columns]

# Hedef değişken filtresi için teslimat durumunu Türkçeleştirme
delivery_status_map = {
    'Late delivery': 'Gecikmiş Teslimat',
    'Advance shipping': 'Erken Teslimat',
    'Shipping on time': 'Zamanında Teslimat',
    'Shipping canceled': 'İptal Edilmiş Teslimat'
}
df['Delivery Status'] = df['Delivery Status'].map(delivery_status_map).fillna(df['Delivery Status'])

# ---------------------------------------------------------
# 2. GEREKSİZ VE KİŞİSEL VERİLERİN TEMİZLENMESİ
# ---------------------------------------------------------
drop_columns = [
    'Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname',
    'Product Image', 'Product Description', 'Customer Street', 'Order Zipcode'
]
df_cleaned = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

# ---------------------------------------------------------
# 3. HEDEF DEĞİŞKENLERİN (TARGET) OLUŞTURULMASI
# ---------------------------------------------------------
# Sınıflandırma modeli için (1: Gecikti, 0: Gecikmedi)
df_cleaned['is_late'] = np.where(df_cleaned['Delivery Status'] == 'Gecikmiş Teslimat', 1, 0)

# Regresyon modeli için (Gecikme Gün Sayısı)
# Negatif değerler erken teslimatı, pozitif değerler kaç gün geciktiğini gösterir.
df_cleaned['delay_days'] = df_cleaned['Days for shipping (real)'] - df_cleaned['Days for shipment (scheduled)']

# ---------------------------------------------------------
# 4. ZAMANSAL ÖZELLİK MÜHENDİSLİĞİ (TIME FEATURE ENGINEERING)
# ---------------------------------------------------------
# Tarih formatına çevirme
df_cleaned['order_date_dt'] = pd.to_datetime(df_cleaned['order date (DateOrders)'], errors='coerce')

# Ay, Yıl, Gün ve Haftanın Günü bilgilerini çıkarma
df_cleaned['order_year'] = df_cleaned['order_date_dt'].dt.year
df_cleaned['order_month'] = df_cleaned['order_date_dt'].dt.month
df_cleaned['order_day'] = df_cleaned['order_date_dt'].dt.day
df_cleaned['order_weekday'] = df_cleaned['order_date_dt'].dt.weekday # 0: Pazartesi, 6: Pazar

# İşlem biten eski tarih sütunlarını veri setinden atma
df_cleaned = df_cleaned.drop(columns=['order date (DateOrders)', 'order_date_dt', 'shipping date (DateOrders)'], errors='ignore')

# ---------------------------------------------------------
# 5. KONTROL ÇIKTILARI
# ---------------------------------------------------------
print("Veri Ön İşleme Tamamlandı!\n")
print(f"Temizlenmiş Veri Seti Boyutu: {df_cleaned.shape}\n")
print("Sınıflandırma Hedef Değişkeni (is_late) Dağılımı (%):")
print(df_cleaned['is_late'].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt

# Görsel boyutu ve ayarları
plt.figure(figsize=(8, 8))

# is_late sütununun değer sayılarını alma
late_counts = df_cleaned['is_late'].value_counts()

# Grafik etiketleri ve renkleri
labels = ['Gecikmiş Teslimat (1)', 'Zamanında/Diğer (0)']
colors = ['#ff6f69', '#6b5b95'] # Gecikme için dikkat çekici kırmızı/kavun, zamanında için mor/mavi tonu
explode = (0.05, 0) # Gecikmiş teslimat dilimini hafifçe ayırarak vurgulama

# Pasta grafiğini oluşturma
plt.pie(late_counts, labels=labels, colors=colors, autopct='%1.2f%%', 
        startangle=90, explode=explode, shadow=True, 
        textprops={'fontsize': 13, 'fontweight': 'bold'})

# Başlık ekleme
plt.title('Sınıflandırma Hedef Değişkeni: Teslimat Gecikme Oranları', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

# --- GİZLİ BOŞLUKLARI TEMİZLEME VE FORMATLAMA ---
for col in ['Shipping Mode', 'Order Region', 'Category Name']:
    if col in df_cleaned.columns:
        # Önce boşlukları silip, baş harflerini büyük yaparak formatı standartlaştırıyoruz
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.title()

# --- GENİŞLETİLMİŞ KATEGORİK VERİ TÜRKÇELEŞTİRME SÖZLÜKLERİ ---

# 1. Nakliye Türleri
shipping_map = {
    'Standard Class': 'Standart Teslimat',
    'Second Class': 'İkinci Sınıf Teslimat',
    'First Class': 'Birinci Sınıf Teslimat',
    'Same Day': 'Aynı Gün Teslimat'
}
if 'Shipping Mode' in df_cleaned.columns:
    df_cleaned['Shipping Mode'] = df_cleaned['Shipping Mode'].map(shipping_map).fillna(df_cleaned['Shipping Mode'])

# 2. Bölgeler (Tüm Olasılıklar Genişletildi)
region_map = {
    'Central America': 'Orta Amerika', 'Western Europe': 'Batı Avrupa',
    'South America': 'Güney Amerika', 'Oceania': 'Okyanusya',
    'Northern Europe': 'Kuzey Avrupa', 'Southeast Asia': 'Güneydoğu Asya',
    'Southern Europe': 'Güney Avrupa', 'Caribbean': 'Karayipler',
    'West Of Usa': 'ABD Batı', 'Eastern Asia': 'Doğu Asya',
    'East Of Usa': 'ABD Doğu', 'Us Center': 'ABD Merkez',
    'South Of Usa': 'ABD Güney', 'South Asia': 'Güney Asya',
    'Central Asia': 'Orta Asya', 'North Africa': 'Kuzey Afrika',
    'Central Africa': 'Orta Afrika', 'West Africa': 'Batı Afrika',
    'Southern Africa': 'Güney Afrika', 'East Africa': 'Doğu Afrika',
    'Canada': 'Kanada'
}
if 'Order Region' in df_cleaned.columns:
    df_cleaned['Order Region'] = df_cleaned['Order Region'].map(region_map).fillna(df_cleaned['Order Region'])

# 3. Ürün Kategorileri (Tüm Olasılıklar Genişletildi)
category_map = {
    'Cleats': 'Kramponlar', "Men'S Footwear": 'Erkek Ayakkabı',
    "Women'S Apparel": 'Kadın Giyim', 'Indoor/Outdoor Games': 'İç/Dış Mekan Oyunları',
    'Fishing': 'Balıkçılık', 'Water Sports': 'Su Sporları',
    'Camping & Hiking': 'Kamp ve Doğa Yürüyüşü', 'Cardio Equipment': 'Kardiyo Ekipmanları',
    'Shop By Sport': 'Spora Göre Alışveriş', 'Electronics': 'Elektronik',
    'Computers': 'Bilgisayarlar', 'Accessories': 'Aksesuarlar',
    'Golf Apparel': 'Golf Giyim', 'Children\'S Clothing': 'Çocuk Giyim',
    'Toys': 'Oyuncaklar', 'Video Games': 'Video Oyunları',
    'Health And Beauty': 'Sağlık ve Güzellik', 'Pet Supplies': 'Evcil Hayvan Ürünleri',
    'Cameras': 'Kameralar', 'Books': 'Kitaplar',
    'Hardware': 'Donanım', 'Music': 'Müzik', 'Crafts': 'El Sanatları',
    'Golf Shoes': 'Golf Ayakkabıları', 'Lacrosse': 'Lakros', 'Tennis & Racquet': 'Tenis ve Raket',
    "Girls' Apparel": 'Kız Çocuk Giyim', "Boys' Apparel": 'Erkek Çocuk Giyim'
}
if 'Category Name' in df_cleaned.columns:
    df_cleaned['Category Name'] = df_cleaned['Category Name'].map(category_map).fillna(df_cleaned['Category Name'])

# Gecikme durumunu grafik lejantı için isimlendirelim
if 'is_late' in df_cleaned.columns:
    df_cleaned['Gecikme Durumu'] = df_cleaned['is_late'].map({1: 'Gecikti', 0: 'Zamanında/Erken'})

# ---------------------------------------------------------
# GRAFİK 1: NAKLİYE TÜRÜNE GÖRE GECİKME RİSKİ
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
ax1 = sns.countplot(data=df_cleaned, x='Shipping Mode', hue='Gecikme Durumu', palette={'Gecikti': '#ff6f69', 'Zamanında/Erken': '#6b5b95'})
plt.title('Nakliye Türüne Göre Sipariş Hacmi ve Gecikme Dağılımı', fontweight='bold', pad=15)
plt.xlabel('Nakliye Türü', fontweight='bold')
plt.ylabel('Sipariş Sayısı', fontweight='bold')

for p in ax1.patches:
    if p.get_height() > 0:
        ax1.annotate(f'{int(p.get_height()/1000)}k', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.legend(title='Durum')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# GRAFİK 2: EN ÇOK SİPARİŞ ALAN İLK 10 BÖLGE
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
top_regions = df_cleaned['Order Region'].value_counts().head(10)
ax2 = sns.barplot(y=top_regions.index, x=top_regions.values, palette='viridis')
plt.title('En Yüksek Teslimat Hacmine Sahip İlk 10 Bölge', fontweight='bold', pad=15)
plt.xlabel('Toplam Sipariş Sayısı', fontweight='bold')
plt.ylabel('Teslimat Bölgesi', fontweight='bold')

for i, v in enumerate(top_regions.values):
    ax2.text(v + 500, i, f"{int(v):,}", color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# GRAFİK 3: EN KARLI İLK 10 ÜRÜN KATEGORİSİ
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
top_profit_categories = df_cleaned.groupby('Category Name')['Order Profit Per Order'].sum().sort_values(ascending=False).head(10)
ax3 = sns.barplot(y=top_profit_categories.index, x=top_profit_categories.values, palette='crest')
plt.title('Tedarik Zincirinde En Fazla Kar Getiren İlk 10 Ürün Kategorisi', fontweight='bold', pad=15)
plt.xlabel('Toplam Kar ($)', fontweight='bold')
plt.ylabel('Ürün Kategorisi', fontweight='bold')

for i, v in enumerate(top_profit_categories.values):
    ax3.text(v + 10000, i, f"${int(v):,}", color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Görselleştirme sonrası geçici sütunu silelim
if 'Gecikme Durumu' in df_cleaned.columns:
    df_cleaned.drop(columns=['Gecikme Durumu'], inplace=True)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. GÖRSELLEŞTİRME AYARLARI
# =========================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

# =========================================================
# 2. VERİ YÜKLEME VE FREKANS HESAPLAMA
# =========================================================
print("Veri yükleniyor ve tam sayılarla nakliye modu grafiği oluşturuluyor...")
path = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')
df.columns = [col.strip() for col in df.columns]

# Teslimat Durumunu 2 sınıfa indirgeme (Gecikti vs Zamanında/Erken)
df['Durum'] = np.where(df['Delivery Status'].str.strip() == 'Late delivery', 'Gecikti', 'Zamanında/Erken')

# Nakliye türlerini Türkçeleştirme
df['Shipping Mode'] = df['Shipping Mode'].astype(str).str.strip().str.title()
shipping_map = {
    'Standard Class': 'Standart Teslimat', 
    'Second Class': 'İkinci Sınıf Teslimat', 
    'First Class': 'Birinci Sınıf Teslimat', 
    'Same Day': 'Aynı Gün Teslimat'
}
df['Shipping Mode'] = df['Shipping Mode'].map(shipping_map).fillna(df['Shipping Mode'])

# Frekansları gruplayıp sayma
counts_df = df.groupby(['Shipping Mode', 'Durum']).size().reset_index(name='Sipariş Sayısı')

# Konsol çıktısı için tabloyu pivotlama
pivot_df = counts_df.pivot(index='Shipping Mode', columns='Durum', values='Sipariş Sayısı').fillna(0).astype(int)
print("\n--- NAKLİYE TÜRÜNE GÖRE SİPARİŞ HACMİ VE GECİKME DURUMU (TAM SAYILAR) ---")
print(pivot_df.to_string())

# =========================================================
# 3. GRAFİK ÇİZİMİ
# =========================================================
plt.figure(figsize=(12, 8))

# Grafikteki eski sıralamayı korumak için
order_list = ['Standart Teslimat', 'Birinci Sınıf Teslimat', 'İkinci Sınıf Teslimat', 'Aynı Gün Teslimat']

# Orijinal grafikteki renk tonlarına yakın renkler (Mor/Kırmızımsı)
custom_palette = {'Zamanında/Erken': '#645b87', 'Gecikti': '#ec7e7a'}

ax = sns.barplot(data=counts_df, x='Shipping Mode', y='Sipariş Sayısı', hue='Durum', 
                 palette=custom_palette, order=order_list)

plt.title('Nakliye Türüne Göre Sipariş Hacmi ve Gecikme Dağılımı (Tam Sayılar)', fontweight='bold', pad=15)
plt.xlabel('Nakliye Türü (Shipping Mode)', fontweight='bold')
plt.ylabel('Frekans (Sipariş Sayısı)', fontweight='bold')
plt.ylim(0, counts_df['Sipariş Sayısı'].max() * 1.15) # Etiketler için üstte boşluk bırakıyoruz

# Çubukların üzerine tam sayıları yazdırma (Binlik ayracı olarak nokta)
for p in ax.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        label_text = f"{int(height):,}".replace(",", ".")
        ax.annotate(label_text, 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

plt.legend(title='Durum', title_fontsize='12', fontsize='11', loc='upper right')
plt.tight_layout()
plt.show()