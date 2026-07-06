import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------
# 1. GÖRSELLEŞTİRME AYARLARI
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14

# ---------------------------------------------------------
# 2. VERİ YÜKLEME VE ÖN İŞLEME
# ---------------------------------------------------------
path = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')

df.columns = [col.strip() for col in df.columns]

# --- EKSEN ETİKETLERİNİ TÜRKÇELEŞTİRME SÖZLÜKLERİ ---
delivery_status_map = {
    'Late delivery': 'Gecikmiş Teslimat',
    'Advance shipping': 'Erken Teslimat',
    'Shipping on time': 'Zamanında Teslimat',
    'Shipping canceled': 'İptal Edilmiş Teslimat'
}
df['Delivery Status'] = df['Delivery Status'].map(delivery_status_map).fillna(df['Delivery Status'])

market_map = {
    'Europe': 'Avrupa',
    'LATAM': 'Latin Amerika',
    'Pacific Asia': 'Asya Pasifik',
    'USCA': 'Kuzey Amerika',
    'Africa': 'Afrika'
}
df['Market'] = df['Market'].map(market_map).fillna(df['Market'])


print("Veri Seti Boyutu:", df.shape)
print("\nEksik Veri Analizi (İlk 10 Sütun):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ---------------------------------------------------------
# 3. HEDEF DEĞİŞKEN ANALİZİ: TESLİMAT DURUMU
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='Delivery Status', order=df['Delivery Status'].value_counts().index)

plt.title('Siparişlerin Teslimat Durumu Dağılımı', pad=15, fontweight='bold')
plt.xlabel('Teslimat Durumu', fontweight='bold')
plt.ylabel('Sipariş Sayısı', fontweight='bold')
plt.xticks(rotation=15)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=11, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. PLANLANAN VE GERÇEKLEŞEN TESLİMAT SÜRELERİ
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

Süre_df = df[['Days for shipping (real)', 'Days for shipment (scheduled)']].melt()
Süre_df['variable'] = Süre_df['variable'].map({
    'Days for shipping (real)': 'Gerçekleşen Süre (Gün)',
    'Days for shipment (scheduled)': 'Planlanan Süre (Gün)'
})

ax2 = sns.countplot(data=Süre_df, x='value', hue='variable')

plt.title('Planlanan ve Gerçekleşen Teslimat Sürelerinin Karşılaştırması', pad=15, fontweight='bold')
plt.xlabel('Gün Sayısı', fontweight='bold')
plt.ylabel('Frekans (Sipariş Sayısı)', fontweight='bold')
plt.legend(title='Süre Tipi')

for p in ax2.patches:
    if p.get_height() > 0: 
        ax2.annotate(f'{int(p.get_height()/1000)}k', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=10, rotation=45)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. PAZAR (MARKET) BAZLI FİNANSAL RİSK VE SATIŞ ANALİZİ
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

market_profit = df.groupby('Market')[['Sales', 'Order Profit Per Order']].mean().reset_index()

ax3 = sns.barplot(data=market_profit, x='Market', y='Sales', alpha=0.8, color='steelblue', label='Ortalama Satış')
ax4 = ax3.twinx()
sns.lineplot(data=market_profit, x='Market', y='Order Profit Per Order', color='darkred', marker='o', linewidth=2.5, markersize=8, ax=ax4, label='Ortalama Kar')

ax3.set_title('Pazarlara Göre Ortalama Satış Hacmi ve Sipariş Başına Kar Dağılımı', fontweight='bold')
ax3.set_xlabel('Pazar (Market)', fontweight='bold')
ax3.set_ylabel('Ortalama Satış Miktarı ($)', fontweight='bold')
ax4.set_ylabel('Ortalama Kar ($)', fontweight='bold')

ax3.legend(loc='upper left')
ax4.legend(loc='upper right')

for p in ax3.patches:
    ax3.annotate(f'${p.get_height():.1f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()


import pandas as pd
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
path = r"C:\Users\tumer\OneDrive\Masaüstü\archive\DataCoSupplyChainDataset.csv"
print("Veri yükleniyor ve tam sayılarla grafik oluşturuluyor...")

df = pd.read_csv(path, encoding='ISO-8859-1')
df.columns = [col.strip() for col in df.columns]

real_col = 'Days for shipping (real)'
scheduled_col = 'Days for shipment (scheduled)'

# 0'dan 6'ya kadar olan günlerin frekanslarını çekme
real_counts = df[real_col].value_counts().sort_index()
scheduled_counts = df[scheduled_col].value_counts().sort_index()

# Tabloyu birleştirme ve eksik günlere 0 atama
comparison_df = pd.DataFrame({
    'Gerçekleşen Süre (Gün)': real_counts,
    'Planlanan Süre (Gün)': scheduled_counts
}).fillna(0).astype(int)

# Sadece 0-6 gün arasını filtreleme
comparison_df = comparison_df.loc[0:6]
comparison_df.index.name = 'Gün Sayısı'

# Konsola net sayıları yazdırma
print("\n--- PLANLANAN VE GERÇEKLEŞEN TESLİMAT SÜRELERİ (TAM SAYILAR) ---")
print(comparison_df.to_string())

# =========================================================
# 3. GRAFİK ÇİZİMİ (Melt işlemi ile veriyi uzun formata çevirme)
# =========================================================
df_melted = comparison_df.reset_index().melt(id_vars='Gün Sayısı', var_name='Süre Tipi', value_name='Sipariş Sayısı')

plt.figure(figsize=(14, 8))
ax = sns.barplot(data=df_melted, x='Gün Sayısı', y='Sipariş Sayısı', hue='Süre Tipi', palette='muted')

plt.title('Planlanan ve Gerçekleşen Teslimat Sürelerinin Karşılaştırması (Tam Sayılar)', fontweight='bold', pad=15)
plt.xlabel('Gün Sayısı', fontweight='bold')
plt.ylabel('Frekans (Sipariş Sayısı)', fontweight='bold')
plt.ylim(0, df_melted['Sipariş Sayısı'].max() * 1.1) # Etiketlerin sığması için üst limiti biraz artırıyoruz

# Çubukların üzerine tam sayıları yazdırma (Binlik ayracı olarak nokta kullanıldı)
for p in ax.patches:
    height = p.get_height()
    if height > 0: # Sadece 0'dan büyük barlara etiket ekle
        # Tam sayıyı Türkiye standartlarına uygun (noktalı) formata çevirme
        label_text = f"{int(height):,}".replace(",", ".")
        ax.annotate(label_text, 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points', rotation=45)

plt.legend(title='Süre Tipi', title_fontsize='12', fontsize='11', loc='upper right')
plt.tight_layout()
plt.show()