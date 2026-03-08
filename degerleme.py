import pandas as pd
import xgboost as xgb
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
dosya_yolu = os.path.join(current_dir, "train_ready_data.csv")

if not os.path.exists(dosya_yolu):
    print(f"Hata: {dosya_yolu} bulunamadı! Önce preprocessing.py çalıştırılmalı.")
    exit()

df = pd.read_csv(dosya_yolu)

# Hedef değişken (y) ve özellikler (X)
X = df.drop('Fiyat', axis=1)
y = df['Fiyat']

#(XGBoost Hyperparameters)
model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42,
    objective='reg:squarederror'
)

print("Model eğitiliyor, lütfen bekleyin...")
model.fit(X, y)
print("Eğitim tamamlandı!")

def fiyat_tahmin_et(yil, km, marka_kodu, vites_kodu, hasar_sozlugu):
    # Eğitimdeki tüm sütunları baz alarak boş bir veri seti oluştur
    tahmin_df = pd.DataFrame(columns=X.columns)
    tahmin_df.loc[0] = 0 # Her şeyi sıfırla başla
    
    # Bilinen değerleri ata
    tahmin_df['Yil'] = yil
    tahmin_df['Km'] = km
    tahmin_df['Marka'] = marka_kodu
    tahmin_df['Vites'] = vites_kodu
    
    #(Puanlama: 1:Lokal, 2:Boya, 3:Değişen)
    for parca, puan in hasar_sozlugu.items():
        if parca in tahmin_df.columns:
            tahmin_df[parca] = puan

    fiyat = model.predict(tahmin_df)[0]
    return fiyat

# Örn: 2021 Egea (Marka Kodu: 1), Manuel (Vites: 0)
test_hasar = {
    "front-hood": 2,          # Kaput Boyalı
    "front-left-mudguard": 3  # Sol ön çamurluk değişen
}

tahmini_deger = fiyat_tahmin_et(2021, 45000, 1, 0, test_hasar)

print(f"ARAÇ ANALİZİ")
print(f"Tahmini Piyasa Değeri: {tahmini_deger:,.2f} TL")
