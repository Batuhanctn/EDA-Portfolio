# 🚗 İkinci El Araç Fiyat Tahmini: AdaBoost Regresyon ile Model Optimizasyonu

Bu proje, ikinci el araç piyasasındaki fiyatları tahmin etmek amacıyla geliştirilmiş bir makine öğrenmesi modelini içermektedir. Projede `cardekho.csv` veri seti kullanılmış; veri temizliği, keşifsel veri analizi, özellik mühendisliği adımları uygulanmış ve **AdaBoost Regresyon** modeli ile tahminleme yapılmıştır. Modelin performansı, **RandomizedSearchCV** ile hiperparametre optimizasyonu yapılarak artırılmıştır.

---

## 📂 İçindekiler
1. [📊 Veri Seti](#-veri-seti)
2. [⚙️ Proje İş Akışı](#️-proje-i̇ş-akışı)
   - [Veri Yükleme ve İlk İnceleme](#veri-yükleme-ve-i̇lk-i̇nceleme)
   - [Keşifsel Veri Analizi (EDA)](#keşifsel-veri-analizi-eda)
   - [Veri Ön İşleme ve Özellik Mühendisliği](#veri-ön-i̇şleme-ve-özellik-mühendisliği)
   - [Modelleme Stratejisi](#modelleme-stratejisi)
   - [Hiperparametre Optimizasyonu](#hiperparametre-optimizasyonu)
3. [🏆 Model Performans Karşılaştırması](#-model-performans-karşılaştırması)
4. [📝 Sonuç ve Değerlendirme](#-sonuç-ve-değerlendirme)
5. [🛠️ Kullanılan Teknolojiler](#️-kullanılan-teknolojiler)
6. [🚀 Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)

---

## 📊 Veri Seti
Bu projede kullanılan veri seti, `cardekho.csv` dosyasından alınmış olup, Hindistan'daki ikinci el araçların çeşitli özelliklerini ve satış fiyatlarını içermektedir. Veri seti, temizleme işlemleri sonrasında **15,235 gözlem (satır)** ve **13 öznitelikten (sütun)** oluşmaktadır. Hedef değişkenimiz `selling_price` (satış fiyatı)'dır.

### Değişkenler
| Özellik (Feature) | Açıklama | Veri Tipi (Data Type) |
|---|---|---|
| `car_name` | Aracın tam adı (Marka + Model) | `object` |
| `brand` | Aracın markası | `object` |
| `model` | Aracın modeli | `object` |
| `vehicle_age` | Aracın yaşı (yıl) | `int64` |
| `km_driven` | Aracın yaptığı toplam kilometre | `int64` |
| `seller_type` | Satıcı türü (Bireysel, Bayi vb.) | `object` |
| `fuel_type` | Yakıt türü (Benzin, Dizel vb.) | `object` |
| `transmission_type`| Vites türü (Manuel, Otomatik) | `object` |
| `mileage` | Aracın yakıt verimliliği (km/L) | `float64` |
| `engine` | Motor hacmi (cc) | `int64` |
| `max_power` | Maksimum motor gücü (bhp) | `float64` |
| `seats` | Koltuk sayısı | `int64` |
| **`selling_price`**| **(Hedef Değişken)** Aracın satış fiyatı | `int64` |

---

## ⚙️ Proje İş Akışı

### Veri Yükleme ve İlk İnceleme
- Veri seti `pandas` ile yüklendi ve `.info()`, `.describe()` gibi fonksiyonlarla temel bir inceleme yapıldı.
- `isnull().sum()` ile yapılan kontrolde, veri setinde eksik değer **bulunmadığı** tespit edildi.
- `.drop_duplicates()` ile tekrar eden satırlar temizlendi.

### Keşifsel Veri Analizi (EDA)
- **Korelasyon Analizi:** `selling_price` ile en yüksek pozitif korelasyona sahip özelliklerin `max_power` ve `engine` olduğu ısı haritası ile gözlemlendi. `vehicle_age` ile ise negatif bir korelasyon mevcuttur.
- **Aykırı Değerler:** Özellikle `km_driven` ve `selling_price` sütunlarında aşırı yüksek aykırı değerler tespit edildi. Bu durum, ön işleme aşamasında dikkate alındı.
- **Kategorik Değişkenler:**
    - Yakıt türü (`fuel_type`) ve vites tipi (`transmission_type`) gibi kategorik özelliklerin satış fiyatı üzerindeki etkileri kutu grafikleriyle incelendi. Dizel ve otomatik vitesli araçların genellikle daha yüksek fiyatlı olduğu görüldü.
    - Koltuk sayısı (`seats`) 0 olan hatalı 2 adet kayıt tespit edildi.

### Veri Ön İşleme ve Özellik Mühendisliği
Modelin performansını artırmak ve veriyi algoritmaların işleyebileceği formata getirmek için aşağıdaki adımlar uygulanmıştır:

| İşlem Adımı | Uygulanan Teknik/Metot | Gerekçe |
|---|---|---|
| **Gereksiz Sütunların Atılması** | `Unnamed: 0` sütunu atıldı. | Modele bir katkısı olmayan ve sadece indeks bilgisi içeren bir sütundu. |
| **Aykırı Değer Yönetimi** | `seats` değeri 0 olan satırlar ve `selling_price` değeri 10 milyondan, `km_driven` değeri 1 milyondan büyük olan satırlar veri setinden çıkarıldı. | Bu değerler, veri girişi hatası veya modelin performansını olumsuz etkileyecek aşırı aykırı değerler olarak değerlendirildi. |
| **Kategorik Veri Kodlama**| Yüksek kardinaliteye sahip `car_name`, `brand`, `model` sütunları için **Frequency Encoding**; düşük kardinaliteye sahip `seller_type`, `fuel_type`, `transmission_type` sütunları için ise **One-Hot Encoding** kullanıldı. | Modelin kategorik verileri işleyebilmesi ve kardinalite lanetinden kaçınmak için iki farklı kodlama tekniği bir arada kullanıldı. |

### Modelleme Stratejisi
Bu projede, model performansını aşamalı olarak geliştirmek için üç farklı model kurulmuştur:
1.  **AdaBoost Regressor (Varsayılan Model):** Ensemble öğrenme tekniklerinden biri olan AdaBoost'un temel performansını ölçmek için varsayılan parametrelerle kurulmuştur.
2.  **Optimize Edilmiş AdaBoost Regressor:** Modelin en iyi performansını bulmak için hiperparametre optimizasyonu uygulanmış versiyonudur. Temel öğrenici olarak varsayılan `DecisionTreeRegressor(max_depth=3)` kullanır.
3.  **Optimize Edilmiş AdaBoost (Karar Ağacı ile):** Temel öğrenicisi (`base_estimator`) olarak `DecisionTreeRegressor` belirtilerek ve ağacın `max_depth` parametresi de optimize edilerek daha esnek ve potansiyel olarak daha güçlü bir model hedeflenmiştir.

### Hiperparametre Optimizasyonu
AdaBoost modelinin performansını maksimize etmek için `RandomizedSearchCV` kullanılmıştır. Bu yöntem, geniş bir parametre uzayında en iyi kombinasyonları verimli bir şekilde bulur.

- **Taranan Parametreler:** `n_estimators`, `learning_rate`, `loss` ve `estimator__max_depth` (3. model için).
- **En İyi Parametreler:** `RandomizedSearchCV` sonucunda en iyi performansı veren parametre seti şöyledir:
    - `n_estimators`: 120
    - `learning_rate`: 0.1
    - `loss`: 'exponential'
    - `estimator__max_depth`: 5

---

## 🏆 Model Performans Karşılaştırması
Tüm modeller, aynı test verisi üzerinde değerlendirilmiş ve sonuçlar aşağıdaki tabloda özetlenmiştir.

| Model | MAE (Ort. Mutlak Hata) | MSE (Ort. Kare Hata) | RMSE (Kök Ort. Kare Hata) | R² Skoru |
|---|---|---|---|---|
| **AdaBoost Regressor** (Varsayılan) | 328,019 | 178,247,917,140 | 422,194 | 0.72 |
| **Tuned AdaBoost Regressor** (Optimize) | 221,199 | 121,531,301,364 | 348,613 | 0.81 |
| **Tuned AdaBoost (Karar Ağacı ile)** | **145,305** | **83,049,628,367** | **288,183** | **0.87** |

Tablodan da görüldüğü üzere, hem temel öğrenicinin hiperparametrelerinin (`max_depth`) hem de AdaBoost'un kendi parametrelerinin optimize edildiği üçüncü model, **test verisi üzerinde %87'lik bir R² skoru** ile en başarılı model olmuştur.

---

## 📝 Sonuç ve Değerlendirme
Bu projenin sonunda, hiperparametre optimizasyonu yapılmış ve temel öğrenici olarak derinliği ayarlanmış bir Karar Ağacı kullanan **AdaBoost Regresyon modeli**, ikinci el araç fiyatlarını tahminlemede en başarılı sonucu vermiştir.

- Modelin tahminleri ortalama olarak **288,183 birim** sapma göstermektedir (RMSE).
- Bu sonuç, özellik mühendisliği ve hiperparametre optimizasyonunun, standart bir modelin performansını ne kadar ciddi ölçüde artırabildiğini göstermektedir.

---

## 🛠️ Kullanılan Teknolojiler
- **Python**
- **Pandas:** Veri manipülasyonu ve analizi
- **NumPy:** Sayısal hesaplamalar
- **Matplotlib & Seaborn:** Veri görselleştirme
- **Scikit-learn:** Makine öğrenmesi modellemesi ve değerlendirmesi

---
