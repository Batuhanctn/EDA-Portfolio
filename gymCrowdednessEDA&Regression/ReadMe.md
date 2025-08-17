### Gym Crowdedness Prediction with Regression Models
*Spor Salonu Yoğunluğu Tahmini: Regresyon Modelleri Uygulaması*

### Genel Bakış (Overview)
Bu proje, bir spor salonunun farklı zaman dilimlerindeki yoğunluğunu (içerideki kişi sayısını) tahmin etmek amacıyla çeşitli regresyon modellerini uygulamaktadır. Proje kapsamında, veri setinin yapısını anlamak için Keşifçi Veri Analizi (EDA) yapılmış ve ardından en uygun regresyon modelini belirlemek için farklı modellerin performansları karşılaştırılmıştır.

### İçerikler (Table of Contents)
* [Veri Seti](#veri-seti)
* [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
* [Keşifçi Veri Analizi (EDA)](#keşifçi-veri-analizi-eda)
* [Veri Ön İşleme](#veri-ön-i̇şleme)
* [Model Oluşturma ve Değerlendirme](#model-oluşturma-ve-değerlendirme)
* [Kurulum](#kurulum)

---

### Veri Seti (Dataset)
Bu projede kullanılan veri seti `15-gym_crowdedness.csv`'dir. Veri seti, toplamda **62,184 gözlem** ve **11 değişken** içermektedir.

**Veri Setindeki Tüm Değişkenler:**
* `number_people`: Salondaki kişi sayısı (Hedef değişken).
* `date`: Tarih ve saat.
* `timestamp`: Zaman damgası.
* `day_of_week`: Haftanın günü (0: Pazar - 6: Cumartesi).
* `is_weekend`: Hafta sonu olup olmadığı (1: Evet, 0: Hayır).
* `is_holiday`: Resmi tatil olup olmadığı (1: Evet, 0: Hayır).
* `temperature`: Dış ortam sıcaklığı.
* `is_start_of_semester`: Sömestr başlangıcı olup olmadığı (1: Evet, 0: Hayır).
* `is_during_semester`: Sömestr dönemi olup olmadığı (1: Evet, 0: Hayır).
* `month`: Ay.
* `hour`: Saat.

---

### Kullanılan Kütüphaneler (Libraries Used)
* **pandas:** Veri manipülasyonu ve analizi için.
* **numpy:** Sayısal işlemler için.
* **seaborn & matplotlib:** Veri görselleştirme için.
* **scikit-learn:** Makine öğrenmesi modellerini oluşturmak ve değerlendirmek için.

---

### Keşifçi Veri Analizi (EDA)
EDA aşaması, veri setinin derinlemesine incelenmesini sağlamıştır. Yapılan temel analizler şunlardır:
* **Veri Tipi ve Boyutu:** Veri setinin `(62184, 11)` boyutunda olduğu ve `date` değişkeninin `object` türünde olduğu belirlenmiştir.
* **Eksik Değerler:** Veri setinde nominal olarak herhangi bir eksik değer (`null`) bulunmamaktadır.
* **İstatistiksel Özet:** Veri setinin istatistiksel özetine bakılarak her bir sütunun dağılımı ve merkezi eğilim ölçüleri incelenmiştir.

---

### Veri Ön İşleme (Data Preprocessing)
Modelin daha iyi performans göstermesi için veri ön işleme adımları aşağıdaki gibi sıralanmıştır:
1.  `date` sütunu `datetime` formatına dönüştürülmüştür.
2.  `date` sütunundan yeni bir `year` (yıl) değişkeni oluşturulmuştur.
3.  Orijinal `date` sütunu modelden çıkarılmak üzere veri setinden silinmiştir.
4.  Kategorik değişkenler (`day_of_week`, `is_weekend`, `is_holiday`, `is_start_of_semester`, `is_during_semester`, `month`, `hour`, `year`) için `One-Hot Encoding` tekniği kullanılarak yeni özellikler oluşturulmuştur.
5.  Veri, model eğitimi ve değerlendirmesi için %70'i eğitim, %30'u test verisi olacak şekilde ikiye ayrılmıştır.

---

### Model Oluşturma ve Değerlendirme (Model Building and Evaluation)
Bu projede birden fazla regresyon modeli kullanılmış ve performansları karşılaştırılmıştır.

**Kullanılan Modeller:**
* Linear Regression
* Lasso
* Ridge
* K-Neighbors Regressor
* Decision Tree Regression
* Random Forest Regression

**Model Performans Metrikleri:**
Modellerin performansını değerlendirmek için aşağıdaki metrikler kullanılmıştır:
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Squared Error (MSE)
* R-squared Score (Skor)

**Model Performans Sonuçları (Test Seti):**

| Model Adı | MAE | RMSE | MSE | R² Skoru |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | 10.78 | 14.45 | 208.82 | 0.599 |
| Lasso | 11.22 | 14.97 | 224.21 | 0.569 |
| Ridge | 10.78 | 14.45 | 208.82 | 0.599 |
| K-Neighbors Regressor | 5.05 | 7.53 | 56.75 | 0.891 |
| Decision Tree Regressor | 4.35 | 6.56 | 42.98 | 0.917 |
| Random Forest Regressor | 4.30 | 6.44 | 41.47 | 0.920 |

---

### Hyperparameter Tuning (Özelleştirilmiş Modeller)
En iyi performansı elde etmek için K-Neighbors ve Random Forest Regressor modelleri için hiperparametre ayarlaması yapılmıştır.

**Model Performans Sonuçları (Ayarlanmış Modeller - Test Seti):**

| Model Adı | Ayarlanmış Parametreler | MAE | RMSE | MSE | R² Skoru |
| :--- | :--- | :--- | :--- | :--- | :--- |
| K-Neighbors Regressor | `n_neighbors=2` | 4.64 | 6.95 | 48.27 | 0.907 |
| Random Forest Regressor | `n_estimators=500`, `max_features=7`, `max_depth=None`, `min_samples_split=2` | 4.29 | 6.42 | 41.21 | 0.921 |

### Sonuç ve Değerlendirme (Conclusion and Evaluation)
Hem temel modeller hem de ayarlanmış modeller incelendiğinde, **Random Forest Regressor** modelinin en düşük MAE ve RMSE değerleri ile en yüksek R² skoruna ulaştığı görülmüştür. Bu durum, veri setinin karmaşık yapısının, Random Forest gibi bir ensemble öğrenme modeli tarafından daha etkili bir şekilde yakalandığını göstermektedir. Projenin sonuçları, dış faktörlerin (sıcaklık, saat, gün) spor salonu yoğunluğunu tahmin etmede oldukça etkili olduğunu kanıtlamaktadır.

---

### Kurulum (Setup)
Bu projeyi yerel olarak çalıştırmak için gerekli kütüphaneleri aşağıdaki komut ile kurabilirsiniz:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
