### Income Prediction with Random Forest Classifier
*Gelir Sınıflandırma Projesi: Random Forest Uygulaması*

### Genel Bakış (Overview)
Bu proje, ABD'deki bireylerin özelliklerini içeren bir veri setini kullanarak yıllık gelirlerinin 50.000 doların altında mı yoksa üstünde mi olduğunu tahmin etmek için bir sınıflandırma modeli geliştirmeyi amaçlamaktadır. Proje kapsamında, veri setinin yapısını anlamak için detaylı bir Keşifçi Veri Analizi (EDA) yapılmış ve ardından en iyi performans gösteren modeli bulmak için Random Forest Sınıflandırıcısı kullanılarak makine öğrenmesi modeli oluşturulmuştur.

### İçerikler (Table of Contents)
* [Veri Seti](#veri-seti)
* [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
* [Keşifçi Veri Analizi (EDA)](#keşifçi-veri-analizi-eda)
* [Veri Ön İşleme ve Özellik Mühendisliği](#veri-ön-i̇şleme-ve-özellik-mühendisliği)
* [Model Oluşturma ve Değerlendirme](#model-oluşturma-ve-değerlendirme)
* [Kurulum](#kurulum)

---

### Veri Seti (Dataset)
Bu projede kullanılan veri seti `income_evaluation.csv`'dir. Veri seti, toplamda **32,561 gözlem** ve **15 değişken** içermektedir. Değişkenler, yaş, meslek, eğitim seviyesi, medeni durum ve çalışma saatleri gibi demografik bilgileri içermektedir.

**Veri Setindeki Tüm Değişkenler:**
* `age`: Yaş (Sayısal)
* `workclass`: Çalışma Sınıfı (Kategorik)
* `finalweight` (orijinal adı: `fnlwgt`): Son ağırlık, tahmini demografik bilgiler için kullanılan bir ağırlıklandırma. (Sayısal)
* `education`: Eğitim Seviyesi (Kategorik)
* `education_num` (orijinal adı: `education-num`): Eğitim seviyesinin sayısal karşılığı. (Sayısal)
* `marital_status`: Medeni Durum (Kategorik)
* `occupation`: Meslek (Kategorik)
* `relationship`: Aile İlişkisi (Kategorik)
* `race`: Irk (Kategorik)
* `sex`: Cinsiyet (Kategorik)
* `capital_gain`: Sermaye Kazancı (Sayısal)
* `capital_loss`: Sermaye Kaybı (Sayısal)
* `hours_per_week`: Haftalık Çalışma Saati (Sayısal)
* `native_country`: Anavatan (Kategorik)
* `income`: Hedef değişken, gelirin `<=50K` veya `>50K` olup olmadığı. (Kategorik)

---

### Kullanılan Kütüphaneler (Libraries Used)
* **pandas:** Veri manipülasyonu ve analizi için.
* **numpy:** Sayısal işlemler için.
* **seaborn & matplotlib:** Veri görselleştirme için.
* **scikit-learn:** Makine öğrenmesi modeli oluşturma, veri ön işleme ve model değerlendirme için.

---

### Keşifçi Veri Analizi (EDA)
EDA aşaması, veri setinin derinlemesine incelenmesini ve veri yapısının anlaşılmasını sağlamıştır. Yapılan temel analizler şunlardır:

* **Veri Tipleri ve Eksik Değerler:** Veri setinin 6 sayısal ve 9 kategorik sütunu vardır. Nominal olarak eksik değer bulunmamaktadır. Ancak `workclass`, `occupation` ve `native_country` gibi bazı kategorik değişkenlerde ' ?' şeklinde eksik veri işaretleri tespit edilmiştir. Bu değerler daha sonra mod değeri ile doldurulmuştur.

* **Tek Değişkenli Analiz:** Her bir değişkenin dağılımı incelenmiş, özellikle kategorik değişkenlerin unique değerleri ve sayıları kontrol edilmiştir.

---

### Veri Ön İşleme ve Özellik Mühendisliği (Data Preprocessing and Feature Engineering)
Modelin daha iyi performans göstermesi için veri ön işleme ve özellik mühendisliği adımları aşağıdaki gibi sıralanmıştır:

1.  **Sütun Adlarını Düzenleme:** Orijinal veri setindeki bazı sütun adları, okunabilirliği artırmak ve boşlukları gidermek amacıyla yeniden adlandırılmıştır (Örn: `fnlwgt` -> `finalweight`, `education-num` -> `education_num`, `capital-gain` -> `capital_gain`).
2.  **Eksik Değerlerin Doldurulması:** Kategorik sütunlardaki (' ?') eksik değerler, her bir sütunun en sık görülen değeriyle (`mode`) doldurulmuştur. Bu, veri kaybını önlerken modelin performansını artırmaya yardımcı olmuştur.
3.  **Değişkenlerin Ayrılması:** Veri seti, sayısal ve kategorik sütunlar olarak ikiye ayrılmıştır. Bu ayrım, her değişken türüne uygun ön işleme adımlarının uygulanabilmesi için yapılmıştır.
4.  **Hedef Değişkenin Kodlanması:** Hedef değişken olan `income`, `LabelEncoder` kullanılarak kategorik değerlerden sayısal değerlere dönüştürülmüştür. Bu sayede `<=50K` 0'a ve `>50K` 1'e çevrilerek modelin çalışabileceği formata getirilmiştir.
5.  **Verinin Ölçeklendirilmesi:** Makine öğrenmesi modellerinde aykırı değerlerin (outliers) etkisini azaltmak ve farklı ölçeklerdeki değişkenlerin eşit ağırlığa sahip olması için sayısal değişkenler **RobustScaler** kullanılarak ölçeklendirilmiştir.
6.  **Eğitim ve Test Kümelerine Ayırma:** Modelin performansını objektif bir şekilde değerlendirebilmek için veri seti, %70'i eğitim ve %30'u test verisi olacak şekilde ikiye ayrılmıştır.

---

### Model Oluşturma ve Değerlendirme (Model Building and Evaluation)
Bu projede bir **Random Forest Sınıflandırma Modeli** kullanılmıştır. Modelin performansını en üst düzeye çıkarmak için `RandomizedSearchCV` ile hiperparametre ayarı yapılmıştır.

**En iyi modelin hiperparametreleri:**
* `n_estimators`: 200
* `min_samples_split`: 2
* `max_features`: 'sqrt'
* `max_depth`: 15

**Model Performans Metrikleri:**
Eğitilen model, test veri seti üzerinde aşağıdaki sonuçları vermiştir:
* **Accuracy Score (Doğruluk Puanı):** `0.8529`
* **Classification Report (Sınıflandırma Raporu):**
    * `0` sınıfı (`<=50K`) için `precision` değeri `0.88`, `recall` değeri `0.94` ve `f1-score` değeri `0.91` olarak bulunmuştur.
    * `1` sınıfı (`>50K`) için `precision` değeri `0.78`, `recall` değeri `0.60` ve `f1-score` değeri `0.68` olarak bulunmuştur.
* **Confusion Matrix (Karmaşıklık Matrisi):**
    ```
    [[6996  411]
     [ 945 1417]]
    ```
    Bu matris, modelin 6996 gözlemi doğru bir şekilde `<=50K` olarak ve 1417 gözlemi doğru bir şekilde `>50K` olarak tahmin ettiğini göstermektedir.

### Kurulum (Setup)
Bu projeyi yerel olarak çalıştırmak için gerekli kütüphaneleri aşağıdaki komut ile kurabilirsiniz:

```bash
pip install -r requirements.txt