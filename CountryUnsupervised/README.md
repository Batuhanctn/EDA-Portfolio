# 🌍 Country Development Analysis: Unsupervised Learning for Budget Allocation

Bu proje, çeşitli ülkelerin sosyoekonomik özelliklerini analiz ederek, bu ülkeleri **bütçe ihtiyaçlarına** göre gruplandırmayı amaçlayan bir denetimsiz öğrenme projesidir. Projede `country_data.csv` veri seti kullanılmış; veri ön işleme, boyut azaltma (PCA), ve çeşitli kümeleme algoritmaları (K-Means, Hierarchical Clustering, DBSCAN, HDBSCAN) ile ülkeler **"Bütçe Gerekli"**, **"Bütçe Gerekmez"** ve **"Orta Düzey"** olmak üzere üç kategoriye ayrılmıştır.

---

## 📂 İçindekiler
1. [📊 Veri Seti](#-veri-seti)
2. [⚙️ Proje İş Akışı](#️-proje-i̇ş-akışı)
   - [Veri Yükleme ve İlk İnceleme](#veri-yükleme-ve-i̇lk-i̇nceleme)
   - [Keşifsel Veri Analizi (EDA)](#keşifsel-veri-analizi-eda)
   - [Veri Ön İşleme ve Boyut Azaltma](#veri-ön-i̇şleme-ve-boyut-azaltma)
   - [Kümeleme Algoritmaları](#kümeleme-algoritmaları)
3. [🏆 Model Performans Karşılaştırması](#-model-performans-karşılaştırması)
4. [🗺️ Görselleştirme ve Sonuçlar](#️-görselleştirme-ve-sonuçlar)
5. [📝 Sonuç ve Değerlendirme](#-sonuç-ve-değerlendirme)
6. [🛠️ Kullanılan Teknolojiler](#️-kullanılan-teknolojiler)
7. [🚀 Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)

---

## 📊 Veri Seti

Bu projede kullanılan veri seti, `country_data.csv` dosyasından alınmış olup, dünyadaki çeşitli ülkelerin sosyoekonomik verilerini içermektedir. Veri seti **167 ülke** ve **10 öznitelikten** oluşmaktadır.

### Değişkenlar
| Özellik (Feature) | Açıklama | Veri Tipi |
|---|---|---|
| `country` | Ülke adı | `object` |
| `child_mort` | Çocuk ölüm oranı (1000 doğumda) | `float64` |
| `exports` | İhracat (GSYİH'nın %'si) | `float64` |
| `health` | Sağlık harcamaları (GSYİH'nın %'si) | `float64` |
| `imports` | İthalat (GSYİH'nın %'si) | `float64` |
| `income` | Kişi başına gelir (USD) | `int64` |
| `inflation` | Enflasyon oranı (%) | `float64` |
| `life_expec` | Yaşam beklentisi (yıl) | `float64` |
| `total_fer` | Toplam doğurganlık oranı | `float64` |
| `gdpp` | Kişi başına GSYİH (USD) | `int64` |

---

## ⚙️ Proje İş Akışı

### Veri Yükleme ve İlk İnceleme
- Veri seti `pandas` ile yüklendi ve `.info()`, `.describe()` gibi fonksiyonlarla temel bir inceleme yapıldı.
- **167 ülke** ve **10 değişken** içeren temiz bir veri seti olduğu tespit edildi.
- Eksik değer kontrolü sonucunda herhangi bir **NULL** değer bulunmadığı görüldü.

### Keşifsel Veri Analizi (EDA)
- **Korelasyon Analizi:** Değişkenler arasındaki ilişkiler ısı haritası ile incelendi.
- **Dağılım Analizi:** Tüm sayısal değişkenlerin histogramları çizilerek veri dağılımları analiz edildi.
- Özellikle `child_mort` ile `life_expec` arasında güçlü negatif korelasyon, `income` ile `gdpp` arasında güçlü pozitif korelasyon gözlemlendi.

### Veri Ön İşleme ve Boyut Azaltma
- **Normalizasyon:** Tüm sayısal değişkenler `MinMaxScaler` kullanılarak 0-1 arasında ölçeklendirildi.
- **PCA (Principal Component Analysis):** 9 boyutlu veri uzayı, varyansın %80'inden fazlasını koruyan **3 ana bileşene** indirgendi.
- PCA sonucunda ilk 3 bileşen toplam varyansın yaklaşık **%85'ini** açıkladığı tespit edildi.

### Kümeleme Algoritmaları
Projede dört farklı kümeleme algoritması karşılaştırıldı:

1. **K-Means Clustering**
   - Elbow yöntemi ile optimal küme sayısı **k=3** olarak belirlendi
   - Silhouette Score kullanılarak model kalitesi değerlendirildi

2. **Hierarchical Clustering (Agglomerative)**
   - Ward linkage kullanılarak hiyerarşik kümeleme uygulandı
   - 3 kümeye ayrım gerçekleştirildi

3. **DBSCAN**
   - Yoğunluk tabanlı kümeleme algoritması
   - `eps=0.1` ve `min_samples=3` parametreleri kullanıldı

4. **HDBSCAN**
   - Gelişmiş yoğunluk tabanlı kümeleme
   - `min_cluster_size=5` parametresi ile uygulandı

---

## 🏆 Model Performans Karşılaştırması

Tüm kümeleme algoritmalarının performansı Silhouette Score metriği ile değerlendirilmiştir. **Önemli:** Yoğunluk tabanlı algoritmalar (DBSCAN, HDBSCAN) için silhouette score hesabında noise noktaları (-1 etiketli) **hariç tutulmuştur**, çünkü silhouette score noise noktaları için tanımlı değildir.

| Algoritma | Silhouette Score | Küme Sayısı | Noise Oranı | Özellikler |
|---|---|---|---|---|
| **HDBSCAN** | **0.497** 🥇 | 3 | 30.5% | En yüksek score, adaptive clustering, outlier detection |
| **K-Means** | **0.439** 🥈 | 3 | 0% | Sabit küme sayısı, noise detection yok |
| **Hierarchical Clustering** | **0.439** 🥈 | 3 | 0% | K-Means ile benzer performans |
| **DBSCAN** | **0.298** 🥉 | 3 | 19.2% | Parametre optimizasyonu gerekli |

**HDBSCAN algoritması** noise noktaları hariç tutularak hesaplanan silhouette score'da en yüksek performansı göstermiştir. Bu algoritmanın adaptive clustering ve outlier detection özelliklerinin veri setine çok uygun olduğu görülmektedir.

---

## 🗺️ Görselleştirme ve Sonuçlar

### Küme Etiketleri
Ülkeler üç ana kategoriye ayrıldı:

- 🔴 **"Budget Needed" (Bütçe Gerekli):** Yüksek çocuk ölüm oranı, düşük gelir, düşük yaşam beklentisi
- 🟡 **"In Between" (Orta Düzey):** Orta seviye sosyoekonomik göstergeler
- 🟢 **"No Budget Needed" (Bütçe Gerekmez):** Düşük çocuk ölüm oranı, yüksek gelir, yüksek yaşam beklentisi

### Coğrafi Görselleştirme
- **Plotly** kullanılarak interaktif dünya haritası oluşturuldu
- Her ülke, bütçe ihtiyacına göre renklendirildi (Kırmızı-Sarı-Yeşil)
- Türkiye'nin hangi kategoride yer aldığı tespit edildi

### Örnek Bulgular
- **Türkiye:** "In Between" kategorisinde yer aldı
- **Gelişmiş ülkeler** (ABD, Almanya, Japonya): "No Budget Needed"
- **Az gelişmiş ülkeler** (Çad, Afganistan, Sierra Leone): "Budget Needed"

---

## 📝 Sonuç ve Değerlendirme

Bu projenin sonunda:
- **HDBSCAN algoritması** en iyi performansı gösterdi (Silhouette Score: 0.497 - noise hariç)
- **Adaptive clustering** ve **outlier detection** özellikleri sayesinde veri yapısına en uygun sonuçları verdi
- **Noise detection** ile belirsiz durumlu ülkeler (%30.5) başarıyla tespit edildi
- Ülkeler başarıyla üç ana kategoriye + noise kategorisine ayrıldı
- PCA ile boyut azaltma, kümeleme performansını artırdı
- **Metodolojik öğrenme:** Density-based clustering algoritmaları için silhouette score hesabında noise noktalarının hariç tutulması gerektiği anlaşıldı

Bu analiz, uluslararası yardım kuruluşları ve politika yapıcılar için ülkelerin bütçe ihtiyaçlarını objektif bir şekilde değerlendirme imkanı sunmaktadır. HDBSCAN'in noise detection özelliği, özel dikkat gerektiren "belirsiz durumdaki ülkeleri" de tanımlayabilmektedir.

---

## 🛠️ Kullanılan Teknolojiler
- **Python**
- **Pandas:** Veri manipülasyonu ve analizi
- **NumPy:** Sayısal hesaplamalar
- **Matplotlib & Seaborn:** Veri görselleştirme
- **Plotly:** İnteraktif coğrafi görselleştirme
- **Scikit-learn:** Makine öğrenmesi ve kümeleme algoritmaları
  - PCA (Principal Component Analysis)
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN
- **HDBSCAN:** Gelişmiş yoğunluk tabanlı kümeleme

---

## 🚀 Kurulum ve Çalıştırma

1. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly hdbscan
   ```

2. **Projeyi çalıştırın:**
   ```bash
   jupyter notebook CountryUnsupervised.ipynb
   ```

3. **Veri seti:** `country_data.csv` dosyasının aynı dizinde olduğundan emin olun.

---

## 📈 Veri Kaynağı
Bu projede kullanılan veri seti [Kaggle - Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data/data) adresinden alınmıştır.
