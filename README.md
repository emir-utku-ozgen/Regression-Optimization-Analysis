# Regression & Optimization Algorithm Analysis 🚀

Bu proje, bir regresyon modeli üzerinde farklı optimizasyon algoritmalarının performansını ve ağırlık uzayındaki hareketlerini analiz etmek amacıyla geliştirilmiştir.

## 📝 Proje Özeti
Proje kapsamında, sentetik olarak üretilmiş metin verileri kullanılarak bir ikili sınıflandırma (binary classification) regresyon modeli eğitilmiştir. Temel odak, optimizasyon algoritmalarının yakınsama hızları ve hata yüzeyindeki davranışlarıdır.

### 🛠 Kullanılan Teknolojiler & Modeller
- **Veri Üretimi:** `Turkish-Gemma-9b-T1` (LLM ile sentetik QA çiftleri)
- **Vektör Temsili:** `turkish-e5-large` (768 boyutlu anlamsal embeddingler)
- **Matematiksel Model:** $y = \tanh(w \cdot x)$
- **Analiz:** T-SNE 2D Trajectory Mapping

## 📊 Optimizasyon Algoritmalarının Karşılaştırması
Aşağıdaki algoritmalar 5 farklı başlangıç ağırlığı ($w$) değeri kullanılarak, süre ve başarı kriterlerine göre analiz edilmiştir:
* **Gradient Descent (GD):** Standart gradyan inişi.
* **Stochastic Gradient Descent (SGD):** Veri örneklemi üzerinden hızlı güncellemeler.
* **Adam Optimizer:** Adaptif öğrenme hızı ile dinamik yakınsama.

### 📉 Görselleştirme (T-SNE)
Optimizasyon sürecindeki yüksek boyutlu ağırlık güncellemeleri ($w_{1:t}$), T-SNE algoritması ile 2 boyuta indirgenmiştir. Bu sayede her bir algoritmanın "ağırlık uzayındaki yörüngesi" görselleştirilmiştir.

![T-SNE Analysis](output/tsne_output.png)
*Görsel: GD, SGD ve Adam algoritmalarının yakınsama yörüngeleri.*

## 📂 Dosya Yapısı
- `Regression-Optimization-Analysis.ipynb`: Veri işleme, model eğitimi ve görselleştirme kodları.
- `/data/`: Ham metin verileri (`raw`) ve vektörleştirilmiş veriler (`processed`).
- `/output/`: Eğitim süreci grafikleri ve T-SNE çıktıları.
- `rapor.pdf`: Projenin teorik detaylarını içeren teknik rapor.

## 🎥 Sunum Videosu
Projenin detaylı anlatımı ve kod incelemesine aşağıdaki bağlantıdan ulaşabilirsiniz:
[📺 YouTube Video Linkinizi Buraya Ekleyin]

---
*Bu proje Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği bünyesinde geliştirilmiştir.*
