# MNIST Yapay Zeka Projesi (CNN + TF.js + Flask API)

Bu proje, el yazısı rakamları tanımak için TensorFlow ile eğitilmiş bir CNN modelini içerir.  
Model farklı platformlara uyarlanabilir:
- 🧠 Eğitim ve değerlendirme (TensorFlow)
- 📊 Web arayüzü (TensorFlow.js)
- 🌐 REST API (Flask)

---

## 🔧 GEREKSİNİMLER

```bash
pip install -r requirements.txt
```

> Gerekli dosya: `requirements.txt`

---

## 🧠 EĞİTİM VE DEĞERLENDİRME

### Modeli eğitmek için:
```bash
python train.py
```

### Eğitilen modeli test etmek için:
```bash
python evaluate.py
```

---

## 📊 TENSORBOARD TAKİBİ

```bash
tensorboard --logdir=logs/fit
```
tensorboard: TensorFlow’un yerleşik eğitim takibi aracı.

--logdir=logs/fit: train.py sırasında kaydedilen log dosyalarının yolu.

🎯 Amaçları:
✅ 1. Eğitim ve Doğrulama Kaybını Gösterir
loss ve val_loss eğrilerini görsel olarak karşılaştırırsın.

Aşırı öğrenme (overfitting) olup olmadığını anlarsın.

✅ 2. Accuracy Takibi
Eğitim doğruluğu (accuracy) ve doğrulama doğruluğu (val_accuracy) grafik olarak gösterilir.

✅ 3. Epoch Başına Performans Takibi
Kaçıncı epoch'ta iyileşme durmuş, ne zaman durmalı?
EarlyStopping için analiz sağlar.

📈 Açmak için:
Tarayıcında şu adrese git: http://localhost:6006
Grafikler, istatistikler, katmanlar ve daha fazlası seni karşılar!


---

## 🌐 FLASK API

### Başlatmak için:
```bash
python app.py
```

### Test etmek için:
```bash
curl -X POST -F "file=@el_yazisi.png" http://localhost:5000/predict
```

---

## 🕸️ WEB DEMO (TF.js)

### Dönüştürmek için:
```bash
bash tfjs_convert.sh
```

### Yerel sunucuda çalıştırmak için:
```bash
python3 -m http.server
```

Sonra tarayıcında aç:
```
http://localhost:8000/demo.html
```

---

## 📁 DOSYA YAPISI

```
mnist_ann_extended/
├── app.py                 # Flask API
├── train.py               # Eğitim süreci
├── evaluate.py            # Test süreci
├── model.py               # CNN mimarisi
├── data_loader.py         # Veri yükleme ve görselleştirme
├── tuner.py               # Hiperparametre arama
├── data_augmentation.py   # Görsel artırma örnekleri
├── tfjs_convert.sh        # TF.js model dönüşüm komutu
├── demo.html              # Web arayüzü (canvas + JS)
├── saved_model/           # Eğitilmiş model (.h5 dosyası)
└── logs/                  # TensorBoard logları
```

---

## ✍️ Notlar

- `demo.html`, TF.js ile eğitilen modeli `tfjs_model/` klasöründen alır.
- `app.py` Flask API olarak çalışır ve POST üzerinden tahmin döndürür.
- `train.py` modeli eğitirken TensorBoard, EarlyStopping ve veri artırmayı içerir.

---

## 🧪 Önerilen Ortam

Python 3.8+ ve TensorFlow 2.10+

---

## 📜 Lisans

MIT Lisansı