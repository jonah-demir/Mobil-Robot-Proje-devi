Mobil-Robot-Proje-devi
Sensör Füzyonu ve Lokalizasyon Kullanarak LIDAR Tabanlı Otonom Navigasyon

Bu proje, 2B akıllı depo ortamında reyonlar arasında otonom navigasyon ve lokalizasyon gerçekleştiren diferansiyel sürüşlü bir mobil robot simülasyonudur.

Proje Senaryosu
Robot, deponun (12x12 m) mal kabul noktasındaki (1.5, 2.0) hareket ederek, 10 adet dikdörtgen reyon engelinin arasından güvenli bir şekilde geçmek ve hedef teslimat rafına (10.0, 10.5) ulaşmakla görevlidir.

Kullanılan Teknolojiler ve Yöntemler
- Navigasyon: Yapay Potansiyel Alanlar (APF) & Non-holonomic Diferansiyel Sürüş Modeli
- Lokalizasyon: Odometri (Dead Reckoning) & Kalman Filtresi (Sensör Füzyonu)
- Sensör İşleme: 2B LiDAR Mesafe Eşikleme ve Filtreleme

Kurulum ve Çalıştırma Talimatları

Gereksinimler
Kodun çalışması için bilgisayarınızda Python 3 ve aşağıdaki kütüphanelerin kurulu olması gerekmektedir:
```bash
pip install numpy matplotlib
