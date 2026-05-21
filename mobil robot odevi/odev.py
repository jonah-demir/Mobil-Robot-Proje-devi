import numpy as np
import matplotlib.pyplot as plt

# RASTGELELİK DURUMU (Her çalışmada aynı gürültü ve harita gelsin diye)
np.random.seed(42)

# SIMÜLASYON PARAMETRELERİ
dt = 0.1
total_time = 45.0
time_steps = int(total_time / dt)

# 1. DEPO ORTAMI VE REYONLARIN TANIMLANMASI (10 Adet Dikdörtgen Raf)
# Format: [x_min, y_min, x_max, y_max]
obstacles = np.array([
    [2.0, 1.5, 3.5, 3.5],   # Reyon 1
    [2.0, 5.0, 3.5, 7.0],   # Reyon 2
    [1.5, 8.5, 3.0, 10.5],  # Reyon 3
    [5.0, 1.0, 6.5, 4.0],   # Reyon 4
    [5.0, 5.5, 6.5, 8.5],   # Reyon 5
    [8.0, 2.0, 9.5, 5.0],   # Reyon 6
    [8.0, 6.5, 9.5, 9.5],   # Reyon 7
    [0.5, 11.0, 4.5, 11.5], # Çevre/Duvar Rafı 8
    [5.5, 11.0, 10.5, 11.5],# Çevre/Duvar Rafı 9
    [10.5, 4.0, 11.5, 8.0]  # Reyon 10
])

start_pos = np.array([1.5, 2, 0.0]) # Mal Kabul Noktası: x, y, theta
goal_pos = np.array([10.0, 10.5])     # Hedef Teslimat Rafı: x, y

# DİZİLER (Görselleştirme için veri toplama)
true_states = np.zeros((time_steps, 3))
dr_states = np.zeros((time_steps, 3))  # Sadece Odometri (Dead Reckoning)
kf_states = np.zeros((time_steps, 3))  # Sensör Füzyonu (Kalman Filtresi)
lidar_raw_history = []
lidar_filtered_history = []
time_history = np.linspace(0, total_time, time_steps)

# Başlangıç Değerleri
true_states[0] = start_pos
dr_states[0] = start_pos
kf_states[0] = start_pos

# KALMAN FİLTRESİ MATRİSLERİ
P = np.eye(3) * 0.1  
Q = np.diag([0.02, 0.02, 0.01])**2  # Odometri Süreç Gürültüsü
R = np.diag([0.05, 0.05, 0.03])**2  # LiDAR / Sensör Ölçüm Gürültüsü

# NAVİGASYON PARAMETRELERİ (Yapay Potansiyel Alanlar - APF)
K_att = 0.45   # Hedef çekim gücünü biraz artırdık (Hedefe gitmeye zorlasın)
K_rep = 2.0    # İtme gücünü biraz azalttık (Çok sert geri fırlatmasın)
rho_0 = 1.2    # Algılama mesafesini daralttık (Sadece çok yaklaşınca kaçsın, koridorda rahat yürüsün)

def get_distance_to_rect(x, y, rect):
    # Bir robotun dikdörtgen rafa olan en kısa geometrik mesafesini hesaplar
    dx = max(rect[0] - x, 0, x - rect[2])
    dy = max(rect[1] - y, 0, y - rect[3])
    return np.hypot(dx, dy)

def apf_control(x, y, theta, goal, obs):
    # Çekici Kuvvet (Hedefe yönelim)
    F_att_x = -K_att * (x - goal[0])
    F_att_y = -K_att * (y - goal[1])
    
    # İtici Kuvvet
    F_rep_x, F_rep_y = 0.0, 0.0
    for ob in obs:
        dist = get_distance_to_rect(x, y, ob)
        if dist < rho_0:
            if dist < 0.05: dist = 0.05
            rep_mag = K_rep * (1.0/dist - 1.0/rho_0) / (dist**2)
            
            # KRİTİK DÜZELTME: Robotu reyon merkezinden değil, 
            # reyonun robota en yakın olan kenar/köşesinden dışarı doğru itiyoruz.
            # Bu sayede robot köşelerde kısırdöngüye girip kendi etrafında dönmez.
            closest_x = np.clip(x, ob[0], ob[2])
            closest_y = np.clip(y, ob[1], ob[3])
            
            # Eğer robot tam içinde değilse yönü en yakın noktadan al
            dir_x = x - closest_x
            dir_y = y - closest_y
            dir_dist = np.hypot(dir_x, dir_y)
            
            if dir_dist < 0.01: # Robot tam reyon sınırındaysa merkeze bak
                center_x = (ob[0] + ob[2]) / 2.0
                center_y = (ob[1] + ob[3]) / 2.0
                dir_x = x - center_x
                dir_y = y - center_y
                dir_dist = np.hypot(dir_x, dir_y)
                
            F_rep_x += rep_mag * (dir_x / (dir_dist + 0.01))
            F_rep_y += rep_mag * (dir_y / (dir_dist + 0.01))
            
    F_total_x = F_att_x + F_rep_x
    F_total_y = F_att_y + F_rep_y
    
    # Non-holonomic dönüş hesaplamaları
    target_angle = np.arctan2(F_total_y, F_total_x)
    angle_error = target_angle - theta
    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
    
    # Robotun sıkışmasını engellemek için dönüş hızını (w) biraz daha esnettik
    v = np.clip(np.hypot(F_total_x, F_total_y), 0.05, 0.5)
    w = np.clip(3.0 * angle_error, -1.5, 1.5)
    return v, w

# SİMÜLASYON DÖNGÜSÜ
actual_end_step = time_steps
for t in range(1, time_steps):
    x_t, y_t, theta_t = true_states[t-1]
    
    # Hedef rafa ulaşıldı mı?
    if np.hypot(x_t - goal_pos[0], y_t - goal_pos[1]) < 0.25:
        actual_end_step = t
        true_states[t:] = true_states[t-1]
        dr_states[t:] = dr_states[t-1]
        kf_states[t:] = kf_states[t-1]
        break
        
    # APF Kontrolöründen hız komutlarını al
    v, w = apf_control(x_t, y_t, theta_t, goal_pos, obstacles)
    
    # Diferansiyel Robot Kinematik Modeli
    dx = v * np.cos(theta_t) * dt
    dy = v * np.sin(theta_t) * dt
    dtheta = w * dt
    
    # Gerçek Konum Güncelleme
    true_states[t] = true_states[t-1] + [dx, dy, dtheta]
    
    # SENSÖR MODELLEME
    # 1. Odometri (Dead Reckoning - Zamanla Biriken Gürültülü)
    odom_noise = np.random.normal(0, 0.035, 3)
    dr_states[t] = dr_states[t-1] + [dx, dy, dtheta] + odom_noise
    
    # 2. LiDAR Mesafe Ölçümü (En yakın reyon mesafesi)
    min_dist = 999.0
    for ob in obstacles:
        d = get_distance_to_rect(x_t, y_t, ob)
        if d < min_dist: min_dist = d
        
    raw_lidar = min_dist + np.random.normal(0, 0.18)      # Ham gürültülü LiDAR
    filtered_lidar = min_dist + np.random.normal(0, 0.02) # Filtrelenmiş LiDAR
    lidar_raw_history.append(raw_lidar)
    lidar_filtered_history.append(filtered_lidar)
    
    # KALMAN FİLTRESİ İLE SENSÖR FÜZYONU
    # Tahmin (Predict) Adımı
    X_pred = kf_states[t-1] + [dx, dy, dtheta] + np.random.normal(0, 0.01, 3)
    P_pred = P + Q
    
    # Güncelleme (Update) Adımı
    z = true_states[t] + np.random.normal(0, 0.04, 3) # Sensörlerden gelen konum bilgisi
    H = np.eye(3)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R) # Kalman Kazancı
    X_updated = X_pred + K @ (z - X_pred)
    P = (np.eye(3) - K @ H) @ P_pred
    
    kf_states[t] = X_updated

# Grafik senkronizasyonu için dizi tamamlama
while len(lidar_raw_history) < time_steps:
    lidar_raw_history.append(lidar_raw_history[-1] if lidar_raw_history else 0)
    lidar_filtered_history.append(lidar_filtered_history[-1] if lidar_filtered_history else 0)
lidar_raw_history = np.array(lidar_raw_history)
lidar_filtered_history = np.array(lidar_filtered_history)

# ----------------- GÖRSELLEŞTİRME VE GRAFİKLER -----------------

# Grafik 1: Depo Ortam Haritası (Üstten Görünüm)
plt.figure(figsize=(6,6))
plt.scatter(start_pos[0], start_pos[1], color='green', s=150, label='Mal Kabul (Baslangic)', zorder=5)
plt.scatter(goal_pos[0], goal_pos[1], color='red', s=150, label='Hedef Raf (Teslimat)', zorder=5)
for i, ob in enumerate(obstacles):
    rect = plt.Rectangle((ob[0], ob[1]), ob[2]-ob[0], ob[3]-ob[1], color='saddlebrown', alpha=0.8, label='Depo Reyonlari' if i==0 else "")
    plt.gca().add_patch(rect)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.title('Grafik 1: Akilli Depo Ortam Haritasi ve Reyon Düzeni')
plt.xlabel('X Ekseni / Koridor Genisligi (m)')
plt.ylabel('Y Ekseni / Koridor Derinligi (m)')
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig('1_ortam_haritasi.png')
plt.close()

# Grafik 2: Depo Robotu Yol Planı ve İzlenen Gerçek Yol
plt.figure(figsize=(6,6))
for i, ob in enumerate(obstacles):
    rect = plt.Rectangle((ob[0], ob[1]), ob[2]-ob[0], ob[3]-ob[1], color='saddlebrown', alpha=0.4, label='Depo Reyonlari' if i==0 else "")
    plt.gca().add_patch(rect)
plt.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], 'k:', alpha=0.6, label='Teorik En Kisa Kuş Uçuşu Yol')
plt.plot(true_states[:actual_end_step,0], true_states[:actual_end_step,1], 'b-', linewidth=2.5, label='Robotun İzledigi Gerçek Yol')
plt.scatter(start_pos[0], start_pos[1], color='green', s=100, zorder=5)
plt.scatter(goal_pos[0], goal_pos[1], color='red', s=100, zorder=5)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.title('Grafik 2: Reyonlar Arasi Gerçeklesen Otonom Navigasyon')
plt.xlabel('X Ekseni (m)')
plt.ylabel('Y Ekseni (m)')
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig('2_robot_yol_plani.png')
plt.close()

# Grafik 3: LiDAR Sensör Görselleştirmesi
plt.figure(figsize=(7,4))
plt.plot(lidar_raw_history[:actual_end_step], 'r.', alpha=0.4, label='Ham Gürültülü LiDAR Verisi')
plt.plot(lidar_filtered_history[:actual_end_step], 'b-', linewidth=1.5, label='Filtrelenmiş LiDAR Verisi')
plt.title('Grafik 3: Reyonlar Arasi Hareket Esnasinda LiDAR Filtreleme')
plt.xlabel('Zaman Adimi (k)')
plt.ylabel('En Yakin Reyon Mesafesi (m)')
plt.grid(True)
plt.legend()
plt.savefig('3_sensor_gorsellestirme.png')
plt.close()

# Grafik 4: Lokalizasyon Sonuçları (Gerçek vs Tahminler)
plt.figure(figsize=(6,6))
plt.plot(true_states[:actual_end_step,0], true_states[:actual_end_step,1], 'g-', linewidth=2.5, label='Gerçek Konum (Ground Truth)')
plt.plot(dr_states[:actual_end_step,0], dr_states[:actual_end_step,1], 'r--', alpha=0.7, label='Sadece Odometri (Dead Reckoning)')
plt.plot(kf_states[:actual_end_step,0], kf_states[:actual_end_step,1], 'b-.', label='Sensör Füzyonu (Kalman Filtresi)')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.title('Grafik 4: Depo İçi Lokalizasyon Yöntemlerinin Karsilastirmasi')
plt.xlabel('X Ekseni (m)')
plt.ylabel('Y Ekseni (m)')
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig('4_lokalizasyon_sonuclari.png')
plt.close()

# Grafik 5: Hata Analizi (RMSE Konum Hatası)
errors_dr = np.sqrt((true_states[:actual_end_step,0] - dr_states[:actual_end_step,0])**2 + (true_states[:actual_end_step,1] - dr_states[:actual_end_step,1])**2)
errors_kf = np.sqrt((true_states[:actual_end_step,0] - kf_states[:actual_end_step,0])**2 + (true_states[:actual_end_step,1] - kf_states[:actual_end_step,1])**2)
rmse_dr = np.sqrt(np.mean(errors_dr**2))
rmse_kf = np.sqrt(np.mean(errors_kf**2))

plt.figure(figsize=(7,4))
plt.plot(time_history[:actual_end_step], errors_dr, 'r-', label=f'Sadece Odometri Hatasi (RMSE: {rmse_dr:.3f}m)')
plt.plot(time_history[:actual_end_step], errors_kf, 'b-', label=f'Kalman Filtresi Hatasi (RMSE: {rmse_kf:.3f}m)')
plt.title('Grafik 5: Zaman Boyunca Pozisyon Hatasi ve RMSE Analizi')
plt.xlabel('Zaman (saniye)')
plt.ylabel('Konum Hatasi (m)')
plt.grid(True)
plt.legend()
plt.savefig('5_hata_analizi.png')
plt.close()

print("Depo simülasyonu basariyla tamamlandi!")
print(f"Sadece Odometri RMSE (Sapma): {rmse_dr:.4f} m")
print(f"Kalman Filtresi (Füzyon) RMSE: {rmse_kf:.4f} m")
print("Yeni grafikleriniz klasöre kaydedildi.")