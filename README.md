# Sistem Pendeteksi Wajah Saat Ujian Online Berbasis Artificial Intelligence

---

## Deskripsi Umum

Sistem Pendeteksi Wajah Saat Ujian Online merupakan sebuah aplikasi berbasis Artificial Intelligence (AI) dan Computer Vision yang dirancang untuk membantu proses pengawasan ujian secara daring (online). Sistem ini berfungsi untuk memonitor peserta ujian melalui kamera (webcam) secara real-time guna memastikan bahwa peserta tetap berada di depan layar dan tidak melakukan tindakan yang mengindikasikan kecurangan.

Dalam pelaksanaan ujian online, pengawasan menjadi salah satu tantangan terbesar karena tidak adanya pengawas fisik di ruang ujian. Oleh karena itu, diperlukan solusi teknologi yang mampu menggantikan fungsi pengawasan tersebut secara otomatis dan efisien. Sistem ini hadir sebagai bentuk implementasi teknologi kecerdasan buatan untuk meningkatkan integritas dan kejujuran dalam pelaksanaan ujian berbasis digital.

---

## Latar Belakang

Perkembangan teknologi informasi telah mengubah sistem pendidikan menjadi lebih fleksibel melalui pembelajaran dan ujian online. Namun, sistem ujian online memiliki kelemahan dalam hal pengawasan peserta.

Beberapa bentuk kecurangan yang sering terjadi antara lain:

- Peserta meninggalkan tempat duduk saat ujian berlangsung  
- Pergantian orang yang mengerjakan ujian  
- Adanya lebih dari satu orang dalam ruangan  
- Menggunakan bantuan dari orang lain di luar kamera  

Untuk mengatasi permasalahan tersebut, dibutuhkan sistem monitoring otomatis berbasis kamera yang mampu mendeteksi wajah peserta secara real-time dan memberikan notifikasi jika terjadi pelanggaran.

---

## Konsep dan Cara Kerja Sistem

Sistem ini bekerja dengan memanfaatkan teknologi Face Detection menggunakan algoritma Computer Vision seperti:

- Haar Cascade Classifier  
- Deep Learning (CNN-based Face Detection)  
- OpenCV Library  

### Mekanisme Kerja

1. Sistem mengakses webcam peserta  
2. Video ditangkap dalam bentuk frame secara real-time  
3. Setiap frame diproses menggunakan algoritma deteksi wajah  
4. Sistem menghitung jumlah wajah yang terdeteksi  
5. Sistem melakukan analisis kondisi:

```
1 wajah  → Kondisi normal
0 wajah  → Peserta tidak berada di depan layar
>1 wajah → Terindikasi adanya orang lain
```

Jika terjadi pelanggaran, sistem akan:

```
- Mencatat waktu kejadian
- Menyimpan log pelanggaran
- (Opsional) Mengambil screenshot sebagai bukti
```

---

## Tujuan Pengembangan

Tujuan utama dari sistem ini adalah:

```
- Meningkatkan integritas ujian online
- Mengurangi potensi kecurangan
- Memberikan sistem monitoring otomatis tanpa pengawas fisik
- Membantu institusi pendidikan dalam menerapkan sistem evaluasi berbasis digital
- Mengimplementasikan teknologi AI dalam bidang pendidikan
```

---

## Fitur Sistem

```
- Deteksi wajah real-time
- Deteksi lebih dari satu wajah
- Notifikasi pelanggaran otomatis
- Pencatatan log kejadian
- Penyimpanan bukti pelanggaran (opsional)
- Integrasi dengan sistem web / LMS
- Monitoring statistik pelanggaran
```

---

## Teknologi yang Digunakan

```
Python           : Bahasa pemrograman utama
OpenCV           : Pengolahan citra dan video
NumPy            : Pengolahan data numerik
Flask (opsional) : Integrasi berbasis web
Pre-trained Model Face Detection
```

---

## Manfaat Sistem

### Bagi Institusi Pendidikan

```
- Meningkatkan kredibilitas ujian online
- Mengurangi kebutuhan pengawas tambahan
- Meningkatkan efisiensi pengawasan
```

### Bagi Peserta

```
- Sistem pengawasan lebih objektif
- Ujian lebih transparan dan adil
```

### Bagi Pengembang

```
- Implementasi nyata teknologi AI dan Computer Vision
- Pengembangan sistem berbasis real-time processing
- Pengalaman dalam membangun sistem monitoring otomatis
```

---

## Keterbatasan Sistem

Walaupun sistem ini mampu mendeteksi wajah secara real-time, terdapat beberapa keterbatasan:

```
- Sensitif terhadap pencahayaan yang buruk
- Bergantung pada kualitas kamera
- Tidak mendeteksi perangkat lain di luar jangkauan kamera
- Tidak mendeteksi aktivitas seperti membuka tab lain
```

---

## Pengembangan Lebih Lanjut

Beberapa pengembangan yang dapat dilakukan:

```
- Face Recognition (verifikasi identitas peserta)
- Eye Tracking (mendeteksi arah pandangan)
- Head Pose Estimation
- Integrasi dengan sistem LMS (Moodle, dll.)
- Penyimpanan data ke database
- Dashboard admin monitoring real-time
- Analisis perilaku mencurigakan berbasis AI
```

---

## Kesimpulan

Sistem Pendeteksi Wajah Saat Ujian Online merupakan solusi inovatif berbasis Artificial Intelligence yang dapat membantu meningkatkan kualitas dan integritas pelaksanaan ujian daring. Dengan memanfaatkan teknologi Computer Vision, sistem ini mampu melakukan monitoring otomatis secara real-time, mendeteksi pelanggaran, dan mencatat aktivitas peserta selama ujian berlangsung.

Implementasi sistem ini menunjukkan bagaimana teknologi AI dapat diterapkan dalam dunia pendidikan untuk menciptakan sistem evaluasi yang lebih transparan, efisien, dan terpercaya.
