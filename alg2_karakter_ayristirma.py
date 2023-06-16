#Gerekli kütüphanelerin import edilmesi
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
#Daha önce oluşturduğum plaka_konum_don fonksiyonunu kullanacağımı belirtiyorum.
from alg1_plaka_tespiti import plaka_konum_don

#os.listdir fonksiyonunu kullanarak 'veriseti' dizinindeki dosya adlarını alıyorum.
# Ardından, ilk görüntüyü seçiyorum ve cv2.imread fonksiyonu ile okuyorum.
# Görüntüyü ayrıca 500x500 boyutuna yeniden boyutlandırıyorum.

veriler = os.listdir('veriseti')

isim = veriler[1]

img = cv2.imread('veriseti/' + isim)
img = cv2.resize(img, (500, 500))

#plaka_konum_don fonksiyonunu kullanarak plaka tespiti yapıyorum ve plakanın konumunu (x, y, w, h) değişkenlerine atıyorum.
# Ardından, plakayı ilgili konumdan kesip plaka_bgr değişkenine atıyorum.
plaka = plaka_konum_don(img)
x, y, w, h = plaka

if (w > h):
    plaka_bgr = img[y:y + h, x:x + w].copy()
else:
    plaka_bgr = img[y:y + w, x:x + h].copy()

plt.imshow(plaka_bgr)
plt.show()

#plaka_bgr görüntüsünün boyutlarını plaka_bgr.shape ile alıyorum ve orijinal boyutları ekrana yazdırıyorum.
# Ardından, boyutları iki katına çıkarıyorum.
# Son olarak, cv2.resize fonksiyonuyla plaka görüntüsünü yeni boyutlara yeniden boyutlandırıyorum.
H, W = plaka_bgr.shape[:2]
print("orjinal boyut:", W, H)
H, W = H * 2, W * 2
print("yeni boyut:", W, H)

plaka_bgr = cv2.resize(plaka_bgr, (W, H))

plt.imshow(plaka_bgr)
plt.show()

#Plaka görüntüsünü gri tonlamaya dönüştürüyorum
plaka_resim = cv2.cvtColor(plaka_bgr, cv2.COLOR_BGR2GRAY)

plt.title("gri format")
plt.imshow(plaka_resim, cmap="gray")
plt.show()

#cv2.adaptiveThreshold fonksiyonunu kullanarak adaptif eşikleme işlemi uyguluyorum.
# Eşiklenmiş görüntüyü th_img değişkenine atıyorum. Son olarak, plt.imshow fonksiyonuyla eşiklenmiş görüntüyü gösteriyorum.
th_img = cv2.adaptiveThreshold(plaka_resim, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

plt.title("eşiklenmiş")
plt.imshow(th_img, cmap="gray")
plt.show()

#np.ones fonksiyonuyla 3x3 boyutunda bir matris oluşturuyorum.
# Daha sonra, cv2.morphologyEx fonksiyonunu kullanarak açma (opening) işlemi uyguluyorum.
# Gürültü giderilmiş görüntüyü th_img değişkenine atıyorum.
# Son olarak, plt.imshow fonksiyonuyla Gürültü giderilmiş görüntüyü gösteriyorum.
kernel = np.ones((3, 3), np.uint8)
th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations=1)

plt.title("Gürültü yok edilmiş")
plt.imshow(th_img, cmap="gray")
plt.show()

#cv2.findContours fonksiyonuyla görüntü üzerindeki konturları buluyorum.
# Ardından, konturları kontur alanına göre sıralıyorum ve en büyük 15 konturu alıyorum.
# Konturları döngü ile dolaşarak her bir karakteri ayrıştırıyorum.
cnt = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:15]

#Karakte ayrıştırmayı aşağıdaki şekilde yapıyorum.
#c kont i indeks
for i, c in enumerate(cnt):
    rect = cv2.minAreaRect(c)
    (x, y), (w, h), r = rect

#kon1 karakterin belirli koşulları sağlaması gerektiğini kontrol eder
    kon1 = max([w, h]) < W / 4
#kon2 koşulu ise karakterin alanının 200 pikselden büyük olması gerektiğini kontrol eder.
    kon2 = w * h > 200

#kon1 ve kon2 şartları sağlanırsa işlemlerin gerçekleşeceği şart bloğu
    if (kon1 and kon2):
        print("karakter ->", x, y, w, h)

        box = cv2.boxPoints(rect)
        box = np.int64(box)
        # (15,20)

        minx = np.min(box[:, 0])
        miny = np.min(box[:, 1])
        maxx = np.max(box[:, 0])
        maxy = np.max(box[:, 1])

#kesim yapılırken karakterin sınırlarını biraz genişletmek için kullanılan bir odak değeridir.
        odak = 2

        minx = max(0, minx - odak)
        miny = max(0, miny - odak)
        maxx = min(W, maxx + odak)
        maxy = min(H, maxy + odak)

#belirlenen sınırlar arasındaki bölgeyi keser ve kesim değişkenine atar.
        kesim = plaka_bgr[miny:maxy, minx:maxx].copy()


#kesilen karakteri diskte bir dosya olarak kaydeder.
 # Dosya adı, orijinal plakanın adı (isim) ve konturun indeksi (i) ile oluşturulur.
        try:
            cv2.imwrite(f'karakterseti/{isim}_{i}.jpg', kesim)
        except:
            pass

#orijinal plaka görüntüsü üzerine karakterin sınırlarını gösteren bir kontur çizimi yapıyorum ve görsel olarak gösteriyorum
        yaz = plaka_bgr.copy()
        cv2.drawContours(yaz, [box], 0, (0, 255, 0), 1)

        plt.imshow(yaz)
        plt.show()
