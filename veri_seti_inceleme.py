#Gerekli kütüphanelerin import edilmesi
import os
import matplotlib.pyplot as plt
import cv2
#Daha önce oluşturduğum plaka_konum_don fonksiyonunu kullanacağımı belirtiyorum.
from alg1_plaka_tespiti import plaka_konum_don

#os.listdir fonksiyonunu kullanarak 'veriseti' dizinindeki dosya adlarını alıyorum.
# Ardından, her bir görüntü için döngüye giriyorum.
# Görüntüyü okuyorum ve cv2.resize fonksiyonu ile 500x500 boyutuna yeniden boyutlandırıyorum.

veri = os.listdir('veriseti')

for image_url in veri:
      img = cv2.imread('veriseti/' + image_url)

      img = cv2.resize(img, (500, 500))

#Plaka tespiti yapmak için daha önce oluşturduğum plaka_konum_don fonksiyonunu kullanıyorum
      plaka = plaka_konum_don(img)  # x,y,w,h
      x, y, w, h = plaka
#Plakayı kesip ayrıştırıyoruz:
      if (w > h):
            plaka_bgr = img[y:y + h, x:x + w].copy()
      else:
            plaka_bgr = img[y:y + w, x:x + h].copy()

#Plaka görüntüsünü RGB formata dönüştürüyorum ve gösteriyorum
      img = cv2.cvtColor(plaka_bgr, cv2.COLOR_BGR2RGB)
      plt.imshow(img)
      plt.show()
