#Gerekli kütüphanelerin import edilmesi
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Verilen görüntü üzerinde plaka tespiti yapmak için plaka_konum_don adında fonk.oluşturdum
def plaka_konum_don(img):
    img_bgr = img

#Görüntüyü BGR fortamttan gri formata dönüştürüyorum
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#görüntü üzerinde görüntüye median bulanıklaştırma uyguluyorum.
# Ardından, kenarları tespit etmek için Canny kenar tespiti kullanıyorum.
# Kenarları genişletmek için görüntüyü biraz genişletiyorum (dilate).
    ir_img = cv2.medianBlur(img_gray, 5)  # 5x5
    ir_img = cv2.medianBlur(ir_img, 5)  # 5x5

    medyan = np.median(ir_img)

    low = 0.67 * medyan
    high = 1.33 * medyan

    kenarlik = cv2.Canny(ir_img, low, high)

    kenarlik = cv2.dilate(kenarlik, np.ones((3, 3), np.uint8), iterations=1)

# findContours fonksiyonunu kullanarak kenarlıkları buluyorum. Ardından, konturları alanlarına göre sıralıyorum
    cnt = cv2.findContours(kenarlik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

#En büyük konturu kullanarak plaka konumunu belirliyorum
    H, W = 500, 500
    plaka = None

    for c in cnt:
        rect = cv2.minAreaRect(c)  # dikdortgen yapıda al (1)
        (x, y), (w, h), r = rect
        if (w > h and w > h * 2) or (h > w and h > w * 2):  # oran en az 2 (2)
            box = cv2.boxPoints(rect)  # [[12,13],[25,13],[20,13],[13,45]]
            box = np.int64(box)

            minx = np.min(box[:, 0])
            miny = np.min(box[:, 1])
            maxx = np.max(box[:, 0])
            maxy = np.max(box[:, 1])

            muh_plaka = img_gray[miny:maxy, minx:maxx].copy()
            muh_medyan = np.median(muh_plaka)

            kon1 = muh_medyan > 85 and muh_medyan < 200  # yogunluk kontrolu (3)
            kon2 = h < 200 and w < 1100  # sınır kontrolu (4)
            kon3 = w < 1100 and h < 200  # sınır kontrolu (4)

            print(f"muh_plaka medyan:{muh_medyan} genislik: {w} yukseklik:{h}")

            kon = False
            if (kon1 and (kon2 or kon3)):
                plaka = [int(i) for i in [minx, miny, w, h]]  # x,y,w,h
                kon = True
            else:
                pass
            if (kon):
                return plaka
    return []
