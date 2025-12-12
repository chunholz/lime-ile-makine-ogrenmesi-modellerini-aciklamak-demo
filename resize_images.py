"""
Görsel Yeniden Boyutlandırma Script'i
Kaynak klasördeki tüm görselleri 224x224 boyutuna getirir, rastgele sıralar ve yeniden adlandırır.
"""

import os
from PIL import Image
import glob
import random

# Konfigürasyon
SOURCE_FOLDER = 'kaynak_gorseller'
OUTPUT_FOLDER = 'kaynak_gorseller_224x224'
TARGET_SIZE = (224, 224)
# OUTPUT_PREFIX = 'ornek'  # Çıktı dosya adı ön eki

def resize_image(input_path, output_path, size=(224, 224)):
    """Görseli yeniden boyutlandır ve JPG olarak kaydet"""
    try:
        # Görseli aç
        img = Image.open(input_path)
        
        # PNG ise RGB'ye çevir (RGBA yerine)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Beyaz arka plan oluştur
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Orijinal boyut bilgisi
        original_size = img.size
        
        # Yeniden boyutlandır (LANCZOS en kaliteli yeniden örnekleme)
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        
        # JPG olarak kaydet
        img_resized.save(output_path, 'JPEG', quality=95)
        
        print(f"✅ {os.path.basename(input_path)}: {original_size} → {size}")
        return True
        
    except Exception as e:
        print(f"❌ Hata ({os.path.basename(input_path)}): {e}")
        return False

def main():
    print("="*60)
    print("Görsel Yeniden Boyutlandırma Script'i")
    print("="*60)
    
    # Kaynak klasörü kontrol et
    if not os.path.exists(SOURCE_FOLDER):
        print(f"\n❌ HATA: '{SOURCE_FOLDER}' klasörü bulunamadı!")
        print(f"Lütfen '{SOURCE_FOLDER}' klasörünü oluşturun ve görselleri içine koyun.")
        return
    
    # Çıktı klasörünü oluştur
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"\n📁 '{OUTPUT_FOLDER}' klasörü oluşturuldu.")
    
    # Tüm görsel dosyalarını bul
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))
    
    # Tekrar edenleri kaldır
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"\n❌ '{SOURCE_FOLDER}' klasöründe görsel dosyası bulunamadı!")
        return
    
    # Rastgele sırala
    random.shuffle(image_files)
    
    print(f"\n📊 Toplam {len(image_files)} adet görsel bulundu")
    print(f"🎯 Hedef boyut: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} piksel")
    print(f"🔀 Görseller rastgele sıralandı")
    print(f"\n{'='*60}")
    print("İşlem başlıyor...\n")
    
    # Her görseli işle
    success_count = 0
    fail_count = 0
    
    for idx, img_path in enumerate(image_files, start=1):
        # Yeni dosya adı: Orijinal dosya adını koru
        new_filename = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_FOLDER, new_filename)
        
        # Orijinal dosya adını göster
        original_name = os.path.basename(img_path)
        
        # Yeniden boyutlandır ve kaydet
        if resize_image(img_path, output_path, TARGET_SIZE):
            print(f"   → Yeni ad: {new_filename}")
            success_count += 1
        else:
            fail_count += 1
    
    # Özet
    print(f"\n{'='*60}")
    print("İşlem Tamamlandı!")
    print(f"{'='*60}")
    print(f"✅ Başarılı: {success_count} görsel")
    print(f"❌ Başarısız: {fail_count} görsel")
    print(f"📁 Çıktı klasörü: {OUTPUT_FOLDER}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
