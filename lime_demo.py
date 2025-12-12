import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.segmentation import mark_boundaries
from PIL import Image
import glob
import tensorflow as tf

# GPU kontrolü ve yapılandırma
print("="*60)
print("GPU/CPU Durum Kontrolü")
print("="*60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU bellek büyümesini etkinleştir (bellek hatalarını önler)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} adet GPU bulundu ve etkinleştirildi:")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    except RuntimeError as e:
        print(f"⚠️ GPU yapılandırma hatası: {e}")
else:
    print("⚠️ GPU bulunamadı, CPU kullanılacak.")
    print("   Not: GPU kullanmak için CUDA ve cuDNN yükleyin.")

print("="*60 + "\n")

# Gerekli Keras ve ResNet50 modülleri
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# LIME Görüntü Açıklayıcı
from lime import lime_image

# --- Konfigürasyon Ayarları ---
# Kaynak klasör ve çıktı klasörü
SOURCE_FOLDER = 'kaynak_gorseller_224x224'  # Görsellerin bulunduğu klasör
OUTPUT_FOLDER = 'lime_ciktilar'     # Sonuçların kaydedileceği klasör
NUM_SAMPLES = 1000  # LIME'ın açıklama için oluşturacağı pertürbasyon örnek sayısı
TOP_CLASSES = 5     # İlk 5 tahmini çözmek için
EXPLANATION_CLASS_INDEX = None # Açıklanacak spesifik sınıf index'i. None ise modelin en iyi tahminini kullanır.

# ----------------------------------------------------
# 1. Yardımcı Fonksiyonlar
# ----------------------------------------------------

def load_and_preprocess_image(img_path):
    """Görseli yükler ve ResNet50'nin beklediği formata hazırlar."""
    # 224x224 boyutuna getir
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Numpy dizisine dönüştür
    x = image.img_to_array(img)
    
    # Boyutu genişlet (batch dimension) -> (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    
    # ResNet50'nin beklediği şekilde ön işlem yap (Model girdisi)
    processed_input = preprocess_input(x)
    
    # LIME'ın beklediği görsel (0-1 aralığında, 3 boyutlu)
    lime_image_data = x[0] / 255.0
    
    return processed_input, lime_image_data

def model_predict(images):
    """LIME'ın kullanacağı tahmin fonksiyonu. (N, H, W, C) alır, olasılıkları döndürür."""
    # LIME, [0, 1] aralığındaki görselleri döndürür, ResNet'in beklediği ön işlemi yapmalıyız
    processed_images = preprocess_input(images * 255.0) 
    return model.predict(processed_images, verbose=0)

def create_heatmap_visualization(explanation, target_class, original_image):
    """LIME açıklamasından heatmap oluşturur - yeşil (pozitif), kırmızı (negatif)"""
    from skimage.color import gray2rgb
    
    # Segment ağırlıklarını al
    segments = explanation.segments
    dict_heatmap = dict(explanation.local_exp[target_class])
    
    # Heatmap array'i oluştur
    heatmap = np.zeros(segments.shape)
    
    for segment_id, weight in dict_heatmap.items():
        heatmap[segments == segment_id] = weight
    
    # Normalize et [-1, 1] aralığına
    max_abs = np.abs(heatmap).max()
    if max_abs > 0:
        heatmap = heatmap / max_abs
    
    # Renk haritası oluştur: kırmızı (negatif), yeşil (pozitif)
    heatmap_colored = np.zeros((*heatmap.shape, 3))
    
    # Pozitif değerler - Yeşil
    positive_mask = heatmap > 0
    heatmap_colored[positive_mask, 1] = heatmap[positive_mask]  # Green channel
    
    # Negatif değerler - Kırmızı
    negative_mask = heatmap < 0
    heatmap_colored[negative_mask, 0] = -heatmap[negative_mask]  # Red channel
    
    # Orijinal görsel ile karıştır
    if original_image.max() > 1:
        original_image = original_image / 255.0
    
    blended = original_image * 0.6 + heatmap_colored * 0.4
    
    return blended, heatmap_colored

def create_highlighted_regions(explanation, target_class, original_image, num_features=5):
    """Sadece en önemli bölgeleri gösterir, geri kalanını karartır"""
    temp, mask = explanation.get_image_and_mask(
        target_class, 
        positive_only=True,  # Sadece pozitif katkıları göster
        num_features=num_features, 
        hide_rest=True  # Geri kalanını gizle
    )
    
    if temp.max() > 1:
        temp = temp / 255.0
    
    return temp

def process_single_image(img_path, model, explainer):
    """Tek bir görseli işler ve görselleştirme yapar."""
    print(f"\n{'='*60}")
    print(f"İşleniyor: {os.path.basename(img_path)}")
    print(f"{'='*60}")
    
    # Orijinal görseli ayrıca yükle (görselleştirme için)
    original_img_display = Image.open(img_path).resize((224, 224))
    original_img_array = np.array(original_img_display) / 255.0
    
    # Görseli yükle ve ön işle (model için)
    input_data, lime_image_data = load_and_preprocess_image(img_path)
    
    # Model tahmini
    preds = model.predict(input_data, verbose=0)
    decoded_preds = decode_predictions(preds, top=TOP_CLASSES)[0]
    
    # En üst tahmini belirle
    top_prediction_class_index = np.argmax(preds[0])
    top_prediction_label = decoded_preds[0][1]
    top_prediction_score = decoded_preds[0][2]

    print("\n--- Model Tahminleri ---")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}. {label}: {score:.4f}")

    # Açıklanacak sınıfı belirle
    if EXPLANATION_CLASS_INDEX is None:
        target_class = top_prediction_class_index
        target_label = top_prediction_label
    else:
        target_class = EXPLANATION_CLASS_INDEX
        target_label = f"Sınıf Index {EXPLANATION_CLASS_INDEX}"
    
    # LIME Açıklaması Üret
    print(f"\n--- LIME Açıklaması ---")
    print(f"Açıklanacak Sınıf: {target_label} ({target_class})")
    print(f"Açıklama üretiliyor ({NUM_SAMPLES} örnekle)...")
    
    explanation = explainer.explain_instance(
        lime_image_data, 
        classifier_fn=model_predict, #model tahminleri alınır
        labels=[target_class],  # Açıklanacak sınıfı açıkça belirt (top_labels kullanma)
        hide_color=0, 
        num_samples=NUM_SAMPLES,
    )
    
    # Farklı görselleştirmeler oluştur
    heatmap_img, heatmap_raw = create_heatmap_visualization(explanation, target_class, original_img_array)
    highlighted_img = create_highlighted_regions(explanation, target_class, original_img_array, num_features=5)
    
    # En önemli segmentleri ve ağırlıklarını al
    local_exp = dict(explanation.local_exp[target_class])
    top_features = sorted(local_exp.keys(), key=lambda x: abs(local_exp[x]), reverse=True)[:8]
    
    # 3 panelli görselleştirme
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    
    # Sol: Orijinal görsel
    ax0.imshow(original_img_array)
    ax0.set_title(f'Orijinal Görsel\nTahmin: {top_prediction_label} ({top_prediction_score:.2%})', 
                  fontsize=14, fontweight='bold', pad=20)
    ax0.axis('off')
    
    # Orta: Heatmap
    ax1.imshow(heatmap_img)
    ax1.set_title(f'LIME Heatmap Açıklaması\nYeşil: Tahmine KATKIDA BULUNAN bölgeler\nKırmızı: Tahmini ENGELLEYEN bölgeler', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Sağ: En önemli bölgeler
    ax2.imshow(highlighted_img)
    ax2.set_title(f'En Önemli 5 Bölge\nModel bu bölgelere bakarak "{target_label}" dedi', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Kaydet (çıktı klasörüne)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_filename = os.path.join(OUTPUT_FOLDER, f"lime_explanation_{base_name}_{top_prediction_label}.png")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Görsel kaydedildi: {output_filename}")
    
    plt.close()  # Pencereyi otomatik kapat
    plt.close()

# ----------------------------------------------------
# 2. Ana İşlem
# ----------------------------------------------------

if __name__ == '__main__':
    # Kaynak klasörü kontrol et
    if not os.path.exists(SOURCE_FOLDER):
        print(f"HATA: '{SOURCE_FOLDER}' klasörü bulunamadı.")
        print(f"Lütfen '{SOURCE_FOLDER}' adında bir klasör oluşturun ve jpg dosyalarını içine koyun.")
        exit()
    
    # Çıktı klasörünü oluştur (yoksa)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 '{OUTPUT_FOLDER}' klasörü oluşturuldu.")
    
    # Kaynak klasördeki tüm jpg dosyalarını bul
    image_files = []
    image_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, '*.JPG')))
    image_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, '*.jpeg')))
    image_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, '*.JPEG')))
    
    # Tekrar eden dosyaları kaldır ve sırala
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"HATA: '{SOURCE_FOLDER}' klasöründe jpg/jpeg dosyası bulunamadı.")
        print("Lütfen görselleri kaynak klasörüne koyun.")
        exit()
    
    print(f"\n📁 Kaynak Klasör: {SOURCE_FOLDER}")
    print(f"📁 Çıktı Klasörü: {OUTPUT_FOLDER}")
    print(f"\nToplam {len(image_files)} adet görsel bulundu:")
    for img in image_files:
        print(f"  - {os.path.basename(img)}")
    
    # Modeli yükle (bir kez)
    print("\n" + "="*60)
    print("ResNet50 modelini yüklüyor...")
    print("="*60)
    model = ResNet50(weights='imagenet')
    
    # LIME açıklayıcıyı oluştur (bir kez)
    explainer = lime_image.LimeImageExplainer()
    
    # Her görseli sırayla işle
    for img_path in image_files:
        process_single_image(img_path, model, explainer)
    
    print("\n" + "="*60)
    print("✅ Tüm görseller başarıyla işlendi!")
    print("="*60)