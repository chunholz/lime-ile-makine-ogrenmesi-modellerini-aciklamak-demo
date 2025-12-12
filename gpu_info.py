"""
GPU/Grafik Kartı Bilgileri Script'i
Sistemdeki grafik kartlarını ve özelliklerini listeler.
"""

import subprocess
import platform

def get_gpu_info_nvidia():
    """NVIDIA GPU bilgilerini nvidia-smi ile al"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpu_info = {
                    'index': parts[0],
                    'name': parts[1],
                    'driver': parts[2],
                    'memory_total': f"{parts[3]} MB",
                    'memory_free': f"{parts[4]} MB",
                    'memory_used': f"{parts[5]} MB",
                    'temperature': f"{parts[6]}°C",
                    'utilization': f"{parts[7]}%"
                }
                gpus.append(gpu_info)
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_gpu_info_windows():
    """Windows için GPU bilgilerini WMIC ile al"""
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 
                               'name,AdapterRAM,DriverVersion,Status', '/format:csv'],
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')[1:]  # İlk satırı (başlık) atla
        gpus = []
        
        for line in lines:
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        ram_gb = int(parts[1]) / (1024**3) if parts[1].isdigit() else 0
                    except:
                        ram_gb = 0
                    
                    gpu_info = {
                        'name': parts[3],
                        'memory': f"{ram_gb:.2f} GB" if ram_gb > 0 else "Bilinmiyor",
                        'driver': parts[2],
                        'status': parts[4]
                    }
                    gpus.append(gpu_info)
        return gpus
    except:
        return None

def get_tensorflow_gpu():
    """TensorFlow'un görebildiği GPU'ları listele"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return gpus
    except:
        return None

def main():
    print("="*70)
    print("GPU/GRAFIK KARTI BİLGİLERİ")
    print("="*70)
    print(f"İşletim Sistemi: {platform.system()} {platform.release()}")
    print(f"Mimari: {platform.machine()}")
    print("="*70)
    
    # NVIDIA GPU Bilgileri (nvidia-smi)
    print("\n📊 NVIDIA GPU Bilgileri (nvidia-smi):")
    print("-"*70)
    nvidia_gpus = get_gpu_info_nvidia()
    
    if nvidia_gpus:
        for gpu in nvidia_gpus:
            print(f"\n🎮 GPU #{gpu['index']}: {gpu['name']}")
            print(f"   Sürücü Versiyonu: {gpu['driver']}")
            print(f"   Toplam Bellek: {gpu['memory_total']}")
            print(f"   Kullanılan Bellek: {gpu['memory_used']}")
            print(f"   Boş Bellek: {gpu['memory_free']}")
            print(f"   Sıcaklık: {gpu['temperature']}")
            print(f"   Kullanım: {gpu['utilization']}")
    else:
        print("❌ NVIDIA GPU bulunamadı veya nvidia-smi yüklü değil.")
    
    # Windows GPU Bilgileri
    if platform.system() == 'Windows':
        print("\n📊 Windows GPU Bilgileri (WMIC):")
        print("-"*70)
        win_gpus = get_gpu_info_windows()
        
        if win_gpus:
            for idx, gpu in enumerate(win_gpus):
                print(f"\n🎮 GPU #{idx}: {gpu['name']}")
                print(f"   Bellek: {gpu['memory']}")
                print(f"   Sürücü: {gpu['driver']}")
                print(f"   Durum: {gpu['status']}")
        else:
            print("❌ Windows GPU bilgileri alınamadı.")
    
    # TensorFlow GPU Desteği
    print("\n📊 TensorFlow GPU Desteği:")
    print("-"*70)
    tf_gpus = get_tensorflow_gpu()
    
    if tf_gpus is not None:
        if tf_gpus:
            print(f"✅ TensorFlow {len(tf_gpus)} adet GPU algıladı:")
            for gpu in tf_gpus:
                print(f"   - {gpu.name}")
        else:
            print("⚠️ TensorFlow GPU algılayamadı.")
            print("   CUDA ve cuDNN kurulumu gerekebilir.")
    else:
        print("❌ TensorFlow yüklü değil.")
    
    print("\n" + "="*70)
    print("Bilgi toplama tamamlandı!")
    print("="*70)

if __name__ == '__main__':
    main()
