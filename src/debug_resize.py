"""
Debugging resize-a slike - pokazuje šta se dešava sa tvojom transformacijom
Pokretanje: python debug_resize.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

# Tvoja trenutna transformacija
def trenutna_transformacija(img_h=32, img_w=128):
    return T.Compose([
        T.Resize((img_h, img_w)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])

# Alternativa: Resize sa čuvanjem proporcija (padding)
def resize_sa_paddingom(target_h=32, target_w=128):
    def transform(img):
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize čuvajući proporcije
        img = T.Resize((new_h, new_w))(img)
        
        # Padding do target veličine
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        
        img = T.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)(img)
        return img
    
    return T.Compose([
        T.Lambda(lambda x: transform(x)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])

# Alternativa: Center crop
def resize_sa_centar_cropom(target_h=32, target_w=128):
    return T.Compose([
        T.CenterCrop((target_h, target_w)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])

def prikazi_sliku(img_tensor, title, ax):
    """Prikazuje tensor slike (C, H, W)"""
    img = img_tensor.squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

def main():
    # Pronađi jednu sliku iz trening skupa
    data_path = Path("data/train")
    slike = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")) + list(data_path.glob("*.jpeg"))
    
    if not slike:
        print("❌ Nema slika u data/train/")
        print("   Prvo pokreni: python src/pripremi_podatke.py")
        return
    
    img_path = slike[0]
    print(f"📸 Test slika: {img_path}")
    
    # Učitaj originalnu sliku
    original = Image.open(img_path).convert("RGB")
    original_size = original.size
    print(f"📏 Originalna veličina: {original_size}")
    
    # Prikaži sve transformacije
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Poređenje resize metoda\nOriginal: {original_size[0]}x{original_size[1]} piksela', fontsize=14)
    
    # 1. Originalna slika (smanjena za prikaz)
    original_small = original.copy()
    original_small.thumbnail((300, 300))
    axes[0, 0].imshow(original_small)
    axes[0, 0].set_title(f'Originalna slika\n{original_size[0]}x{original_size[1]}')
    axes[0, 0].axis('off')
    
    # 2. Tvoj trenutni resize (force resize)
    transform1 = trenutna_transformacija()
    img1 = transform1(original)
    axes[0, 1].imshow(img1.squeeze().numpy(), cmap='gray')
    axes[0, 1].set_title(f'Trenutni resize\n32x128 (force)\n{img1.shape}')
    axes[0, 1].axis('off')
    
    # 3. Resize sa paddingom (preporučeno)
    transform2 = resize_sa_paddingom()
    img2 = transform2(original)
    axes[0, 2].imshow(img2.squeeze().numpy(), cmap='gray')
    axes[0, 2].set_title(f'Resize + Padding\n32x128 (čuva proporcije)\n{img2.shape}')
    axes[0, 2].axis('off')
    
    # 4. Center crop
    transform3 = resize_sa_centar_cropom()
    img3 = transform3(original)
    axes[1, 0].imshow(img3.squeeze().numpy(), cmap='gray')
    axes[1, 0].set_title(f'Center crop\n32x128 (odsecanje)\n{img3.shape}')
    axes[1, 0].axis('off')
    
    # 5. Detaljniji prikaz tvoje transformacije (uvećano)
    axes[1, 1].imshow(img1.squeeze().numpy(), cmap='gray', interpolation='nearest')
    axes[1, 1].set_title(f'Tvoj resize - detalj\n(vidi se distorzija)')
    axes[1, 1].axis('off')
    
    # 6. Detaljniji prikaz padding metode
    axes[1, 2].imshow(img2.squeeze().numpy(), cmap='gray', interpolation='nearest')
    axes[1, 2].set_title(f'Padding metod - detalj\n(čuva oblik slova)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('resize_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafikoni sačuvani: resize_comparison.png")
    plt.show()
    
    # Ispiši informacije o gubitku
    print("\n" + "="*60)
    print("  ANALIZA GUBITKA INFORMACIJA")
    print("="*60)
    
    original_array = np.array(original.convert('L'))
    resized_force = img1.squeeze().numpy()
    resized_pad = img2.squeeze().numpy()
    
    print(f"\n  Original: {original_array.shape[1]}x{original_array.shape[0]}")
    print(f"  Tvoj metod: {resized_force.shape[1]}x{resized_force.shape[0]}")
    print(f"  Padding metod: {resized_pad.shape[1]}x{resized_pad.shape[0]}")
    
    # Procentualni gubitak površine
    original_area = original_array.shape[0] * original_array.shape[1]
    force_area = resized_force.shape[0] * resized_force.shape[1]
    pad_area = resized_pad.shape[0] * resized_pad.shape[1]
    
    print(f"\n  Gubitak površine (tvoj metod): {(1 - force_area/original_area)*100:.1f}%")
    print(f"  Gubitak površine (padding): {(1 - pad_area/original_area)*100:.1f}%")
    
    # Test čitljivosti - ispiši nekoliko piksela
    print("\n  Prvih 10 piksela 1. reda (tvoj metod):")
    print(f"  {resized_force[0, :10]}")
    print("\n  Prvih 10 piksela 1. reda (padding metod):")
    print(f"  {resized_pad[0, :10]}")
    
    print("\n" + "="*60)
    print("  ZAKLJUČAK")
    print("="*60)
    print("""
    Ako vidiš:
    - Izdužena/izobličena slova → force resize je loš
    - Crne ivice ali normalna slova → padding je dobar
    - Odsečene krajeve teksta → center crop je loš
    
    Preporuka: Koristi RESIZE + PADDING metod!
    """)

if __name__ == "__main__":
    main()