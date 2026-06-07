# CRNN OCR za tablice

## Instalacija

```bash
# 1. Kreiraj virtualno okruženje
python -m venv .venv

# 2. Aktiviraj ga
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instaliraj PyTorch za CUDA (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Instaliraj ostale biblioteke
pip install -r requirements.txt