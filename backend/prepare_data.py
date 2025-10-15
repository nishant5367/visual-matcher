# backend/prepare_data.py
import os
import json
import io
from tqdm import tqdm
import requests
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "product_images")
os.makedirs(OUT_DIR, exist_ok=True)

NUM = 60  # number of products (>=50 per assignment)
# Using picsum.photos with seed ensures stable images for repeat runs
urls = [f"https://picsum.photos/seed/{i}/400/400" for i in range(1, NUM + 1)]

products = []
print("Downloading images...")
for i, url in enumerate(tqdm(urls, desc="download")):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        fname = f"prod_{i+1:03d}.jpg"
        fpath = os.path.join(OUT_DIR, fname)
        img.save(fpath, format="JPEG", quality=85)
        # image path used by frontend will be served by backend under /static/...
        products.append({
            "id": i + 1,
            "name": f"Product {i+1}",
            "category": "misc",
            "image": f"/static/product_images/{fname}"
        })
    except Exception as e:
        print(f"Failed to download {url}: {e}")

products_json_path = os.path.join(BASE, "products.json")
with open(products_json_path, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2)
print("Saved products.json with", len(products), "items.")

# Compute embeddings using ResNet50 (feature vector from avgpool)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(pretrained=True)  # works with torchvision versions supporting pretrained
model = model.to(device)
model.eval()
# remove final FC: take everything up to avgpool
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

embs = []
print("Computing embeddings...")
for p in tqdm(products, desc="embed"):
    try:
        img_path = os.path.join(BASE, p["image"].lstrip("/static/"))
        # Note: image was saved to product_images/...; build path robustly:
        # p["image"] is "/static/product_images/prod_001.jpg"
        # so remove leading "/static/" to get "product_images/prod_001.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            # fallback: try product_images/filename directly
            img = Image.open(os.path.join(OUT_DIR, os.path.basename(p["image"]))).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(x)  # shape (1, 2048, 1, 1)
            feat = feat.reshape(feat.size(0), -1).cpu().numpy()  # (1, 2048)
        embs.append(feat[0])
    except Exception as e:
        print("Failed embedding for", p["image"], ":", e)
        embs.append(np.zeros(2048, dtype=np.float32))

embs = np.stack(embs, axis=0)  # (N, 2048)
emb_path = os.path.join(BASE, "embeddings.npy")
np.save(emb_path, embs)
print("Saved embeddings to", emb_path)
print("All done.")
