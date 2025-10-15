# backend/main.py
import os
import io
import json
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(__file__)
PRODUCTS_PATH = os.path.join(BASE, "products.json")
EMB_PATH = os.path.join(BASE, "embeddings.npy")
IMAGES_DIR = os.path.join(BASE, "product_images")

app = FastAPI(title="Visual Product Matcher")

# Allow frontend dev server (Vite) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve product images at /static/product_images/...
app.mount("/static/product_images", StaticFiles(directory=IMAGES_DIR), name="static")

# Load products and embeddings
if not os.path.exists(PRODUCTS_PATH):
    raise RuntimeError(f"Missing products.json at {PRODUCTS_PATH}. Run prepare_data.py first.")
with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
    PRODUCTS = json.load(f)

if not os.path.exists(EMB_PATH):
    raise RuntimeError(f"Missing embeddings.npy at {EMB_PATH}. Run prepare_data.py first.")
EMBS = np.load(EMB_PATH)  # (N, D)

# Model setup (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
feature_extractor.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image_bytes_to_embedding(content: bytes):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(x)  # (1, 2048, 1, 1)
        feat = feat.reshape(1, -1).cpu().numpy()
    return feat  # (1, D)

@app.get("/")
def health():
    return {"status": "ok", "service": "visual-matcher-backend"}

@app.get("/products")
def get_products(request: Request):
    base = str(request.base_url).rstrip("/")  # e.g., http://127.0.0.1:8000
    prods = []
    for p in PRODUCTS:
        p_copy = p.copy()
        # Build absolute image URL for frontend (it runs on different origin during dev)
        # p["image"] points to "/static/product_images/filename.jpg"
        if p_copy.get("image", "").startswith("/"):
            p_copy["image"] = base + p_copy["image"]
        prods.append(p_copy)
    return {"products": prods}

@app.post("/match")
async def match_image(request: Request, file: UploadFile = File(None), image_url: str = Form(None), top_k: int = Form(6)):
    if file is None and not image_url:
        return JSONResponse(status_code=400, content={"error": "Provide file or image_url form field."})
    try:
        if file:
            content = await file.read()
        else:
            # fetch the image from the URL
            r = requests.get(image_url, timeout=12)
            r.raise_for_status()
            content = r.content

        # compute embedding
        emb = image_bytes_to_embedding(content)  # (1, D)
        # compute cosine similarity with catalog embeddings
        sims = cosine_similarity(emb, EMBS)[0]  # (N,)
        idxs = sims.argsort()[::-1][:top_k]

        base = str(request.base_url).rstrip("/")
        results = []
        for i in idxs:
            prod = PRODUCTS[i].copy()
            prod["score"] = float(sims[i])
            if prod.get("image", "").startswith("/"):
                prod["image"] = base + prod["image"]
            results.append(prod)
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
