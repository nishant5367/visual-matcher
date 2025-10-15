# Visual Product Matcher

A demo web app that finds visually similar products from a small catalog using precomputed image embeddings.

## Live demo
- Frontend: https://visual-matcher-4v96iixev-nishants-projects-02b3fb08.vercel.app/
- Backend:  https://visual-matcher-wi8u.onrender.com


## What it does
- Upload an image or paste an image URL.
- Backend computes an embedding for the query (ResNet50) and returns the top-K visually similar catalog items via cosine similarity against precomputed embeddings.
- Frontend (React + Vite) displays preview and result cards with images and similarity scores.

## Tech stack
- Frontend: React (Vite), JavaScript, axios
- Backend: FastAPI, Uvicorn, PyTorch (ResNet50), NumPy, scikit-learn
- Hosting: Vercel (frontend) and Render (backend) — free tiers

## Repo layout
visual-matcher/
├─ backend/
│ ├─ main.py
│ ├─ prepare_data.py
│ ├─ products.json
│ ├─ embeddings.npy
│ ├─ product_images/
│ └─ requirements.txt
└─ frontend/
└─ visual-matcher-frontend/
├─ src/
└─ package.json

> **Approach**  
> I implemented a Visual Product Matcher that returns visually similar catalog items for a query image. The system precomputes image embeddings using a ResNet50 feature extractor. A data-preparation script downloads the catalog images, saves metadata in `products.json`, and computes 2048-dimensional embeddings with the pretrained ResNet50 model; embeddings are stored in `embeddings.npy` for fast runtime lookup. The backend is a FastAPI service exposing `/products` (catalog) and `/match` endpoints. `/match` accepts either an uploaded file or an image URL, computes the query embedding on-the-fly, and finds nearest neighbors by cosine similarity against the precomputed catalog embeddings — returning the top-K results with similarity scores and absolute image URLs. The frontend is a lightweight React (Vite) app that allows image uploads or URL input, previews the image, calls `/match`, and displays results responsively. The architecture separates compute-heavy preprocessing from runtime matching, enabling low-latency responses. For scale, the embedding store can easily be migrated to FAISS or Milvus and the extractor swapped to CLIP for semantic-aware retrieval.

