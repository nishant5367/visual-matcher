# backend/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "service": "visual-matcher-backend"}
