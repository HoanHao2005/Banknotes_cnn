# main.py
# FastAPI backend for VN Money Detector (HTML version)
# ---------------------------------------------------
import os, io, time, csv, json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import numpy as np

# Optional deps
HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
except Exception:
    HAS_TORCH = False

# ---- App config ----
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "outputs_cnn")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH")  # allow absolute path via env
DEFAULT_CLASSES_PATH = os.getenv("CLASSES_PATH")
DEFAULT_IMG_SIZE = int(os.getenv("IMG_SIZE", 224))

APP = FastAPI(title="VN Money Detector API", version="1.0")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

THEME = {
    "bg": "#0f172a",
    "fg": "#e2e8f0",
    "card": "#111827",
    "accent": "#22c55e",
}

# ---- Model code (port từ Tkinter app) ----
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        self.model = None
        self.classes: List[str] = []
        self.img_size = DEFAULT_IMG_SIZE
        self.transform = None
        self.model_path = None
        self.classes_path = None

    def is_ready(self):
        return HAS_TORCH and (self.model is not None) and (len(self.classes) > 0)

    def _build_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    def _find_latest_model(self, dir_):
        p = Path(dir_)
        if not p.exists():
            return None
        cands = sorted(p.glob("best_train*.h5"), key=lambda x: x.stat().st_mtime, reverse=True)
        return str(cands[0]) if cands else None

    def load(self, model_path=None, classes_path=None, img_size=None, device_choice="auto"):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch chưa được cài đặt trên server.")

        # resolve defaults
        model_path = model_path or DEFAULT_MODEL_PATH or self._find_latest_model(DEFAULT_MODEL_DIR)
        classes_path = classes_path or DEFAULT_CLASSES_PATH or (Path(DEFAULT_MODEL_DIR)/"classes.json")

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        if not classes_path or not Path(classes_path).exists():
            raise FileNotFoundError(f"Không tìm thấy classes: {classes_path}")
        if img_size:
            self.img_size = int(img_size)

        # device
        if device_choice.lower() == "cpu":
            self.device = torch.device("cpu")
        elif device_choice.lower() == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("Không có CUDA. Hãy chọn device=CPU hoặc cài driver/CUDA.")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load classes
        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        # model
        ckpt = torch.load(model_path, map_location=self.device)
        self.img_size = int(ckpt.get("img_size", self.img_size))
        model = SmallCNN(num_classes=len(self.classes))
        model.load_state_dict(ckpt["model_state"])  # theo file của bạn
        model.to(self.device).eval()
        self.model = model
        self.model_path = model_path
        self.classes_path = str(classes_path)
        self._build_transform()

    @torch.no_grad()
    def predict_pil(self, pil_img: Image.Image):
        if not self.is_ready():
            raise RuntimeError("Model chưa sẵn sàng. Hãy nạp ở tab Settings.")
        x = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        top_idx = int(np.argmax(probs))
        return top_idx, float(probs[top_idx]), probs

MM = ModelManager()
# try autoload
try:
    MM.load()
except Exception:
    pass

class LoadReq(BaseModel):
    model_path: str | None = None
    classes_path: str | None = None
    img_size: int | None = None
    device: str = "auto"  # auto|cuda|cpu

@APP.get("/api/info")
def info():
    return {
        "pytorch": HAS_TORCH,
        "device": (None if not HAS_TORCH else MM.device.type),
        "model": MM.model_path,
        "classes": MM.classes_path,
        "img_size": MM.img_size,
        "ready": MM.is_ready(),
        "theme": THEME,
    }

@APP.post("/api/load_model")
def load_model(req: LoadReq):
    try:
        MM.load(req.model_path, req.classes_path, req.img_size, req.device)
        return {"ok": True, "device": MM.device.type, "img_size": MM.img_size}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

@APP.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
        idx, prob, probs = MM.predict_pil(pil)
        # top-3
        order = np.argsort(probs)[::-1][:3]
        top3 = [
            {"label": str(MM.classes[i]), "prob": float(probs[i])}
            for i in order
        ]
        return {"ok": True, "pred": str(MM.classes[idx]), "prob": prob, "top3": top3}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

@APP.post("/api/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    rows = []
    for f in files:
        try:
            pil = Image.open(io.BytesIO(await f.read())).convert("RGB")
            idx, prob, _ = MM.predict_pil(pil)
            rows.append({"path": f.filename, "label": str(MM.classes[idx]), "prob": float(prob)})
        except Exception as e:
            rows.append({"path": f.filename, "label": f"ERR: {e}", "prob": None})
    return {"ok": True, "rows": rows}

# helper: make CSV from batch JSON
@APP.post("/api/batch_csv")
async def batch_csv(rows_json: str = Form(...)):
    rows = json.loads(rows_json)
    def iter_csv():
        yield "path,label,prob\n"
        for r in rows:
            p = "" if r["prob"] is None else f"{r['prob']*100:.2f}%"
            yield f"{r['path']},{r['label']},{p}\n"
    return StreamingResponse(iter_csv(), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=vn_money_batch.csv"
    })

# To run: uvicorn main:APP --reload --port 8000
