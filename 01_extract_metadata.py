"""Tahap 1: citra IMSFD mentah -> metadata_IMSFD.csv (22 kolom). Refaktor dari notebook EDA_IMSFD.ipynb.
Pakai: python 01_extract_metadata.py --root /path/IMSFD --out data/metadata_IMSFD.csv
Struktur dataset: <root>/{A1,A2,B1}/{training,testing}/<identity>/<file>.png
CATATAN: belum dijalankan ulang pada sesi penyusunan laporan (dataset 5,83 GB tidak tersedia); logika mengikuti notebook.
"""
import argparse, os
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", default="data/metadata_IMSFD.csv"); A = ap.parse_args()
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
assert not det.empty(), "Haar Cascade gagal dimuat"
rows = []
for subset in ["A1", "A2", "B1"]:
    for split in ["training", "testing"]:
        sp = os.path.join(A.root, subset, split)
        if not os.path.isdir(sp): continue
        for ident in tqdm([i for i in os.listdir(sp) if os.path.isdir(os.path.join(sp, i))], desc=f"{subset} {split}"):
            for root, _, files in os.walk(os.path.join(sp, ident)):
                for fn in files:
                    if not fn.lower().endswith(EXT): continue
                    p = os.path.join(root, fn); im = cv2.imread(p)
                    if im is None: continue
                    h, w = im.shape[:2]; g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    r = dict(subset=subset, split=split, identity=int(ident), filename=fn, image_path=p, width=w, height=h,
                             channels=im.shape[2] if im.ndim == 3 else 1, aspect_ratio=w / h, file_size_kb=os.path.getsize(p) / 1024,
                             brightness_mean=float(g.mean()), contrast=float(g.std()), sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var()))
                    f = det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(f):
                        x, y, fw, fh = max(f, key=lambda b: b[2] * b[3])
                        r.update(face_detected=True, face_x=x, face_y=y, face_width=fw, face_height=fh, face_area=fw * fh,
                                 face_area_ratio=fw * fh / (w * h), face_center_x=x + fw / 2, face_center_y=y + fh / 2)
                    else:
                        r.update(face_detected=False, **{k: np.nan for k in ["face_x", "face_y", "face_width", "face_height", "face_area", "face_area_ratio", "face_center_x", "face_center_y"]})
                    rows.append(r)
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True); pd.DataFrame(rows).to_csv(A.out, index=False); print("Selesai:", len(rows), "citra ->", A.out)
