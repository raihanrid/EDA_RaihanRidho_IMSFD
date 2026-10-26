# Ujian EDA Sains Data - IMSFD (Raihan Ridho, 1226800003)
Tindak lanjut EDA dataset selfie wajah Indonesia (IMSFD) untuk tesis Face Matching berbasis Deep Face Recognition.

## Isi
- `01_extract_metadata.py` : citra mentah -> `data/metadata_IMSFD.csv` (butuh dataset 5,83 GB + OpenCV; belum diuji ulang di sesi penyusunan).
- `02_ujian_eda_analysis.py` : metadata -> pembersihan, uji hipotesis H1-H3, model logistik, sensitivitas, grafik, `results/results.json`.
- `data/metadata_IMSFD.csv` : metadata hasil notebook EDA (26.760 baris x 22 kolom).
- `results/` : keluaran skrip 02 (metadata_clean.csv berisi split_v2 dan in_balanced, results.json, fig1-3).

## Menjalankan
```
pip install -r requirements.txt
python 02_ujian_eda_analysis.py                      # dari metadata ke hasil akhir (seed 42)
python 01_extract_metadata.py --root /path/IMSFD     # opsional: ulangi dari citra mentah
```
## Parameter (untuk eksperimen)
`--tiny-thr 64` ambang sisi terpendek (px) | `--blank-thr 5` ambang brightness citra hitam | `--cap 300` batas citra/identitas | `--test-frac 0.3` porsi identitas uji | `--seed 42` | `--boot 400` ulangan cluster bootstrap.
Contoh: `python 02_ujian_eda_analysis.py --tiny-thr 128 --out results_128`

## Catatan
Analisis pada level metadata (bukan embedding). Duplikat dideteksi dari metadata identik; verifikasi dengan perceptual hash disarankan.
