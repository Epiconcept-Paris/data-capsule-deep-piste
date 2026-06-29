"""
Prétraitement GMIC minimal — pour le cours breast-cancer-course
----------------------------------------------------------------
Condensation fidèle de `utils/preprocess.py` du dépôt de recherche, pensée
pour tourner sur une POIGNÉE d'images (chapitre 2.5). Mêmes règles que la
version complète, mais orchestrées en une fonction `run()` et sans les
options CLI / marqueurs de reprise.

Les 6 étapes :
  1. DICOM -> PNG 16-bit (inversion MONOCHROME1, normalisation [0, 65535])
  2. Construction du PKL au format GMIC (chemins des vues + labels cancer)
  3. Recadrage via le script ORIGINAL de GMIC (`src/cropping/crop_mammogram.py`)
  4. Resize 2944x1920 + normalisation uint8
  5. Flip horizontal des vues droites (R-CC, R-MLO)
  6. data.pkl final (copie de cropped_exam_list.pkl)

GMIC est attendu dans ~/GMIC (cloné par le Dockerfile du cours).
"""

import os
import sys
import csv
import glob
import pickle
import shutil
import zipfile
import subprocess
from multiprocessing import Pool

import cv2
import numpy as np
import pydicom

GMIC_DIR = os.environ.get("GMIC_DIR", os.path.expanduser("~/GMIC"))
GMIC_H, GMIC_W = 2944, 1920
VIEWS = ("L-CC", "L-MLO", "R-CC", "R-MLO")
_VIEWS_RIGHT = {"R-CC", "R-MLO"}

if GMIC_DIR not in sys.path:
    sys.path.insert(0, GMIC_DIR)


# ── Étape 1 : DICOM -> PNG 16-bit ────────────────────────────────────────────

def _convert_dcm_one(task):
    """Convertit 1 DICOM en PNG 16-bit. MONOCHROME1 inversé, normalisation
    [0, 65535], skip si le PNG existe déjà."""
    pid, iid, raw_dir, png_dir = task
    out_path = os.path.join(png_dir, pid, f"{iid}.png")
    if os.path.exists(out_path):
        return "skip"
    zip_path = os.path.join(raw_dir, pid, f"{iid}.dcm.zip")
    dcm_path = os.path.join(raw_dir, pid, f"{iid}.dcm")
    try:
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                dcm_name = next(n for n in z.namelist() if n.endswith(".dcm"))
                with z.open(dcm_name) as f_dcm:
                    ds = pydicom.dcmread(f_dcm)
        elif os.path.exists(dcm_path):
            ds = pydicom.dcmread(dcm_path)
        else:
            return "missing"

        arr = ds.pixel_array.astype(np.float32)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr            # inverse les niveaux : blanc <-> noir
        arr_max = arr.max()
        arr = (arr / arr_max * 65535).astype(np.uint16) if arr_max > 0 else arr.astype(np.uint16)

        os.makedirs(os.path.join(png_dir, pid), exist_ok=True)
        cv2.imwrite(out_path, arr)
        return "ok"
    except Exception as e:
        return f"err:{pid}/{iid}:{e}"


def convert_dcm_to_png(raw_dir, png_dir, csv_path, num_processes=2):
    print("ÉTAPE 1 — Conversion DICOM -> PNG 16-bit")
    image_ids = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            image_ids.add((row["patient_id"], row["image_id"]))
    tasks = [(pid, iid, raw_dir, png_dir) for pid, iid in sorted(image_ids)]

    ok = skip = fail = 0
    with Pool(max(1, num_processes)) as pool:
        for status in pool.imap_unordered(_convert_dcm_one, tasks, chunksize=4):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                if status not in ("missing",):
                    print("  ", status[4:])
    print(f"  convertis={ok} déjà-OK={skip} échecs={fail}")
    return ok + skip


# ── Étape 2 : PKL GMIC ───────────────────────────────────────────────────────

def build_exam_pkl(csv_path, png_dir, pkl_path):
    print("ÉTAPE 2 — Construction du PKL GMIC")
    patients = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            if pid not in patients:
                patients[pid] = {
                    "horizontal_flip": "NO",
                    "L-CC": [], "L-MLO": [], "R-CC": [], "R-MLO": [],
                    "cancer_label": {
                        "benign": 0, "left_benign": 0, "right_benign": 0,
                        "malignant": 0, "left_malignant": 0, "right_malignant": 0,
                        "unknown": 0,
                    },
                }
            iid = row["image_id"]
            view_key = f"{row['laterality']}-{row['view']}"
            cancer = int(row["cancer"])
            biopsy = int(row["biopsy"])
            if not os.path.exists(os.path.join(png_dir, pid, f"{iid}.png")):
                continue
            if view_key in patients[pid]:
                patients[pid][view_key].append(f"{pid}/{iid}")

            # malignant = cancer biopsié ; benign = biopsié sans cancer.
            # Un sein jamais biopsié reste NI benign NI malignant (pas "unknown").
            side = "left" if row["laterality"] == "L" else "right"
            if cancer == 1:
                patients[pid]["cancer_label"]["malignant"] = 1
                patients[pid]["cancer_label"][f"{side}_malignant"] = 1
            if biopsy == 1 and cancer == 0:
                patients[pid]["cancer_label"]["benign"] = 1
                patients[pid]["cancer_label"][f"{side}_benign"] = 1

    exam_list = [e for e in patients.values() if any(e[v] for v in VIEWS)]
    with open(pkl_path, "wb") as f:
        pickle.dump(exam_list, f)
    mal = sum(1 for e in exam_list if e["cancer_label"]["malignant"])
    print(f"  examens={len(exam_list)} (malignant={mal}) -> {os.path.basename(pkl_path)}")
    return exam_list


# ── Étape 3 : Crop (script GMIC original) ────────────────────────────────────

def run_crop(png_dir, cropped_dir, pkl_raw, pkl_cropped, num_processes=2):
    print("ÉTAPE 3 — Recadrage (crop_mammogram.py de GMIC)")
    if os.path.exists(cropped_dir):
        shutil.rmtree(cropped_dir)
    cmd = (
        f"cd {GMIC_DIR} && PYTHONPATH={GMIC_DIR}:$PYTHONPATH {sys.executable} "
        f"src/cropping/crop_mammogram.py "
        f"--input-data-folder {png_dir} "
        f"--output-data-folder {cropped_dir} "
        f"--exam-list-path {pkl_raw} "
        f"--cropped-exam-list-path {pkl_cropped} "
        f"--num-processes {num_processes}"
    )
    res = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
    n = len(glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True))
    if res.returncode != 0:
        raise RuntimeError("crop_mammogram a échoué :\n" + res.stderr.decode()[-800:])
    print(f"  croppées={n}")


# ── Étape 4 : Resize 2944x1920 + normalisation uint8 ─────────────────────────

def _normalize_uint8(img):
    f = img.astype(np.float32)
    lo, hi = f.min(), f.max()
    return ((f - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(f, np.uint8)


def _resize_one(args):
    path, cropped_dir = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return ("none", None, None)
    h, w = img.shape[:2]
    if h == GMIC_H and w == GMIC_W:
        if img.max() > 255 or img.dtype != np.uint8:
            cv2.imwrite(path, _normalize_uint8(img))
            return ("skipnorm", None, None)
        return ("skip", None, None)
    sfp = os.path.relpath(path, cropped_dir)[:-4]
    interp = cv2.INTER_AREA if (h > GMIC_H or w > GMIC_W) else cv2.INTER_LINEAR
    img_r = cv2.resize(img, (GMIC_W, GMIC_H), interpolation=interp)
    if img_r.max() > 255 or img_r.dtype != np.uint8:
        img_r = _normalize_uint8(img_r)
    cv2.imwrite(path, img_r)
    return ("resized", sfp, (GMIC_H / h, GMIC_W / w))


def resize_all(cropped_dir, pkl_cropped, output_dir):
    print(f"ÉTAPE 4 — Resize {GMIC_H}x{GMIC_W} + normalisation [0, 255]")
    import src.utilities.pickling as pickling  # GMIC

    exam_list = pickling.unpickle_from_file(pkl_cropped)
    all_pngs = glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True)

    nproc = min(8, os.cpu_count() or 4)
    with Pool(nproc) as pool:
        results = pool.map(_resize_one, [(p, cropped_dir) for p in all_pngs], chunksize=8)

    scale_map = {sfp: sc for st, sfp, sc in results if st == "resized"}
    resized = len(scale_map)
    print(f"  redimensionnées={resized} déjà-OK={sum(1 for r in results if r[0].startswith('skip'))}")

    # Recale les coordonnées stockées par le crop pour les images redimensionnées.
    for exam in exam_list:
        for view in VIEWS:
            for j, sfp in enumerate(exam.get(view, [])):
                if sfp not in scale_map:
                    continue
                sh, sw = scale_map[sfp]
                (ry1, ry2), rx = exam["rightmost_points"][view][j]
                exam["rightmost_points"][view][j] = (
                    (int(round(ry1 * sh)), int(round(ry2 * sh))), int(round(rx * sw)))
                by, (bx1, bx2) = exam["bottommost_points"][view][j]
                exam["bottommost_points"][view][j] = (
                    int(round(by * sh)), (int(round(bx1 * sw)), int(round(bx2 * sw))))
    pickling.pickle_to_file(pkl_cropped, exam_list)


# ── Étape 5 : Flip vues droites ──────────────────────────────────────────────

def _flip_one(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0
    cv2.imwrite(path, cv2.flip(img, 1))      # 1 = axe vertical = flip horizontal
    return 1


def apply_right_view_flip(cropped_dir, pkl_cropped):
    print("ÉTAPE 5 — Flip horizontal des vues droites (R-CC, R-MLO)")
    with open(pkl_cropped, "rb") as f:
        exam_list = pickle.load(f)
    paths = [os.path.join(cropped_dir, sfp + ".png")
             for exam in exam_list for view in _VIEWS_RIGHT
             for sfp in exam.get(view, [])
             if os.path.exists(os.path.join(cropped_dir, sfp + ".png"))]
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        flipped = sum(pool.map(_flip_one, paths, chunksize=8))
    print(f"  retournées={flipped}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(input_dir, output_dir, csv_path=None, num_processes=2):
    """Lance le pipeline complet sur `input_dir` (DICOM sous train_images/<pid>/).

    Retourne le chemin du data.pkl final.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    png_dir = os.path.join(output_dir, "png_images")
    cropped_dir = os.path.join(output_dir, "cropped_images")
    pkl_raw = os.path.join(output_dir, "exam_list_before_cropping.pkl")
    pkl_cropped = os.path.join(output_dir, "cropped_exam_list.pkl")
    pkl_final = os.path.join(output_dir, "data.pkl")

    if csv_path is None:
        csv_path = os.path.join(input_dir, "train.csv")
    raw_dir = os.path.join(input_dir, "train_images")
    if not os.path.isdir(raw_dir):
        raw_dir = input_dir

    os.makedirs(png_dir, exist_ok=True)
    convert_dcm_to_png(raw_dir, png_dir, csv_path, num_processes=num_processes)
    with open(os.path.join(output_dir, "source_dir.txt"), "w") as f:
        f.write(png_dir)
    build_exam_pkl(csv_path, png_dir, pkl_raw)
    run_crop(png_dir, cropped_dir, pkl_raw, pkl_cropped, num_processes=num_processes)
    resize_all(cropped_dir, pkl_cropped, output_dir)
    apply_right_view_flip(cropped_dir, pkl_cropped)
    shutil.copy(pkl_cropped, pkl_final)
    print(f"TERMINÉ — PNG dans {cropped_dir} | PKL final {pkl_final}")
    return pkl_final
