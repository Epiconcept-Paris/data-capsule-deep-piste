"""Utilitaires partagés par les notebooks du cours.

Deux familles d'outils :

1. Chemins du repo (`course_root`, `data_in`, `data_work`, ...) : localisent les
   données et sous-modules sans aucun chemin codé en dur.

2. `flowchart()` : dessine un organigramme en SORTIE d'une cellule de code
   (matplotlib). On évite ainsi Mermaid, qui ne s'affiche que dans certaines
   versions de JupyterLab et reste blanc partout ailleurs.

3. Détection de capacités matérielles (`pick_device`, `dataloader_kwargs`,
   `describe_env`, ...) : permet aux notebooks d'entraînement de tourner AUSSI
   BIEN sur un petit PC portable / un conteneur Docker minimal que sur la VM GPU,
   sans jamais modifier le code. On ne suppose RIEN sur la machine (GPU ? combien
   de cœurs ? /dev/shm limité ?) : on détecte et on choisit des réglages sûrs.
"""
import os
import platform


def course_root():
    """Racine du repo cloné, telle que vue depuis le notebook (conteneur OU local).

    Aucun chemin n'est codé en dur : on suit l'endroit où la personne a fait son
    `git clone`.
      1. Si la variable d'environnement `COURSE_ROOT` existe (définie par l'image
         Docker = point de montage du repo), on l'utilise.
      2. Sinon on remonte les dossiers depuis ce fichier jusqu'à trouver le marqueur
         du repo (`pyproject.toml`) — fonctionne aussi hors conteneur.
    """
    env = os.environ.get("COURSE_ROOT")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    # Repli : le dossier du notebook lui-même.
    return os.path.dirname(os.path.abspath(__file__))


def data_in(*parts):
    """Données brutes en ENTRÉE : `<repo>/data/in/...` (téléchargements RSNA, CIFAR…)."""
    return os.path.join(course_root(), "data", "in", *parts)


def data_work(*parts):
    """Sorties PRODUITES : `<repo>/data/work/...` (prétraitements, crops, checkpoints…)."""
    return os.path.join(course_root(), "data", "work", *parts)


# Sortie du prétraitement (ch3) pour chaque jeu de données d'entrée.
# UNE seule table pour tout le cours : le ch3 ÉCRIT ici, les ch4+ LISENT ici. Tant que
# tout le monde passe par preprocess_dir(), écrivain et lecteurs ne peuvent pas diverger
# (le bug classique : le ch3 produit preprocess_sample et le ch4 cherche ailleurs).
_PREPROCESS_DIRS = {
    "rsna_sample": "preprocess_sample",   # échantillon de démo (ch1 section A)
    "rsna": "preprocess_image",           # dataset RSNA complet (ch1 section B)
}


def preprocess_dir(dataset, *parts):
    """Dossier de sortie du prétraitement du ch3, pour `dataset` ('rsna_sample' ou 'rsna').

    Exemples :
        preprocess_dir('rsna_sample')                     -> <repo>/data/work/preprocess_sample
        preprocess_dir('rsna', 'cropped_images')          -> <repo>/data/work/preprocess_image/cropped_images

    Lève une ValueError sur un nom inconnu : mieux vaut échouer tout de suite que
    construire un chemin qui n'existera jamais et laisser un `glob` renvoyer une liste vide.
    """
    if dataset not in _PREPROCESS_DIRS:
        raise ValueError(
            f"dataset inconnu : {dataset!r}. Valeurs possibles : {sorted(_PREPROCESS_DIRS)}")
    return data_work(_PREPROCESS_DIRS[dataset], *parts)


def sample_data(*parts):
    """Petits extraits de données RÉELLES bundlés avec le cours (committés, contrairement à
    `data_in`/`data_work` qui sont vides côté git) : `<repo>/notebooks/data_samples/...`."""
    return os.path.join(course_root(), "notebooks", "data_samples", *parts)


def save_run(tag, config, history=(), model=None, **extra):
    """Archive le bilan d'un entraînement en JSON horodaté ; renvoie le chemin écrit.

    Le notebook ne décrit que ce qui lui est PROPRE ; tout ce qui ne dépend pas du chapitre
    est rempli ici (horodatage, GPU, version de torch, pic mémoire CUDA, comptes de
    paramètres, loss finale) — sinon chaque chapitre recopierait le même bloc.

    Args:
        tag: préfixe du dossier. 'ch7' -> `data_samples/ch7_runs/<AAAAMMJJ-HHMMSS>.json`.
            Un fichier PAR run : ils s'accumulent au lieu de s'écraser, c'est tout l'intérêt.
        config: les réglages du run (résolution, batch, lr, régime de gel...). Le champ à
            relire en premier quand on se demande « qu'est-ce que j'avais lancé ? ».
        history: un dict par epoch (p. ex. `{'epoch': 1, 'loss': 3.2}`). Sa DERNIÈRE entrée
            fournit `final_loss`, d'où l'intérêt de la remplir dans l'ordre.
        model: si fourni, on en tire params totaux / entraînables / gelés. Évite au notebook
            de recompter, et rend l'archive auto-suffisante pour vérifier le régime après coup.
        **extra: tout champ libre propre au chapitre (`data=...`, `ms_per_step=...`).

    Deux contraintes du .gitignore dictent le nommage — sans elles l'archive serait perdue :
      * le dossier s'appelle `<tag>_runs` et NON `runs`, car `runs/` y est ignoré ;
      * on écrit du JSON, pas du .p/.pkl/.npz/.pt, tous ignorés aussi.
    """
    import json
    from datetime import datetime

    history = list(history)
    rec = {"tag": tag, "date": datetime.now().isoformat(timespec="seconds"),
           "config": config, "history": history, **extra}
    if history:
        rec["final_loss"] = history[-1].get("loss")

    import torch  # présent dès qu'on a entraîné quelque chose

    rec["env"] = {"torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if model is not None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rec["params"] = {"total": total, "trainable": trainable, "frozen": total - trainable}

    d = sample_data(f"{tag}_runs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return path


def predictions_frame(p, y_true, split=None, threshold=0.5):
    """Table de prédictions au format lu par la classification sélective (ch8).

    Reproduit le schéma de `data_samples/gmic_ens5_nyu_sgp_set.csv`. Les trois colonnes
    OBLIGATOIRES sont celles que `sgp_dicho` / `sgp_greedy_search` lisent réellement :
    elles filtrent `Sn.loc[Sn.kappa >= theta]` puis comptent les erreurs sur `y_pred`/`y_true`.

    Args:
        p: probas prédites de la classe positive, une par échantillon. Sur RSNA, un
            échantillon = un SEIN (moyenne des vues CC et MLO), pas une image.
        y_true: labels 0/1, même longueur que `p`.
        split: 'cal' / 'test' par échantillon, ou None pour omettre la colonne. La
            calibration doit être un jeu TENU À L'ÉCART de l'entraînement : la garantie
            suppose des échantillons non utilisés pour ajuster le modèle.
        threshold: seuil de décision de `y_pred` (0.5 par défaut). Ne PAS le confondre avec
            le θ de l'abstention : `threshold` décide de la classe, θ décide si on répond.

    Colonnes produites : `p`, `kappa` (= max(p, 1-p), la confiance — elle ne descend jamais
    sous 0.5, d'où le `theta_min=0.5` par défaut de ces fonctions), `y_pred`, `y_true`,
    et `split` si fourni.
    """
    import numpy as np
    import pandas as pd

    p = np.asarray(p, dtype=float)
    df = pd.DataFrame({"p": p,
                       "kappa": np.maximum(p, 1.0 - p),
                       "y_pred": (p >= threshold).astype(int),
                       "y_true": np.asarray(y_true).astype(int)})
    if split is not None:
        df["split"] = split
    return df


def auc_score(y_true, y_score):
    """AUC ROC, ou `None` si l'AUC n'est pas définie sur ces données.

    Pourquoi l'AUC plutôt que la loss pour suivre un entraînement : la loss peut baisser
    alors que le modèle ne SÉPARE rien (sur un jeu à 2 % de positifs, « réponds toujours
    non » donne déjà une loss basse). L'AUC ne regarde que l'ORDRE des scores — 0.5 = hasard,
    1.0 = séparation parfaite — et ne peut pas être flattée par le déséquilibre des classes.

    Args:
        y_true: labels 0/1.
        y_score: score continu (une proba, pas une décision binarisée — passer `y_pred`
            écraserait l'information d'ordre et plafonnerait artificiellement l'AUC).

    Renvoie None si `y_true` ne contient qu'une seule classe : l'AUC mesure une séparation
    entre deux groupes, elle n'existe pas s'il n'y en a qu'un. Ce cas est FRÉQUENT sur un
    petit batch en cancer du sein, d'où ce garde-fou plutôt qu'une exception sklearn.
    """
    from sklearn.metrics import roc_auc_score

    if len(set(int(v) for v in y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def auc_ovr(y_true, proba, names="ABCD"):
    """AUC 1-vs-all pour chaque classe, plus la moyenne macro. Dict {classe: auc, "macro": ...}.

    Pourquoi l'AUC 1-vs-all plutôt que l'accuracy sur du multiclasse déséquilibré : avec des
    densités mammaires B et C qui font 84 % des cas, un modèle qui répond TOUJOURS « B »
    décroche une accuracy flatteuse sans rien discriminer. Ici ce collapse tombe à 0.500
    exactement — l'AUC ne regarde que l'ORDRE des scores, elle ne peut pas être gonflée par
    la fréquence d'une classe.

    Args:
        y_true: labels entiers 0..len(names)-1.
        proba: tableau (n, len(names)) de PROBABILITÉS (softmax), pas des décisions
            argmax — binariser écraserait l'information d'ordre dont vit l'AUC.
        names: noms des classes, dans l'ordre des colonnes de `proba`.

    Chaque classe absente de `y_true` vaut None (délégué à `auc_score` : une AUC compare deux
    groupes, elle n'existe pas s'il n'y en a qu'un). C'est fréquent sur un petit échantillon
    dont le split laisse une seule densité en test — et c'est précisément le cas où
    l'accuracy affiche 1.000 en ne mesurant rien. "macro" ne moyenne que les classes définies,
    et vaut None si aucune ne l'est.
    """
    import numpy as np

    y_true, proba = np.asarray(y_true), np.asarray(proba)
    out = {names[k]: auc_score((y_true == k).astype(int), proba[:, k])
           for k in range(len(names))}
    defined = [v for v in out.values() if v is not None]
    out["macro"] = float(np.mean(defined)) if defined else None
    return out


def list_runs(tag):
    """Runs archivés pour `tag`, du plus ancien au plus récent (liste de chemins)."""
    import glob

    d = sample_data(f"{tag}_runs")
    return sorted(glob.glob(os.path.join(d, "*.json"))) if os.path.isdir(d) else []


def gmic_dir():
    """Sous-module GMIC : `<repo>/modules/GMIC`."""
    return os.path.join(course_root(), "modules", "GMIC")


def selclass_dir():
    """Sous-module selective-classification : `<repo>/modules/selective-classification`."""
    return os.path.join(course_root(), "modules", "selective-classification")


# ---------------------------------------------------------------------------
# Détection de capacités matérielles
# ---------------------------------------------------------------------------
#
# Pourquoi ce bloc existe :
#   Un DataLoader PyTorch avec `num_workers > 0` lance des processus enfants qui
#   renvoient les batches au processus principal via la MÉMOIRE PARTAGÉE
#   (/dev/shm sous Linux). Dans un conteneur Docker, /dev/shm fait 64 Mo par
#   défaut : un seul batch d'images haute résolution le sature, le worker reçoit
#   un SIGBUS ("Bus error") et meurt. D'où l'erreur classique :
#       "DataLoader worker (pid ...) is killed by signal: Bus error"
#   Le vrai correctif côté infra est `docker run --ipc=host` (fait dans
#   docker-run.sh ; `--shm-size=8g` marcherait aussi), mais on veut que le
#   NOTEBOOK reste robuste même quand l'infra est mal configurée.
#
# La règle d'or : détecter les capacités réelles, ne jamais les supposer.


def pick_device(verbose=True):
    """Retourne le meilleur `torch.device` disponible : CUDA > MPS (Mac) > CPU.

    - CUDA : GPU NVIDIA (la VM du cours, la plupart des serveurs).
    - MPS  : GPU Apple Silicon (M1/M2/M3) -> accélère sur les Mac récents.
    - CPU  : repli universel, marche partout.

    À utiliser à la place de la ligne répétée dans chaque notebook :
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    import torch  # import paresseux : les helpers de chemin restent utilisables sans torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Device sélectionné : {device}")
        if device.type == "cuda":
            print(f"  GPU : {torch.cuda.get_device_name(0)}")
    return device


def _shm_free_bytes():
    """Octets libres dans /dev/shm, ou None si la notion ne s'applique pas.

    Renvoie None sous Windows / macOS (pas de /dev/shm) : dans ce cas on
    désactivera les workers de toute façon (voir `recommended_num_workers`).
    """
    try:
        st = os.statvfs("/dev/shm")           # disponible uniquement sous Linux
        return st.f_bavail * st.f_frsize
    except (OSError, AttributeError):
        return None


def recommended_num_workers(cap=4):
    """Nombre de workers DataLoader adapté à la machine (0 = aucun sous-processus).

    Logique de décision :
      * Variable d'env COURSE_NUM_WORKERS -> on respecte le choix explicite.
      * Windows / macOS -> 0. Le multiprocessing y utilise `spawn`, qui réimporte
        le module : fragile et souvent cassé dans un notebook. Plus lent mais sûr.
      * Linux avec /dev/shm trop petit (< 512 Mo, typique d'un Docker par défaut)
        -> 0, sinon SIGBUS garanti sur des images volumineuses.
      * Sinon -> min(cap, nb de cœurs), borné pour ne pas saturer un petit CPU.
    """
    forced = os.environ.get("COURSE_NUM_WORKERS")
    if forced is not None:
        return max(0, int(forced))

    if platform.system() != "Linux":          # Windows / macOS : pas de fork fiable
        return 0

    shm = _shm_free_bytes()
    if shm is not None and shm < 512 * 1024**2:   # /dev/shm < 512 Mo -> danger SIGBUS
        return 0

    return min(cap, os.cpu_count() or 1)


def dataloader_kwargs(batch_size=16, shuffle=False, sampler=None, num_workers=None):
    """Construit les kwargs d'un DataLoader adaptés au matériel courant.

    Réglés automatiquement :
      * num_workers      -> voir recommended_num_workers()
      * pin_memory       -> True seulement avec un GPU CUDA (inutile ailleurs)
      * persistent_workers / prefetch_factor -> seulement si num_workers > 0
        (les passer avec 0 worker lève une erreur)

    `sampler` et `shuffle` sont mutuellement exclusifs : si un sampler est fourni
    (ex. WeightedRandomSampler pour équilibrer les classes), `shuffle` est ignoré.

    À utiliser partout à la place d'un DataLoader nu, en déballant les kwargs :
        from torch.utils.data import DataLoader
        from course_utils import dataloader_kwargs
        train_loader = DataLoader(train_ds, **dataloader_kwargs(batch_size=64, shuffle=True))
        bal_loader   = DataLoader(train_ds, **dataloader_kwargs(batch_size=16, sampler=sampler))
    """
    import torch

    if num_workers is None:
        num_workers = recommended_num_workers()

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),   # accélère le transfert CPU->GPU
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True     # ne pas recréer les workers à chaque epoch
        kwargs["prefetch_factor"] = 2           # chaque worker précharge 2 batches
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    return kwargs


def describe_env():
    """Affiche un rapport des capacités détectées (à mettre en tête de notebook)."""
    import torch

    nw = recommended_num_workers()
    shm = _shm_free_bytes()
    print("=== Environnement détecté ===")
    print(f"OS                 : {platform.system()} ({platform.machine()})")
    print(f"PyTorch            : {torch.__version__}")
    print(f"Device             : {pick_device(verbose=False)}")
    if torch.cuda.is_available():
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"CPU cœurs          : {os.cpu_count()}")
    if shm is not None:
        print(f"/dev/shm libre     : {shm / 1024**3:.2f} Go")
    else:
        print("/dev/shm           : indisponible (Windows/macOS)")
    print(f"DataLoader workers : {nw}" + ("  (mode mono-processus, sans mémoire partagée)"
                                          if nw == 0 else ""))
    print("=============================")


def flowchart(steps, title=None, width=8.5, box_h=0.62, gap=0.45,
              facecolor="#e7f0fb", edgecolor="#2b6cb0", fontsize=11):
    """Dessine un organigramme vertical : une boîte par étape, flèches entre elles.

    `steps` : liste de chaînes (du haut vers le bas).
    Le diagramme est rendu via `plt.show()` -> visible dans tout Jupyter, nbconvert
    et l'aperçu GitHub.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n = len(steps)
    unit = box_h + gap
    fig, ax = plt.subplots(figsize=(width, n * unit + 0.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(-gap, n * unit)
    ax.axis("off")
    for i, label in enumerate(steps):
        y = (n - 1 - i) * unit
        ax.add_patch(FancyBboxPatch((1, y), 8, box_h,
                     boxstyle="round,pad=0.08", linewidth=1.6,
                     facecolor=facecolor, edgecolor=edgecolor))
        ax.text(5, y + box_h / 2, label, ha="center", va="center", fontsize=fontsize)
        if i < n - 1:
            ax.annotate("", xy=(5, y - gap), xytext=(5, y),
                        arrowprops=dict(arrowstyle="-|>", color=edgecolor, lw=1.8))
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.show()
