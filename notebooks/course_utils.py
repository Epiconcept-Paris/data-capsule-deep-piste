"""Utilitaires partagés par les notebooks du cours.

Deux familles d'outils :

1. Chemins du repo (`course_root`, `data_in`, `data_work`, ...) : localisent les
   données et sous-modules sans aucun chemin codé en dur.

2. `flowchart()` : dessine un organigramme en SORTIE d'une cellule de code
   (matplotlib). On évite ainsi Mermaid, qui ne s'affiche que dans certaines
   versions de JupyterLab et reste blanc partout ailleurs.

3. Détection de capacités matérielles (`pick_device`, `make_loader`,
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


def data_path(*parts):
    """Chemin sous `<repo>/data/...` (volume persistant, ignoré par git)."""
    return os.path.join(course_root(), "data", *parts)


def data_in(*parts):
    """Données brutes en ENTRÉE : `<repo>/data/in/...` (téléchargements RSNA, CIFAR…)."""
    return data_path("in", *parts)


def data_work(*parts):
    """Sorties PRODUITES : `<repo>/data/work/...` (prétraitements, crops, checkpoints…)."""
    return data_path("work", *parts)


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
#   Le vrai correctif côté infra est `docker run --shm-size=8g`, mais on veut que
#   le NOTEBOOK reste robuste même quand l'infra est mal configurée.
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
    Utile quand on veut les kwargs sans créer le loader (ex. bench multi-batch).
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


def make_loader(dataset, batch_size=16, shuffle=False, sampler=None, num_workers=None):
    """DataLoader portable, à utiliser partout à la place de `DataLoader(...)`.

    Exemple :
        from course_utils import make_loader
        train_loader = make_loader(train_ds, batch_size=64, shuffle=True)
        val_loader   = make_loader(val_ds,   batch_size=128)              # eval
        bal_loader   = make_loader(train_ds, batch_size=16, sampler=sampler)
    """
    from torch.utils.data import DataLoader

    return DataLoader(dataset, **dataloader_kwargs(
        batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers))


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
