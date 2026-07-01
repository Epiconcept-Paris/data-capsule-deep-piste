# ============================================================================
#  Dockerfile — image du cours (environnement GPU + JupyterLab)
# ============================================================================
# CONCEPT CLÉ : chaque instruction ci-dessous crée une COUCHE d'image, mise en
# cache. Docker les exécute de haut en bas ; dès qu'une ligne (ou un fichier
# qu'elle COPY) change, cette couche ET toutes celles du dessous sont refaites.
# -> on met ce qui change RAREMENT en haut (image de base, dépendances) et ce
#    qui change SOUVENT en bas.
#
# OÙ VONT LES DÉPENDANCES ? Dans l'environnement Python de l'image, c.-à-d.
# /opt/conda/lib/python3.11/site-packages/ (l'env conda hérité de l'image de
# base). pip et uv ne "stockent" rien en eux-mêmes : ce sont des INSTALLEURS qui
# copient les paquets dans ce site-packages. Comme ça se passe au BUILD, ces
# fichiers sont figés dans une couche -> présents dans tout conteneur lancé.
#
# L'image ne contient QUE cet environnement. Le dépôt (notebooks, modules/,
# data/) est MONTÉ en volume au runtime (docker-run.sh), pas copié dans l'image.
# ============================================================================

# --- 1) Image de base : on part d'un environnement tout prêt -----------------
# Fournit Python 3.11 + PyTorch 2.4.1 + CUDA 12.1 + cuDNN, déjà installés dans
# /opt/conda. C'est pour ça que torch/torchvision NE sont PAS réinstallés + bas.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# --- 2) Variables d'environnement (valent au build ET dans le conteneur) -----
# DEBIAN_FRONTEND=noninteractive : apt ne pose aucune question bloquante.
# PYTHONUNBUFFERED=1             : les print sortent tout de suite (logs en direct).
# PIP_NO_CACHE_DIR=1             : pip ne garde pas de cache -> image plus légère.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- 3) Dépendances SYSTÈME (paquets Linux, pas Python) ----------------------
# libgl1 + libglib2.0-0 : requis par OpenCV (cv2) pour le pré-traitement (ch2.5).
# Tout est chaîné dans UN SEUL RUN, avec le rm des listes apt dans la MÊME
# couche : sinon la liste des paquets resterait dedans et alourdirait l'image.
# --no-install-recommends : n'installe que le strict nécessaire.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- 4) Dépendances PYTHON (le cœur de l'installation) -----------------------
# On copie d'ABORD pyproject.toml SEUL (avant les notebooks) : tant que la liste
# des deps ne change pas, cette couche (longue) reste en cache même si tu édites
# un notebook. C'est la fameuse optimisation du cache de couches.
COPY pyproject.toml /tmp/pyproject.toml
# Étape 1 : pip installe "uv" (un installeur ultra-rapide) dans le site-packages.
# Étape 2 : uv installe TES deps (kornia, pydicom, opencv, jupyterlab...) au MÊME
#   endroit. --system = "installe dans l'env Python existant (celui de l'image,
#   /opt/conda)", et NON dans un .venv séparé. torch absent du fichier -> intact.
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r /tmp/pyproject.toml

# --- 5) Utilisateur non-root aligné sur TOI (permissions du volume monté) ----
# ARG = variable disponible SEULEMENT pendant le build (fournie par
# docker-build.sh via --build-arg). Valeur par défaut 1000 si non fournie.
ARG HOST_UID=1000
ARG HOST_GID=1000
# On crée l'utilisateur "deep-piste" avec TON UID/GID. Ainsi les fichiers qu'il
# écrit dans le dépôt monté t'appartiennent (pas à root -> pas besoin de sudo
# pour les modifier/supprimer). Linux compare des NUMÉROS d'UID, pas des noms
# (le nom "deep-piste" est donc cosmétique).
# "2>/dev/null || true" : ne pas planter le build si le groupe/user existe déjà.
RUN groupadd -g "${HOST_GID}" deep-piste 2>/dev/null || true \
    && useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash deep-piste 2>/dev/null || true \
    && mkdir -p /home/deep-piste/course \
    && chown -R "${HOST_UID}:${HOST_GID}" /home/deep-piste

# --- 6) HOME + COURSE_ROOT ---------------------------------------------------
# HOME        : où JupyterLab range sa config et son cache.
# COURSE_ROOT : le point de montage du dépôt ; les notebooks le lisent via
#   course_utils.course_root() -> AUCUN chemin en dur, ça marche partout.
ENV HOME=/home/deep-piste \
    COURSE_ROOT=/home/deep-piste/course

# --- 7) On bascule sur l'utilisateur non-root + son dossier de travail -------
# Tout ce qui suit (et le conteneur au démarrage) tourne en tant que deep-piste.
USER deep-piste
WORKDIR /home/deep-piste/course

# --- 8) Port documenté (purement déclaratif) ---------------------------------
# Indique que le service écoute sur 8888. N'ouvre PAS le port tout seul : c'est
# "docker run -p" (ou le tunnel SSH) qui le publie réellement côté hôte.
EXPOSE 8888

# --- 9) Commande de démarrage (exécutée au "docker run", PAS au build) -------
# Lance JupyterLab :
#  --ip=0.0.0.0         : écoute sur toutes les interfaces DU conteneur.
#  --no-browser         : n'essaie pas d'ouvrir un navigateur (il n'y en a pas).
#  --token= --password= : PAS d'authentification -> acceptable UNIQUEMENT parce
#                         que l'accès passe par un tunnel SSH vers 127.0.0.1.
#  --root_dir=...       : l'explorateur JupyterLab s'ouvre sur le dépôt monté.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/home/deep-piste/course"]
