# Breast Cancer Course — image GPU + JupyterLab
#
# Même stack torch que la VM (torch 2.4.1 / cu121 / torchvision 0.19.1).
# L'image ne contient QUE l'environnement Python/CUDA. Le dépôt entier (notebooks,
# sous-modules modules/GMIC + modules/selective-classification, et data/) est MONTÉ
# en volume au runtime sur /home/deep-piste/course (cf. docker-run.sh), de sorte que
# JupyterLab affiche exactement l'arborescence du dépôt cloné — où qu'il ait été cloné.
#
# L'utilisateur du conteneur reprend l'UID/GID de celui qui construit l'image
# (build-args HOST_UID/HOST_GID) : les fichiers écrits dans le dépôt monté lui
# appartiennent, et non à root.

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système :
#  - libgl1 / libglib2.0-0 : requis par opencv pour le pré-traitement (ch 2.5)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python des 6 chapitres, installées avec uv (en tant que root, dans
# l'env Python de l'image où vit torch -> torch/torchvision NE sont PAS réinstallés,
# car absents de pyproject.toml).
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r /tmp/pyproject.toml

# Utilisateur non-root, aligné sur celui qui build (évite les fichiers root dans le
# dépôt monté). Le nom est cosmétique ; seuls l'UID/GID comptent pour les permissions.
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g "${HOST_GID}" deep-piste 2>/dev/null || true \
    && useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash deep-piste 2>/dev/null || true \
    && mkdir -p /home/deep-piste/course \
    && chown -R "${HOST_UID}:${HOST_GID}" /home/deep-piste

# HOME = home de l'utilisateur (config/cache Jupyter). COURSE_ROOT = point de montage
# du dépôt : les notebooks s'y réfèrent (course_utils.course_root()), aucun chemin en dur.
ENV HOME=/home/deep-piste \
    COURSE_ROOT=/home/deep-piste/course

USER deep-piste
WORKDIR /home/deep-piste/course

EXPOSE 8888

# Accès via tunnel SSH vers 127.0.0.1 → token désactivé (réseau local au tunnel).
# root_dir = le dépôt monté -> l'explorateur JupyterLab montre l'arbo du dépôt
# (notebooks/, modules/, data/...).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/home/deep-piste/course"]
