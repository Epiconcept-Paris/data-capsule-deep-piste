#!/usr/bin/env bash
# ============================================================================
#  docker-run.sh — lance JupyterLab dans un conteneur, sur le GPU
# ============================================================================
# Le DÉPÔT entier est monté dans le conteneur -> tu modifies un notebook ou tu
# ajoutes des données SANS reconstruire l'image.
# JupyterLab écoute sur 127.0.0.1:8888 de la machine hôte (la VM). Depuis le
# laptop : ssh -L 8888:localhost:8888 <alias-vm>  puis  http://localhost:8888
#
# Aucun chemin n'est codé en dur : tout part de l'endroit où ce script (donc le
# dépôt) a été cloné. On NE dépend PAS de $HOME.
# ============================================================================
set -euo pipefail   # cf. docker-build.sh : stoppe à la 1re erreur / variable non définie / erreur dans un pipe

# Racine du dépôt = dossier de ce script, résolu en chemin ABSOLU.
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

IMAGE="${IMAGE:-breast-cancer-course:latest}"
# Point de montage du dépôt DANS le conteneur (= COURSE_ROOT défini dans l'image).
COURSE_MNT="/home/deep-piste/course"

# Les notebooks lisent GMIC / selective-classification depuis modules/ (des
# sous-modules git). S'ils ne sont pas encore récupérés, on le fait maintenant.
if [ ! -f modules/GMIC/models/sample_model_1.p ]; then
    echo ">> Sous-modules non initialisés -> git submodule update --init --recursive"
    git submodule update --init --recursive
fi

# Identifiants Kaggle, relatifs au dépôt (jamais via $HOME). Le dossier .kaggle/
# est versionné (mais son CONTENU est ignoré par git) ; on le monte en LECTURE
# SEULE (:ro) sur /home/deep-piste/.kaggle dans le conteneur.
#   - access_token : token récent KGAT_ (recommandé ; seul compatible kaggle 2.x)
#   - kaggle.json  : ancien format username + key
mkdir -p "$REPO_DIR/.kaggle"
# Tableau bash d'arguments -v (montage). On l'injecte plus bas via "${KAGGLE_ARGS[@]}".
KAGGLE_ARGS=(-v "$REPO_DIR/.kaggle":/home/deep-piste/.kaggle:ro)

# Avertit si aucun identifiant présent -> le téléchargement du ch1 échouerait.
if [ ! -f "$REPO_DIR/.kaggle/access_token" ] && [ ! -f "$REPO_DIR/.kaggle/kaggle.json" ]; then
    echo "AVERTISSEMENT : aucun identifiant Kaggle dans $REPO_DIR/.kaggle/" >&2
    echo "  -> dépose ton token dans .kaggle/access_token (recommandé) ou un kaggle.json," >&2
    echo "     sinon le téléchargement (ch1) échouera." >&2
fi

echo "Montage du dépôt : $REPO_DIR -> $COURSE_MNT"

# docker run, options :
#   --rm                   : supprime le conteneur à l'arrêt (aucun résidu).
#   -it                    : interactif + terminal (voir les logs, Ctrl-C).
#   --gpus all             : donne accès aux GPU NVIDIA (nvidia-container-toolkit requis).
#   -p 127.0.0.1:8888:8888 : publie le port 8888 UNIQUEMENT sur la boucle locale de l'hôte.
#   -v REPO_DIR:COURSE_MNT : monte le dépôt entier dans le conteneur.
#   "${KAGGLE_ARGS[@]}"    : ajoute le montage .kaggle défini plus haut.
#   "$IMAGE"               : l'image à lancer (sa CMD démarre JupyterLab).
docker run --rm -it \
    --gpus all \
    -p 127.0.0.1:8888:8888 \
    -v "$REPO_DIR":"$COURSE_MNT" \
    "${KAGGLE_ARGS[@]}" \
    "$IMAGE"
