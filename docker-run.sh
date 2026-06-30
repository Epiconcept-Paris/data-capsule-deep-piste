#!/usr/bin/env bash
# Lance JupyterLab dans le conteneur, sur le GPU, avec le DÉPÔT entier monté.
# JupyterLab écoute sur 127.0.0.1:8888 de la machine hôte (la VM).
# Depuis le laptop : ssh -L 8888:localhost:8888 <alias-vm>  puis  http://localhost:8888
#
# Aucun chemin n'est codé en dur : tout part de l'endroit où ce script (donc le
# dépôt) a été cloné. On NE dépend PAS de $HOME.
set -euo pipefail

# Racine du dépôt = dossier de ce script, résolu en chemin absolu.
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

IMAGE="${IMAGE:-breast-cancer-course:latest}"
# Point de montage du dépôt dans le conteneur (= COURSE_ROOT de l'image).
COURSE_MNT="/home/deep-piste/course"

# Les notebooks lisent GMIC/selective-classification depuis modules/. Si les
# sous-modules ne sont pas initialisés, on le fait maintenant.
if [ ! -f modules/GMIC/models/sample_model_1.p ]; then
    echo ">> Sous-modules non initialisés -> git submodule update --init --recursive"
    git submodule update --init --recursive
fi

# Identifiants Kaggle, relatifs au dépôt (jamais via $HOME) :
#  - soit un dossier .kaggle/ à la racine du dépôt (kaggle.json), monté en ro ;
#  - soit un fichier .env (KAGGLE_USERNAME / KAGGLE_KEY).
# Les deux sont ignorés par git (.gitignore).
KAGGLE_ARGS=()
if [ -d "$REPO_DIR/.kaggle" ]; then
    KAGGLE_ARGS+=(-v "$REPO_DIR/.kaggle":/home/deep-piste/.kaggle:ro)
fi

ENV_ARGS=()
[ -f "$REPO_DIR/.env" ] && ENV_ARGS+=(--env-file "$REPO_DIR/.env")

if [ ${#KAGGLE_ARGS[@]} -eq 0 ] && [ ${#ENV_ARGS[@]} -eq 0 ]; then
    echo "AVERTISSEMENT : ni .kaggle/ ni .env trouvés -> le téléchargement Kaggle (ch1) échouera." >&2
fi

echo "Montage du dépôt : $REPO_DIR -> $COURSE_MNT"

docker run --rm -it \
    --gpus all \
    -p 127.0.0.1:8888:8888 \
    -v "$REPO_DIR":"$COURSE_MNT" \
    "${KAGGLE_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE"
