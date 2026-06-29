#!/usr/bin/env bash
# Lance JupyterLab dans le conteneur, sur le GPU, avec les volumes montés.
# JupyterLab écoute sur 127.0.0.1:8888 de la machine hôte (la VM).
# Depuis le laptop : ssh -L 8888:localhost:8888 <alias-vm>  puis  http://localhost:8888
set -euo pipefail

# joshua n'est pas dans le groupe docker -> ce script est lancé via `sudo`.
# Sous sudo, $HOME vaut /root : on résout le home de l'utilisateur RÉEL,
# sinon les montages .kaggle/data pointent vers /root (vide) au lieu de /home/joshua.
REAL_USER="${SUDO_USER:-$(id -un)}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
REAL_HOME="${REAL_HOME:-$HOME}"

IMAGE="${IMAGE:-breast-cancer-course:latest}"
DATA_DIR="${DATA_DIR:-$REAL_HOME/data}"
KAGGLE_DIR="${KAGGLE_DIR:-$REAL_HOME/.kaggle}"

cd "$(dirname "$0")"
mkdir -p "$DATA_DIR"

ENV_ARGS=()
[ -f .env ] && ENV_ARGS+=(--env-file .env)

KAGGLE_ARGS=()
if [ -d "$KAGGLE_DIR" ]; then
    KAGGLE_ARGS+=(-v "$KAGGLE_DIR":/root/.kaggle:ro)
else
    echo "AVERTISSEMENT : $KAGGLE_DIR absent -> Kaggle ne sera pas monte." >&2
fi

echo "Montages : data=$DATA_DIR  kaggle=$KAGGLE_DIR"

docker run --rm -it \
    --gpus all \
    -p 127.0.0.1:8888:8888 \
    -v "$DATA_DIR":/root/data \
    -v "$PWD/notebooks":/root/course/notebooks \
    "${KAGGLE_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE"
