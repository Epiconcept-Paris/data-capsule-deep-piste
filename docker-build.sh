#!/usr/bin/env bash
# Construit l'image du cours (env GPU + deps Python via uv).
# Rien d'autre n'est intégré à l'image : notebooks, sous-modules et données sont
# montés au runtime (voir docker-run.sh).
#
# L'utilisateur du conteneur reçoit l'UID/GID de CELUI QUI BUILD, pour que les
# fichiers écrits dans le dépôt monté ne soient pas la propriété de root.
set -euo pipefail

IMAGE="${IMAGE:-breast-cancer-course:latest}"

# Si lancé via sudo, on veut l'UID/GID de l'utilisateur réel, pas ceux de root.
REAL_USER="${SUDO_USER:-$(id -un)}"
HOST_UID="$(id -u "$REAL_USER")"
HOST_GID="$(id -g "$REAL_USER")"

cd "$(dirname "$0")"
echo ">> docker build -t ${IMAGE}  (HOST_UID=${HOST_UID} HOST_GID=${HOST_GID})"
docker build \
    --build-arg HOST_UID="${HOST_UID}" \
    --build-arg HOST_GID="${HOST_GID}" \
    -t "${IMAGE}" .
echo ">> Image construite : ${IMAGE}"
