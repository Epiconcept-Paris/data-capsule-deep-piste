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

# Garantie « jamais root » : si on build en tant que root (UID/GID 0, typique sous
# WSL), on retombe sur 1000 pour que l'utilisateur du conteneur ne soit JAMAIS root.
[ "$HOST_UID" = "0" ] && { echo ">> build en root -> HOST_UID forcé à 1000 (jamais root dans le conteneur)"; HOST_UID=1000; }
[ "$HOST_GID" = "0" ] && HOST_GID=1000

cd "$(dirname "$0")"
echo ">> docker build -t ${IMAGE}  (HOST_UID=${HOST_UID} HOST_GID=${HOST_GID})"
docker build \
    --build-arg HOST_UID="${HOST_UID}" \
    --build-arg HOST_GID="${HOST_GID}" \
    -t "${IMAGE}" .
echo ">> Image construite : ${IMAGE}"
