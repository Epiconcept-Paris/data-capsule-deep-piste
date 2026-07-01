#!/usr/bin/env bash
# ============================================================================
#  docker-build.sh — construit l'image du cours (env GPU + deps Python via uv)
# ============================================================================
# Rien d'autre n'est intégré à l'image : notebooks, sous-modules et données sont
# montés au runtime (voir docker-run.sh).
#
# Point clé : l'utilisateur du conteneur reçoit l'UID/GID de CELUI QUI BUILD,
# pour que les fichiers écrits dans le dépôt monté ne soient pas la propriété de
# root.
# ============================================================================

# set -e : stoppe au 1er échec ; -u : utiliser une variable non définie = erreur ;
# -o pipefail : une erreur au milieu d'un pipe fait échouer toute la commande.
set -euo pipefail

# Nom:tag de l'image. Surchargable via la variable d'env IMAGE, sinon défaut.
IMAGE="${IMAGE:-breast-cancer-course:latest}"

# Si lancé via sudo, $SUDO_USER = ton vrai nom d'utilisateur ; sinon l'utilisateur
# courant. On veut TON UID/GID, pas ceux de root.
REAL_USER="${SUDO_USER:-$(id -un)}"
HOST_UID="$(id -u "$REAL_USER")"   # ton identifiant utilisateur (numérique)
HOST_GID="$(id -g "$REAL_USER")"   # ton identifiant de groupe (numérique)

# Garde-fou "jamais root" : sous WSL on build souvent en root (UID/GID 0). Dans ce
# cas on retombe sur 1000 pour que l'utilisateur du conteneur ne soit JAMAIS root.
[ "$HOST_UID" = "0" ] && { echo ">> build en root -> HOST_UID forcé à 1000 (jamais root dans le conteneur)"; HOST_UID=1000; }
[ "$HOST_GID" = "0" ] && HOST_GID=1000

cd "$(dirname "$0")"   # se placer à la racine du dépôt (= dossier de ce script)
echo ">> docker build -t ${IMAGE}  (HOST_UID=${HOST_UID} HOST_GID=${HOST_GID})"

# docker build :
#   --build-arg HOST_UID/GID : injecte ces valeurs dans les ARG du Dockerfile.
#   -t IMAGE                 : donne un nom:tag à l'image produite.
#   .                        : le "contexte de build" = ce dossier ; c'est là que
#                              l'instruction COPY va chercher pyproject.toml.
docker build \
    --build-arg HOST_UID="${HOST_UID}" \
    --build-arg HOST_GID="${HOST_GID}" \
    -t "${IMAGE}" .
echo ">> Image construite : ${IMAGE}"
