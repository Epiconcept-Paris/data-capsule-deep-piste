# Breast Cancer Course — Deep Learning sur mammographies

Cours pratique en 8 chapitres, du PyTorch de base jusqu'au **contrôle de risque
garanti** (abstention / selective classification), appliqué à la détection du
cancer du sein sur mammographies (jeu de données **RSNA**).

Chaque chapitre est un **notebook Jupyter qui exécute réellement du code** : il
entraîne de vrais réseaux, télécharge des jeux de données, et reproduit les
résultats. L'environnement GPU complet est packagé dans une **image Docker** qui ne
contient *que* cet environnement. Le **dépôt entier est monté en volume** dans le
conteneur au runtime : JupyterLab affiche donc exactement l'arborescence du dépôt
cloné — peu importe où il a été cloné (Bureau, WSL, VM…), **aucun chemin n'est codé
en dur**. Les deux dépôts externes utilisés (`GMIC` et `selective-classification`)
sont des **sous-modules git** (dossier `modules/`) : un `git clone --recurse-submodules`
les récupère, et comme ils sont montés (pas copiés), les mettre à jour ne demande
aucun rebuild.

---

## Architecture

```mermaid
flowchart LR
    laptop["💻 Laptop\nnavigateur"] -- "tunnel SSH\n-L 8888:localhost:8888" --> vmport["127.0.0.1:8888"]

    subgraph VM["🖥️ Hôte GPU (VM Scaleway / WSL / Bureau…)"]
        subgraph repo["📁 dépôt cloné = $REPO_DIR (où qu'il soit)"]
            mNb["notebooks/"]
            mMod["modules/GMIC + modules/selective-classification\n(sous-modules)"]
            mData["data/in (téléchargements) + data/work (prétraitements) — ignoré par git"]
            mKag[".kaggle/  (token API : access_token)"]
        end
        vmport --> container
        subgraph container["🐳 Conteneur (run en tant que deep-piste, UID du builder)"]
            jlab["JupyterLab + kernel GPU\nroot_dir = COURSE_ROOT"]
        end
        repo -. "monté → /home/deep-piste/course (= COURSE_ROOT)" .-> container
    end

    kserv["☁️ Kaggle\njeu RSNA ~300 Go"] -- "ch.1 : download" --> mData
```

Le kernel (donc les entraînements) tourne **dans le conteneur, sur le GPU de l'hôte**.
Le tunnel SSH ne transporte que l'interface web. **Un seul montage** : le dépôt cloné
→ `COURSE_ROOT` dans le conteneur. Notebooks, sous-modules et données suivent donc le
dépôt, et l'explorateur JupyterLab montre exactement son arborescence. L'image ne
contient que l'environnement Python/CUDA.

---

## Les chapitres

| # | Notebook | Sujet | GPU |
|---|----------|-------|-----|
| 1 | `notebooks/01_download_data.ipynb` | **Télécharger les données RSNA via la clé Kaggle** (prérequis aux ch. 3→7) | non |
| 2 | `notebooks/02_pytorch_basics.ipynb` | PyTorch de base : tenseurs, autograd, un réseau de vision simple | non (CPU OK) |
| 3 | `notebooks/03_preprocessing.ipynb` | Pré-traitement des mammographies : DICOM → PNG, crop, conversions qui libèrent le CPU (ne laisser que le décodage) | recommandé |
| 4 | `notebooks/04_resnet18_breast_density.ipynb` | Entraînement multiclasse avec un réseau connu (ResNet-18) — cas d'usage : **densité mammaire** | oui |
| 5 | `notebooks/05_bench_normalisation.ipynb` | *Aparté de performance (optionnel)* — qui doit faire le z-score : les workers CPU ou le GPU ? Mesure du débit de la pipeline complète | oui |
| 6 | `notebooks/06_gmic_architecture.ipynb` | Architecture d'un réseau récent : **GMIC** (ensemble de réseaux) — cas d'usage : **cancer malin RSNA** | oui |
| 7 | `notebooks/07_gmic_finetuning.ipynb` | Fine-tuning de GMIC sur RSNA | oui |
| 8 | `notebooks/08_risk_control_abstention.ipynb` | **Risques garantis** (selective classification, papier Beyond Accuracy) et comment appliquer ces fonctions | non (CPU OK) |

> Le chapitre 2 et le chapitre 8 peuvent tourner sur une machine sans GPU
> puissant. Les chapitres 3 à 7 supposent un GPU NVIDIA (entraînements réels).
> Le chapitre 5 est un aparté de performance : à lire après le ch4, il n'est
> prérequis de rien.

**Un seul curseur pour les données.** Les chapitres 3, 4 et 5 exposent tous la même
variable `DATASET` (`'rsna_sample'` pour la démo du ch1 section A, `'rsna'` pour le dataset
complet). Le chemin de prétraitement n'est écrit qu'à **un seul endroit**
(`course_utils.preprocess_dir`), de sorte que le notebook qui écrit et ceux qui lisent ne
peuvent pas pointer vers deux dossiers différents.

---

## Prérequis (à régler **avant** de construire l'image)

1. **Une machine avec un GPU NVIDIA** + pilotes, `docker`, et le
   `nvidia-container-toolkit` (pour `docker run --gpus all`).
2. **Accès au démon Docker** : votre utilisateur doit pouvoir lancer `docker`
   (membre du groupe `docker`, ou `sudo docker`, ou Docker rootless).
3. **Une clé d'API Kaggle** (pour télécharger le jeu RSNA). Voir
   [Configuration Kaggle](#configuration-kaggle) — c'est la chose à ne pas
   oublier sinon les notebooks de téléchargement échoueront.

Les données RSNA (~300 Go) ne sont **jamais** dans l'image : elles sont
téléchargées par les notebooks dans `data/` **à la racine du dépôt** (ignoré par
git), donc elles persistent entre deux `docker run`. Convention : `data/in/` pour les
**entrées brutes** (téléchargements) et `data/work/` pour les **sorties produites**
(prétraitements, crops, checkpoints).

---

## Démarrage rapide

```bash
# 1. Cloner ce dépôt AVEC ses sous-modules (GMIC + selective-classification)
#    Le dépôt est public : l'URL HTTPS marche partout, sans authentification.
git clone --recurse-submodules https://github.com/Epiconcept-Paris/data-capsule-deep-piste.git
#    (variante SSH, seulement si tu as déjà une clé SSH déclarée sur GitHub —
#     sinon elle échoue sur « Permission denied (publickey) ») :
#    git clone --recurse-submodules git@github.com:Epiconcept-Paris/data-capsule-deep-piste.git
cd data-capsule-deep-piste
# (si déjà cloné sans --recurse-submodules, ou si un sous-module est resté vide :)
git submodule update --init --recursive
# Vérifie que les DEUX sous-modules sont remplis : chaque ligne doit commencer par un
# ESPACE. Un « - » en tête = sous-module vide -> relance la commande ci-dessus.
git submodule status

# 2. Configurer Kaggle (voir section dédiée) — une seule fois
#    => place ton token (access_token) dans le dossier .kaggle/ à la racine du dépôt

# 3. Construire l'image (env GPU + deps via uv ; reprend l'UID/GID de l'utilisateur courant)
./docker-build.sh          # docker build --build-arg HOST_UID=... --build-arg HOST_GID=... -t ...

# 4. Lancer JupyterLab dans le conteneur (GPU + volumes montés)
./docker-run.sh            # écoute sur 127.0.0.1:8888 DANS la VM

# 5. Depuis votre laptop, ouvrir un tunnel SSH vers la VM puis le navigateur
ssh -L 8888:localhost:8888 <votre-alias-vm>
#    puis ouvrir http://localhost:8888 dans le navigateur du laptop
```

Le kernel Jupyter (donc les entraînements) s'exécute **dans le conteneur sur la
VM**, avec le GPU. Le tunnel SSH ne transporte que l'interface web.

> ⚠️ **Ne remplace pas `127.0.0.1:8888:8888` par `8888:8888`.** JupyterLab tourne
> **sans mot de passe ni token** (`--ServerApp.token=` dans le `Dockerfile`) : c'est
> confortable, et sans danger *tant que* le port n'est publié que sur la boucle
> locale de la VM — on y accède alors uniquement par le tunnel SSH, donc seulement
> avec un accès SSH à la VM. Publier le port sur `0.0.0.0` (ou lancer avec
> `--network host`) l'expose au réseau, et **quiconque l'atteint exécute du code
> arbitraire sur le GPU**. Même remarque si la VM est partagée : tout compte ayant
> un shell sur la machine atteint `127.0.0.1:8888`. Dans ce cas, retire
> `--ServerApp.token=` / `--ServerApp.password=` du `Dockerfile` pour que Jupyter
> génère un token aléatoire, et connecte-toi via l'URL complète qu'il affiche.

Une fois dans JupyterLab, commencez par **`notebooks/01_download_data.ipynb`**
pour récupérer le jeu RSNA via la clé Kaggle, puis suivez les chapitres dans
l'ordre.

---

## Configuration Kaggle

Les notebooks téléchargent le jeu RSNA via l'API Kaggle. Les identifiants se placent
**à la racine du dépôt** (jamais dans l'image, et ignorés par git), montés en lecture
seule dans le conteneur.

Le dossier **`.kaggle/` existe déjà dans le dépôt**, avec un gabarit
**`.kaggle/access_token.example`**. Il suffit d'en faire une copie nommée `access_token`
et d'y coller ton token.

1. Sur [kaggle.com](https://www.kaggle.com) → *Settings* → *Create New Token* :
   tu obtiens un token qui commence par **`KGAT_`**.
2. À la racine du dépôt cloné, crée `.kaggle/access_token` depuis le gabarit, puis colle
   ton token dedans (remplace le placeholder, le fichier doit contenir **uniquement** le token) :
   ```bash
   cp .kaggle/access_token.example .kaggle/access_token
   nano .kaggle/access_token       # colle ton token KGAT_... (rien d'autre)
   chmod 600 .kaggle/access_token
   ```
   `access_token` est ignoré par git (jamais committé) ; seul `access_token.example` est versionné.
3. `docker-run.sh` monte `<dépôt>/.kaggle` → `~/.kaggle` dans le conteneur automatiquement,
   et le **notebook 1** lit ce token.

> **Pourquoi `access_token` et pas `kaggle.json` ?** Les tokens Kaggle récents (préfixe
> `KGAT_`) ne fonctionnent qu'avec `kaggle` ≥ 2.x **via `access_token`**. L'ancien
> `kaggle.json` (username + key) renvoie une **401** avec ces tokens. L'image installe
> `kaggle` 2.x ; `kaggle.json` reste accepté en repli si tu as encore une clé à l'ancien format.

---

## Ce que fait l'image Docker

- Base : `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (même stack torch que la VM).
- Installe les dépendances Python des 8 chapitres avec **`uv`** (déclarées dans `pyproject.toml`).
- Crée un utilisateur non-root `deep-piste` **avec l'UID/GID de celui qui build**
  (`--build-arg HOST_UID/HOST_GID`, posés par `docker-build.sh`) : les fichiers écrits
  dans le dépôt monté lui appartiennent, pas à `root`. Le conteneur tourne donc en
  `USER deep-piste` — **jamais root** (si le build est lancé en root, ex. sous WSL,
  l'UID retombe sur `1000` pour garantir un utilisateur non privilégié).
- Démarre JupyterLab sur le port 8888, `root_dir = COURSE_ROOT`.

L'image **ne contient que l'environnement** (ni données, ni sous-modules, ni notebooks).
`docker-run.sh` initialise les sous-modules s'ils manquent, puis fait **un seul montage** —
le dépôt entier — plus les identifiants Kaggle :

| Hôte | Conteneur | Rôle |
|------|-----------|------|
| `<dépôt>` (= `$REPO_DIR`) | `/home/deep-piste/course` (= `COURSE_ROOT`) | tout le cours : `notebooks/`, `modules/GMIC` + `modules/selective-classification`, `data/` |
| `<dépôt>/.kaggle` | `/home/deep-piste/.kaggle` (ro) | token Kaggle (`access_token`, ou `kaggle.json` en repli) |

Comme tout vit sous `COURSE_ROOT`, les notebooks ne référencent **aucun chemin en
dur** : ils passent par `course_utils.course_root()` (qui lit `COURSE_ROOT`), d'où
`modules/GMIC`, `modules/selective-classification` et `data/`.

---

## Permissions : conteneur non-root vs hôte root

Le conteneur tourne **toujours en utilisateur non-root** (`deep-piste`, dont l'UID est
celui de la personne qui a fait le build — ou `1000` si le build est lancé en root).
On ne le lance **jamais** en root, par principe de moindre privilège. En contrepartie,
pour qu'il puisse écrire dans le dépôt monté (`data/`, autosave des notebooks…),
**les fichiers de l'hôte doivent appartenir au même UID** que cet utilisateur.

Si tu **opères en root sur l'hôte** (typique sous WSL), le dépôt cloné appartient à
`root` (UID 0). Le conteneur (UID 1000) ne peut alors pas y écrire — le notebook 1
échoue par exemple sur :

```
PermissionError: [Errno 13] Permission denied: '/home/deep-piste/course/data/in/rsna'
```

### Résoudre

**Le plus simple — opérer en utilisateur non-root.** Clone et builds avec un compte
normal (UID 1000) : son UID correspond à celui du conteneur, tout s'aligne, aucun conflit.

**Si tu dois rester root sur l'hôte —** aligne le propriétaire du dépôt sur l'UID du
conteneur :

```bash
# 1. UID réel du conteneur (normalement 1000)
docker run --rm breast-cancer-course:latest id -u

# 2. Donner le dépôt à cet UID (remplace 1000 par la valeur ci-dessus si différente)
chown -R 1000:1000 /chemin/vers/data-capsule-deep-piste

# 3. git en root sur un dépôt possédé par 1000 -> éviter le warning « dubious ownership »
git config --global --add safe.directory /chemin/vers/data-capsule-deep-piste
```

Puis relance `./docker-run.sh`. ⚠️ Tout fichier recréé **en root** ensuite (`git pull`,
nouveau `git clone`…) repassera à `root:root` et recassera l'écriture → refais le
`chown`, ou — mieux — travaille en utilisateur non-root.

---

## Structure du dépôt

```
data-capsule-deep-piste/
├── README.md              # ce fichier
├── Dockerfile             # image GPU + deps Python via uv (sous-modules montés au runtime)
├── pyproject.toml         # dépendances Python des 8 chapitres (installées via uv)
├── docker-build.sh        # construit l'image
├── docker-run.sh          # lance JupyterLab (GPU + volumes)
├── .gitmodules            # déclare les 2 sous-modules
├── .kaggle/                # token Kaggle : copie access_token.example -> access_token (clé ignorée par git)
├── data/                  # données — vide dans git, contenu ignoré (data/.gitignore)
│   ├── in/                #   entrées brutes téléchargées (RSNA, échantillon, CIFAR)
│   └── work/              #   sorties produites (prétraitements, crops, checkpoints)
├── modules/               # sous-modules git
│   ├── GMIC/              #   → nyukat/GMIC (+ 5 poids pré-entraînés)
│   └── selective-classification/   # → EmilienJemelen/selective-classification
└── notebooks/             # les 8 chapitres + course_utils.py (helpers, dont course_root())
```
