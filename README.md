# Template Cog — SDXL-Turbo sur EKS (unified-inf)

Exemple concret : déployer **[SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)** (Stability AI) sur l’infra EKS unified-inf. Text-to-image en 1 step, GPU, poids dans S3 (pas dans l’image).

## Modèle : SDXL-Turbo

- **Hugging Face** : [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)
- **Usage** : text-to-image, 1 à 4 steps, `guidance_scale=0.0`, sortie 512×512 par défaut.
- **Licence** : [sai-nc-community](https://huggingface.co/stabilityai/sdxl-turbo) ; usage commercial via [Stability AI](https://stability.ai/license).

## Obligations pour l’infra

- **cog.yaml** avec `predict: "predict.py:Predictor"` et `gpu: true` pour SDXL-Turbo.
- **predict.py** : `setup()` charge le pipeline depuis **`/weights`** (EFS mount ou sync S3 par l’init container) ; `predict(prompt, request_id)` génère l’image et renvoie des URIs S3 ou des `Path`. Des logs `[sdxl-turbo]` (setup, dénoising step, pipeline done, save, S3) sont émis sur stdout pour le diagnostic dans les logs du pod.
- **Port 5000** : health-check et `/predictions` pour EKS.
- **Sorties S3** : si `S3_DELIVERY_BUCKET` est défini, les images sont uploadées sous `prefix/MODEL_ID/request_id/` et le predictor retourne une liste d’URIs S3.

## config.json (exemple SDXL-Turbo)

Le template est configuré pour **sdxl-turbo** avec le bucket weights réel :

```json
{
  "model_id": "sdxl-turbo",
  "model_name": "SDXL-Turbo",
  "version": "1.0.0",
  "git_url": "https://github.com/your-org/model-template-cog.git",
  "git_ref": "main",
  "min_replicas": 0,
  "max_replicas": 3,
  "gpu": true,
  "instance_types": ["g4dn.xlarge", "g5.xlarge"],
  "s3_weights_uri": "s3://unified-inf-weights/models/sdxl-turbo/1.0.0/",
  "volume_size_gb": 50,
  "hf_repo_id": "stabilityai/sdxl-turbo",
  "cooldown_period_seconds": 900
}
```

- **hf_repo_id** : utilisé par CodeBuild pour télécharger les weights depuis Hugging Face et les envoyer vers S3.
- **s3_weights_uri** : dérivé par CodeBuild si besoin, ou fixé ici (convention `s3://unified-inf-weights/models/<model_id>/<version>/`).
- **volume_size_gb** : 50 Go suffisent pour SDXL-Turbo (~6 Go de poids) ; min 100 si vous préférez une marge.
- **cooldown_period_seconds** (optionnel) : délai en secondes avant scale-down KEDA (inactivité file SQS). Défaut 900 (15 min) si absent. Borné entre 60 et 86400.

## Weights (image légère)

Les poids **ne sont pas** dans l’image Docker. CodeBuild les télécharge (HF), les envoie vers S3, puis les supprime.

- **Mode S3** (`WEIGHTS_STORAGE=s3`) : au démarrage du pod, un init container fait `aws s3 sync s3_weights_uri → /weights`. Cold start plus lent (sync à chaque démarrage).
- **Mode EFS** (`WEIGHTS_STORAGE=efs`) : un Job Kubernetes sync S3 → EFS une fois par modèle ; les pods montent EFS en `/weights` et un init « wait-for-weights » attend la fin du sync. Cold start plus rapide (lecture directe sur EFS).

## Setup conforme à la model card HF (SDXL-Turbo)

La page Hugging Face recommande le pattern Diffusers suivant pour SDXL-Turbo :

- `AutoPipelineForText2Image`
- `torch_dtype=torch.float16` + `variant="fp16"`
- `num_inference_steps=1` (jusqu’à 4)
- `guidance_scale=0.0`
- taille d’image par défaut **512x512**

Ce repo applique déjà ce pattern dans `predict.py`.

### Versions Python recommandées

Pour limiter les incompatibilités runtime entre `diffusers` et `torch` (ex: erreur `module 'torch' has no attribute 'xpu'`), conserver des versions bornées dans `requirements.txt` et **rebuild l’image** après changement.

### Procédure rapide (prod)

1. Mettre à jour/pousser ce repo (`config.json`, `requirements.txt`, `predict.py`).
2. Relancer `make add-model MODEL=sdxl-turbo GIT_URL=<repo>` pour déclencher un nouveau CodeBuild.
3. Vérifier dans les logs pod que `setup()` termine sans exception.
4. Envoyer une requête test avec `make sqs-send`.

### Si erreur au startup sur `torch`/`diffusers`

- Forcer un rebuild complet de l’image (pas de cache obsolète).
- Vérifier que l’image déployée correspond bien au dernier commit (`image_uri` en DynamoDB / logs CodeBuild).
- Vérifier que le pod utilise bien cette nouvelle image (pas un ancien ReplicaSet).

### Flow automatique

1. **Admin / SQS create** avec `model_id=sdxl-turbo`, `git_url`, `s3_weights_uri`, `volume_size_gb`, etc.
2. **CodeBuild** : clone → `cog build` (sans poids) → télécharge `stabilityai/sdxl-turbo` → upload S3 → met à jour DynamoDB (`s3_weights_uri`, `status=ready`, `image_uri`).
3. **Reconciler** : déploie le pod. En mode S3 : init container sync S3 → `/weights` ; en mode EFS : Job sync S3 → EFS, pod monte EFS en `/weights` et init attend `.synced`. Le container Cog charge depuis `/weights`.

## Déploiement réel

### Prérequis

- Infra déployée : `make deploy-infra` (Lambda, S3 delivery + weights, CodeBuild).
- Repo Git contenant ce template (config.json, cog.yaml, predict.py, requirements.txt) poussé sur `git_url` (ex. GitHub).

### 1. Ajouter le modèle (depuis la racine du repo runtime)

**Seuls le model_id et le git_url sont requis.** GPU, instance types, volume_size_gb, etc. viennent de **config.json** dans le repo : CodeBuild met à jour DynamoDB après le build. Pas besoin de spécifier GPU ou INSTANCE_TYPES à la main.

```bash
make add-model MODEL=sdxl-turbo GIT_URL=https://github.com/lucypaureau/sdxl-turbo.git
```

CodeBuild lit **config.json** (hf_repo_id, gpu, instance_types, volume_size_gb), build l'image Cog, télécharge les weights HF, les envoie vers S3, met à jour DynamoDB (status=ready, image_uri, s3_weights_uri + champs config). Le reconciler déploie le pod : en mode S3 (PVC EBS + init sync), en mode EFS (PVC EFS + Job sync + init wait + container modèle).

Les champs sont synchronisés depuis config.json par CodeBuild dans DynamoDB.

### 2. Lancer une inférence

```bash
make sqs-send MODEL=sdxl-turbo INPUT='{"prompt": "A cinematic shot of a baby racoon wearing an intricate italian priest robe."}'
```

Ou avec un prompt personnalisé :

```bash
make sqs-send MODEL=sdxl-turbo INPUT='{"prompt": "A cat wizard, Gandalf style, detailed fantasy, 8k"}'
```

La réponse (ou l’URI S3 de l’image) est renvoyée sur la queue SQS de réponse (selon la config du worker).

### 3. Test local (sans EKS)

Avec les weights en local (ex. après `huggingface-cli download stabilityai/sdxl-turbo --local-dir ./weights`) et le dossier monté en `/weights` ou en adaptant le chemin dans `predict.py` pour du dev local :

```bash
cog predict -i prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe."
```

## Structure

```
model-template-cog/
├── README.md
├── config.json          # sdxl-turbo, hf_repo_id=stabilityai/sdxl-turbo, s3_weights_uri, volume_size_gb
├── cog.yaml             # gpu: true, mkdir /weights
├── predict.py           # AutoPipelineForText2Image depuis /weights, 1 step, guidance_scale=0.0
├── requirements.txt     # torch, diffusers, transformers, accelerate, boto3
└── upload_weights_to_s3.sh   # optionnel : upload manuel HF → S3
```

## Adapter pour un autre modèle

1. **config.json** : changer `model_id`, `model_name`, `hf_repo_id`, `s3_weights_uri`, `volume_size_gb`, `git_url`.
2. **predict.py** : charger depuis `/weights` (même pattern : `from_pretrained("/weights", ...)`) et adapter la logique d’inférence.
3. **requirements.txt** : dépendances du modèle (diffusers, transformers, etc.).
4. **cog.yaml** : garder `gpu: true` si le modèle utilise le GPU.

## Références

- [SDXL-Turbo — Hugging Face](https://huggingface.co/stabilityai/sdxl-turbo)
- [Diffusers — SDXL Turbo](https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo)
- Worker SQS : `runtime/sqs-worker/worker.py`
- Reconciler : `runtime/lambda-reconciler/handler.py`
- Build : `model-deployer/codebuild/buildspec.yml`
