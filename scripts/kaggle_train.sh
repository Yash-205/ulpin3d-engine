#!/usr/bin/env bash
# One-command GPU training on Kaggle (free P100/T4, ~30 GPU-hours a week).
#
#   1. kaggle.com -> Settings -> API -> "Create New Token"  -> saves kaggle.json
#   2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   3. Verify your phone number on Kaggle (needed for GPU + internet in notebooks)
#   4. bash scripts/kaggle_train.sh push      # uploads the code + starts training
#      bash scripts/kaggle_train.sh status    # RUNNING / COMPLETE / ERROR
#      bash scripts/kaggle_train.sh fetch     # downloads trained models into models/kaggle/
#
# Full-length training that reuses the chips the first notebook prepared (no 6 h of data prep):
#      bash scripts/kaggle_train.sh push-full
#      KERNEL=ulpin3d-train-full bash scripts/kaggle_train.sh status   (or fetch)
#
# QUICK=1 bash scripts/kaggle_train.sh push   -> ~1 h smoke run instead of the full ~6-8 h
set -euo pipefail
cd "$(dirname "$0")/.."
command -v kaggle >/dev/null || pip install -q kaggle
USER_NAME=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
DS="$USER_NAME/ulpin3d-engine"; PREP="$USER_NAME/ulpin3d-train"; KN="$USER_NAME/${KERNEL:-ulpin3d-train}"
case "${1:-push}" in
  push)
    rm -rf dist/kaggle && mkdir -p dist/kaggle/dataset dist/kaggle/kernel
    zip -qr dist/kaggle/dataset/ulpin3d-engine.zip . -x "data/*" "runs/*" "dist/*" "examples/*" "models/kaggle*" "*.egg-info/*" "*__pycache__*" ".git/*"
    cat > dist/kaggle/dataset/dataset-metadata.json <<JSON
{"title": "ulpin3d-engine", "id": "$DS", "licenses": [{"name": "other"}]}
JSON
    if kaggle datasets status "$DS" >/dev/null 2>&1; then
      kaggle datasets version -p dist/kaggle/dataset -m "update $(date +%F_%H%M)" --dir-mode zip
    else
      kaggle datasets create -p dist/kaggle/dataset --dir-mode zip
    fi
    sed "s/\"QUICK = False/\"QUICK = $([ "${QUICK:-0}" = 1 ] && echo True || echo False)/" notebooks/train_kaggle.ipynb > dist/kaggle/kernel/train_kaggle.ipynb
    cat > dist/kaggle/kernel/kernel-metadata.json <<JSON
{"id": "$KN", "title": "ulpin3d-train", "code_file": "train_kaggle.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$DS"], "competition_sources": [], "kernel_sources": []}
JSON
    echo "waiting 60 s for the dataset to finish processing..."; sleep 60
    kaggle kernels push -p dist/kaggle/kernel
    echo "Started. Follow it at https://www.kaggle.com/code/$KN  (bash scripts/kaggle_train.sh status)";;
  push-full)
    FULL="$USER_NAME/ulpin3d-train-full"
    rm -rf dist/kaggle && mkdir -p dist/kaggle/dataset dist/kaggle/kernel
    zip -qr dist/kaggle/dataset/ulpin3d-engine.zip . -x "data/*" "runs/*" "dist/*" "examples/*" "models/kaggle*" "*.egg-info/*" "*__pycache__*" ".git/*"
    echo '{"title": "ulpin3d-engine", "id": "'"$DS"'", "licenses": [{"name": "other"}]}' > dist/kaggle/dataset/dataset-metadata.json
    kaggle datasets version -p dist/kaggle/dataset -m "update $(date +%F_%H%M)" --dir-mode zip
    cp notebooks/train_kaggle_full.ipynb dist/kaggle/kernel/
    cat > dist/kaggle/kernel/kernel-metadata.json <<JSON
{"id": "$FULL", "title": "ulpin3d-train-full", "code_file": "train_kaggle_full.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$DS"], "competition_sources": [], "kernel_sources": ["$PREP"]}
JSON
    echo "waiting 90 s for the dataset to finish processing..."; sleep 90
    kaggle kernels push -p dist/kaggle/kernel
    echo "Started. Follow it at https://www.kaggle.com/code/$FULL  (KERNEL=ulpin3d-train-full bash scripts/kaggle_train.sh status)";;
  status) kaggle kernels status "$KN";;
  fetch)  mkdir -p models/kaggle && kaggle kernels output "$KN${VERSION:+/$VERSION}" -p models/kaggle --file-pattern '^models/' --page-size 200 && ls -la models/kaggle;;
  *) echo "usage: $0 push|push-full|status|fetch"; exit 1;;
esac
