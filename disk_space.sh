sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=400G tmpfs /mnt/ramdisk
df -h /mnt/ramdisk

# 2.1 Python’s tempfiles (used by hug huggingface_hub)
export TMPDIR=/mnt/ramdisk/tmp
mkdir -p $TMPDIR

# 2.2 Pip’s cache & temp (if you ever pip‑install large wheels here)
export PIP_CACHE_DIR=/mnt/ramdisk/pip-cache

# 2.3 HuggingFace cache roots
export XDG_CACHE_HOME=/mnt/ramdisk/cache        # moves ~/.cache/*
export HF_HOME=/mnt/ramdisk/cache/huggingface  # moves ~/.cache/huggingface/hub

# 2.4 (Optional) Datasets/metrics/transforms‑specific caches
export HF_DATASETS_CACHE=/mnt/ramdisk/cache/datasets
export HF_METRICS_CACHE=/mnt/ramdisk/cache/metrics
export TRANSFORMERS_CACHE=/mnt/ramdisk/cache/transformers

mkdir -p \
  /mnt/ramdisk/{tmp,pip-cache,cache/{huggingface,datasets,metrics,transformers}}

# Copy your project into RAM
cp -r ~/orpheus-1b /mnt/ramdisk/orpheus-1b
cd /mnt/ramdisk/orpheus-1b

# Activate your venv (you can also recreate it here so that the venv itself lives in RAM)
source ~/ramtmp/venv/bin/activate  # or python3 -m venv venv; source venv/bin/activate

# Now run:
python train_data.py

