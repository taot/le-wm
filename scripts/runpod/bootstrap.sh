set -e

START_TIME=$SECONDS

log() {
    echo
    echo "[$(date '+%H:%M:%S')] [local] $*"
}

log "Copying secrets to runpod"
scp -r ~/.runpod/secrets runpod:/root/

log "Running setup on runpod"
ssh runpod bash -s <<'EOF'
set -e
log() {
    echo
    echo "[$(date '+%H:%M:%S')] [runpod] $*"
}

log "Installing apt packages"
apt update
apt install gh

cd ~
if [ ! -d le-wm ]; then
    log "Cloning le-wm"
    git clone https://github.com/taot/le-wm.git
else
    log "le-wm already exists, skipping clone"
fi
grep -qxF 'source ~/le-wm/.env' ~/.bashrc || echo 'source ~/le-wm/.env' >> ~/.bashrc

cd ~/le-wm
log "Running uv sync"
source .env-runpod
uv sync

log "Logging in to GitHub"
gh auth login --with-token < ~/secrets/github && gh auth setup-git && gh auth status
log "Logging in to Hugging Face"
uv run hf auth login --token "$(<~/secrets/huggingface)"
log "Logging in to wandb"
uv run wandb login "$(<~/secrets/wandb)"

log "Downloading dataset"
source .env-runpod && uv run hf download librakevin/lewm-pusht --repo-type dataset --local-dir $STABLEWM_HOME/datasets/librakevin--lewm-pusht

log "Downloading checkpoints"
source .env-runpod && uv run hf buckets sync hf://buckets/librakevin/lewm-checkpoints $STABLEWM_HOME/checkpoints

log "Setup done"
EOF

ELAPSED=$((SECONDS - START_TIME))
log "Done in $((ELAPSED / 60))m $((ELAPSED % 60))s"
