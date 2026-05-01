#!/usr/bin/env bash
# One-shot setup: initialise git, create initial commit, push to GitHub.
# Call ONCE, then use normal git/gh workflows.
#
# Prereqs:
#   - gh CLI installed (already done on this machine)
#   - HF token not in any tracked file (verified clean)
#
# Usage:
#   bash scripts/init_repo.sh                             # public repo
#   VISIBILITY=private bash scripts/init_repo.sh
#   REPO_NAME=my-custom-name bash scripts/init_repo.sh

set -e
cd "$(dirname "$0")/.."

REPO_NAME="${REPO_NAME:-persona-vectors-icml2026}"
VISIBILITY="${VISIBILITY:-public}"

echo "=== Init git repo ==="
if [ ! -d .git ]; then
    git init -b main
fi

# Pull commit identity from gh if not already set globally.
if [ -z "$(git config user.name 2>/dev/null)" ]; then
    NAME="${GIT_NAME:-$(gh api user -q .name 2>/dev/null)}"
    [ -z "$NAME" ] && NAME=$(gh api user -q .login 2>/dev/null)
    git config user.name "$NAME"
fi
if [ -z "$(git config user.email 2>/dev/null)" ]; then
    EMAIL="${GIT_EMAIL:-$(gh api user -q .email 2>/dev/null)}"
    if [ -z "$EMAIL" ] || [ "$EMAIL" = "null" ]; then
        UID_NUM=$(gh api user -q .id 2>/dev/null)
        ULOGIN=$(gh api user -q .login 2>/dev/null)
        EMAIL="${UID_NUM}+${ULOGIN}@users.noreply.github.com"
    fi
    git config user.email "$EMAIL"
fi
echo "  using: $(git config user.name) <$(git config user.email)>"

echo "=== Verify no secrets ==="
if git ls-files --others --exclude-standard --modified --cached 2>/dev/null \
    | xargs grep -l -E "hf_[a-zA-Z0-9]{30,}" 2>/dev/null; then
    echo "!! Found suspected HF token in tracked files. Aborting."
    exit 1
fi

echo "=== Stage and commit ==="
git add .
git commit -m "Initial commit: persona-vectors-icml2026

Per-user persona vectors for LLM personalization (ICML 2026 mech-interp workshop).

- src/: PersonaVectors / PersonaSteering / FactExtractor + LaMPDataset + metrics
- experiments/: layer_search, geometry_analysis, alpha_sweep, n_questions,
                full_run, positive_control, case_study
- results/: layer_search, geometry, alpha_sweep, n_questions, full_run,
            positive_control, case_study, main_table (smoke n=200 reference)
- figures/: 14 PDFs (paper figures + per-model geometry plots)
- paper/: 5-page LaTeX skeleton + bibliography + 3 auto-generated tables
- scripts/: env.sh, generate_paper_figures.py, run launchers, this init script
"

echo "=== Push to GitHub ==="
if ! gh auth status >/dev/null 2>&1; then
    echo "!! Not logged in to GitHub. Run:  gh auth login   (interactive)"
    exit 1
fi

if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
    echo "Repo $REPO_NAME exists; setting remote and pushing."
    OWNER=$(gh api user -q .login)
    git remote add origin "git@github.com:$OWNER/$REPO_NAME.git" 2>/dev/null || true
    git push -u origin main
else
    gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
fi

echo ""
echo "Done. Repo: $(gh repo view --json url -q .url)"
