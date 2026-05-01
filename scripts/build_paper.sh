#!/usr/bin/env bash
# Compile the LaTeX paper. Tries local pdflatex; falls back to Docker
# texlive image if not present.
#
# Output: paper/persona_vectors_icml2026.pdf

set -e
cd "$(dirname "$0")/.."

TEX_FILE="persona_vectors_icml2026"

build_local() {
    cd paper
    echo "=== pdflatex pass 1 ==="
    pdflatex -interaction=nonstopmode "$TEX_FILE.tex"
    if [ -f "$TEX_FILE.aux" ]; then
        echo "=== bibtex ==="
        bibtex "$TEX_FILE" || echo "(bibtex non-fatal warnings)"
        echo "=== pdflatex pass 2 ==="
        pdflatex -interaction=nonstopmode "$TEX_FILE.tex"
        echo "=== pdflatex pass 3 ==="
        pdflatex -interaction=nonstopmode "$TEX_FILE.tex"
    fi
    echo "=== Done: $(pwd)/$TEX_FILE.pdf ==="
}

build_docker() {
    echo "=== building via Docker (texlive/texlive:latest) ==="
    docker run --rm -v "$(pwd)":/work -w /work texlive/texlive:latest \
        bash -c "cd paper && pdflatex -interaction=nonstopmode $TEX_FILE.tex \
                 && bibtex $TEX_FILE \
                 && pdflatex -interaction=nonstopmode $TEX_FILE.tex \
                 && pdflatex -interaction=nonstopmode $TEX_FILE.tex"
    echo "=== Done: paper/$TEX_FILE.pdf ==="
}

if command -v pdflatex >/dev/null 2>&1; then
    build_local
elif command -v docker >/dev/null 2>&1; then
    build_docker
else
    cat <<'MSG'
!! No pdflatex and no docker found.
Options:
  1. Local TeX:    sudo apt install texlive-latex-recommended texlive-publishers texlive-science
  2. Or just upload paper/ + figures/ to overleaf.com and compile there.
MSG
    exit 1
fi
