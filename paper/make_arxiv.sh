#!/usr/bin/env bash
# Assemble a self-contained arXiv submission archive from the final paper sources.
#
# arXiv specifics handled here (per info.arxiv.org/help/submit_tex):
#   * include main.bbl (name matches main.tex) so arXiv need not run bibtex;
#   * include the PDF figure (pdflatex mode accepts PDF/PNG/JPG);
#   * strip all auxiliary/output files (.aux .log .out .fls .fdb_latexmk .pdf);
#   * acmart.cls ships with TeX Live, so it is NOT bundled;
#   * the submission build uses a 'preprint' (non-review) class so there are no
#     referee line numbers.
#
# Usage:  bash paper/make_arxiv.sh
# Output: arxiv/  (staged sources) and arxiv/verisolv-arxiv.tar.gz (upload this).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$REPO/paper"
OUT="$REPO/arxiv"
STAGE="$OUT/src"

rm -rf "$OUT"
mkdir -p "$STAGE"

# 1. Produce the submission .tex: same content, but without the 'review' class
#    option (which adds referee line numbers). Everything else is identical.
sed 's/\[sigplan,nonacm,screen,review\]/[sigplan,nonacm,screen]/' \
    "$PAPER/main.tex" > "$STAGE/main.tex"

# 2. Bring the figure and the bibliography database.
cp "$PAPER/convergence.pdf" "$STAGE/convergence.pdf"
cp "$PAPER/refs.bib"        "$STAGE/refs.bib"

# 2a. Generate the arXiv 00README declaring the build.
cat > "$STAGE/00README.txt" <<'EOF'
00README — arXiv submission for "verisolv"
==========================================

Title:   verisolv: A Unified Artifact Bridging a Numerical ODE/PDE Solver
         and a Machine-Checked Convergence Proof
Authors: Anubhav Prasai <anubhavprasai123@gmail.com>, Himangsu Adhikari <himangsuadk@gmail.com>

Build:   PDFLaTeX (the document class is ACM acmart, sigplan mode, which ships
         with TeX Live and is available on arXiv).

Files in this submission
------------------------
  main.tex          The paper source (acmart, sigplan; preprint mode, no
                    referee line numbers).
  main.bbl          Pre-generated bibliography (matches main.tex). arXiv uses
                    this directly; bibtex need not be run.
  refs.bib          BibTeX source for the references (included for completeness;
                    main.bbl is authoritative).
  convergence.pdf   Figure 2 — empirical error-vs-step convergence plot
                    (vector PDF; embedded via \includegraphics in pdflatex mode).

Notes
-----
* No automated figure-format conversion is required: convergence.pdf is already
  a PDF and is embedded directly.
* The companion software/proof artifact (Python solver, Rust kernel, Lean 4
  proof) is an open-source monorepo; this submission is the paper only.
* Compiles in two PDFLaTeX passes (references resolve from main.bbl).
EOF

# 3. Build once in the staging dir to generate a matching main.bbl, then keep
#    ONLY main.bbl (arateleX uses it directly) and drop every aux/output file.
( cd "$STAGE"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 || {
    echo "ERROR: staged build failed; inspect $STAGE/main.log"; exit 1; }
  # Keep main.bbl; remove everything arXiv regenerates or rejects.
  latexmk -c >/dev/null 2>&1 || true
  rm -f main.pdf main.aux main.log main.out main.fls main.fdb_latexmk \
        main.blg main.synctex.gz texput.log
)

# 4. Sanity: the staged tree must contain exactly the sources arXiv needs.
echo "=== staged submission tree ==="
ls -la "$STAGE"
test -f "$STAGE/main.tex" || { echo "missing main.tex"; exit 1; }
test -f "$STAGE/main.bbl" || { echo "missing main.bbl"; exit 1; }
test -f "$STAGE/convergence.pdf" || { echo "missing convergence.pdf"; exit 1; }

# 5. Clean-room validation: copy the staged sources elsewhere and confirm they
#    compile WITHOUT refs.bib present (i.e. using only the bundled main.bbl),
#    exactly as arXiv will process them.
VALID="$(mktemp -d)"
cp "$STAGE"/*.tex "$STAGE"/*.bbl "$STAGE"/*.pdf "$VALID/"
( cd "$VALID"
  # Two pdflatex passes resolve refs/citations from the bundled .bbl.
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
  if [ ! -f main.pdf ]; then echo "CLEAN-ROOM BUILD FAILED"; exit 1; fi
  undef=$(grep -c "undefined" main.log || true)
  pages=$(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}')
  echo "=== clean-room validation: pages=$pages undefined-refs=$undef ==="
)
rm -rf "$VALID"

# 6. Tarball the staged sources (flat layout — arXiv wants files at the root).
( cd "$STAGE" && tar czf "$OUT/verisolv-arxiv.tar.gz" ./* )
echo "=== arXiv archive ready ==="
ls -la "$OUT/verisolv-arxiv.tar.gz"
tar tzf "$OUT/verisolv-arxiv.tar.gz"
