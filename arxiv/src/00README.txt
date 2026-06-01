00README — arXiv submission for "verisolv"
==========================================

Title:   verisolv: A Unified Artifact Bridging a Numerical ODE/PDE Solver
         and a Machine-Checked Convergence Proof
Author:  Anubhav Prasai <anubhavprasai123@gmail.com>

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
