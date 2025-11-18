# QADI Paper Source Package

This archive contains the LaTeX source for the paper:

**TITLE GOES HERE**

- Institute: **Quantum AetherDynamics Institute (QADI), Alma, Illinois, USA**
- Author: **David W. Thomson III** (ORCID: 0000-0002-5830-5427)
- Ledger: **QMU (Quantum Measurement Units)** — no SI in the main text
- Class: `qadi-article.cls` (cover page by default)

## Files
- `main.tex` — paper source (ledger-first)
- `qadi-article.cls` — QADI-branded article class
- `qadi_macros.tex` — QMU/APM minimal macro set
- `qadi_logo_small.png` — logo used on the cover/title page
- `qadi_refs.bib` — includes the institutional identity citation
- `LICENSE.txt` — license (CC BY 4.0)
- `README.md` — this file
- (optional) `latexmkrc`, `Makefile`, `.gitignore`, `CHANGELOG.md`, `CITATION.cff`

## Compile
Use `latexmk` (recommended):
```bash
latexmk -pdf main.tex
