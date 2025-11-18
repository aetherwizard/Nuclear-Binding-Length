% File: qadi-article.cls
% Date: 2025-10-29
% QADI-branded research article class (ledger-first; no SI in main text)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{qadi-article}[2025/10/29 v1.0 QADI article class]

% ---------- Options ----------
\newif\ifqadi@cover
\qadi@covertrue              % default: cover page + standard title page
\DeclareOption{coverpage}{\qadi@covertrue}
\DeclareOption{compact}{\qadi@coverfalse} % single, elegant title page
\DeclareOption{draft}{\PassOptionsToClass{draft}{article}}
\DeclareOption{final}{\PassOptionsToClass{final}{article}}
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{article}}
\DeclareOption{twoside}{\PassOptionsToClass{twoside}{article}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% ---------- Base class ----------
\LoadClass[11pt]{article}

% ---------- Core packages (no SI packages) ----------
\RequirePackage[margin=1in]{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{microtype}
\RequirePackage{amsmath,amssymb,mathtools,bm}
\RequirePackage{graphicx}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[authoryear,round]{natbib}
\RequirePackage{enumitem}
\RequirePackage{booktabs,array,tabularx,longtable}
\RequirePackage{titling}
\RequirePackage{etoolbox}

% ---------- Sensible defaults ----------
\setlength{\parskip}{4pt}
\setlength{\parindent}{12pt}
\graphicspath{{./}{Zimg/}}

% ---------- QADI identity variables (user-overridable) ----------
% Set via \renewcommand in the document preamble as needed.
\newcommand{\QADIInstitution}{\textbf{Quantum AetherDynamics Institute (QADI)}, Alma, Illinois, USA}
\newcommand{\QADIROR}{} % e.g., \renewcommand{\QADIROR}{\href{https://ror.org/XXXXXXXXX}{https://ror.org/XXXXXXXXX}}
\newcommand{\QADILogo}{qadi_logo_small.png} % logo file expected in project root
\newcommand{\QADIVersion}{Version 1.0}
\newcommand{\QADIDate}{November 17, 2025}

% -------- Core symbols --------
\newcommand{\ePart}{\mathrm e}                   % electron symbol (text-safe)
\DeclareRobustCommand{\me}{\ensuremath{m_{\ePart}}}

% Tight-subscript base units
\DeclareRobustCommand{\Fq}{\ensuremath{{F_q}}}
\DeclareRobustCommand{\lC}{\ensuremath{{\lambda_C}}}

% Aether / field ledger primitives
\DeclareRobustCommand{\curlapm}{\ensuremath{\mathrm{curl}}}
\DeclareRobustCommand{\Au}{\ensuremath{{A_u}}}
\DeclareRobustCommand{\kC}{\ensuremath{{k_C}}}
\DeclareRobustCommand{\ma}{\ensuremath{{m_a}}}
\DeclareRobustCommand{\Gforce}{\ensuremath{\mathrm{Gforce}}}

% Charges (square-charge convention)
\DeclareRobustCommand{\eesq}{\ensuremath{{e^2}}}
\DeclareRobustCommand{\eemaxsq}{\ensuremath{{e_\mathrm{emax}}^{2}}}
\DeclareRobustCommand{\ea}{\ensuremath{{e_a}^{2}}}

% Other named QMU units / symbols
\DeclareRobustCommand{\volm}{\ensuremath{\mathrm{volm}}}
\DeclareRobustCommand{\rson}{\ensuremath{\mathrm{rson}}}
\DeclareRobustCommand{\mflx}{\ensuremath{\mathrm{mflx}}}
\DeclareRobustCommand{\alphafs}{\ensuremath{\alpha}}
\DeclareRobustCommand{\PhiGR}{\ensuremath{\Phi}}
\DeclareRobustCommand{\Hbase}{\ensuremath{\frac{\pi}{3\sqrt{2}}}}

% Canonical constants / ratios
\DeclareRobustCommand{\AGCratio}{\ensuremath{16\pi^{2}}}

% -------- Sets and identities (ready-to-drop ledger snippets) --------
\DeclareRobustCommand{\BaseSet}{\ensuremath{\{\me,\lC,\Fq,\eesq,\eemaxsq\}}}
% ---------- Affiliation helpers ----------
% \QADIAffiliation[<RORcode>]  (optional arg not required if \QADIROR set)
\newcommand{\QADIAffiliation}[1][]{%
  \QADIInstitution%
  \ifx\QADIROR\@empty
    \ifx#1\@empty\relax\else\\ ROR: \href{https://ror.org/#1}{https://ror.org/#1}\fi
  \else\\ ROR: \QADIROR
  \fi}

% \QADIAuthor{Name}{ORCID}
\newcommand{\QADIAuthor}[2]{%
  #1\\
  \QADIAffiliation\\
  \href{https://orcid.org/#2}{ORCID: #2}%
}

% Optional keywords block
\newcommand{\QADIKeywords}[1]{%
  \par\smallskip\noindent\textbf{Keywords:} #1\par
}

% ---------- Compact title styling (used when \ifqadi@cover = false) ----------
\newcommand{\qadi@compacttitle}{%
  \pretitle{%
    \begin{center}\vspace*{-10pt}%
      \includegraphics[width=1.15in]{\QADILogo}\par\vspace{8mm}\LARGE
    }
  \posttitle{\par\end{center}\vspace{2mm}}
}

% ---------- Cover page (used when \ifqadi@cover = true) ----------
\newcommand{\qadi@coverpage}{%
  \begin{titlepage}
  \centering
  \vspace*{10mm}
  \includegraphics[width=1.7in]{\QADILogo}

  \vspace{10mm}
  {\Large\bfseries Quantum AetherDynamics Institute (QADI)\par}
  \vspace{6mm}
  {\large Alma, Illinois, USA\par}

  \vspace{20mm}
  {\huge\bfseries \thetitle\par}

  \vspace{12mm}
  {\large\bfseries \QADIVersion{} — \QADIDate\par}

  \vspace{22mm}
  {\small QADI is a private non-profit 501(c)(3) research organization advancing the Aether Physics Model (APM)
  and Quantum Measurement Units (QMU) through open, falsifiable science and fully reproducible metrology.
  All publications, data, and code are archived openly with DOIs.\par}

  \vfill
  {\small This document is published ledger-first (QMU).}
  \end{titlepage}
}

% ---------- Public title command ----------
% Use \QADIMakeTitle (instead of \maketitle) in your document.
\newcommand{\QADIMakeTitle}{%
  \ifqadi@cover
    \qadi@coverpage
    \maketitle
  \else
    \qadi@compacttitle
    \maketitle
  \fi
}

% ---------- Bibliography defaults ----------
% Users can override as needed in their document.
\AtBeginDocument{%
  \providecommand{\bibliographystyle}[1]{}%
}

% ---------- Ledger-first guardrails (comments for authors) ----------
% A) Use only QMU ledger units in derivations: {me, λC, Fq, e^2, e_emax^2, Au, curl, volm, rson, mflx}
% B) Do NOT introduce SI units in the main text; if needed, put translation in an Appendix.
% C) Check dimensional closure with: \LedgerOne.
% D) Avoid permittivity/permeability; use distributed-charge QMU constructs.

% Keep ledger patch at end of file
\RequirePackage{qadi_ledger_patch}
% End of class