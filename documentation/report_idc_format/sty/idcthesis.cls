%% $Id$
\NeedsTeXFormat{LaTeX2e}
% location of included style files
\newcommand*{\stypath}{sty}

\ProvidesClass{\stypath/idcthesis}[2011/03/17 LaTeX2e Klasse fuer Arbeiten am IDC]

\RequirePackage{xkeyval}

\newif{\iftwosided}
\DeclareOptionX{twoside}{\twosidedtrue}

\newlength{\@bcor}\setlength{\@bcor}{0mm}
\DeclareOptionX{BCOR}[0mm]{\setlength{\@bcor}{#1}}

\define@boolkey{idcthesis.cls}[cls]{onehalfspacing}[]{}
\define@boolkey{idcthesis.cls}[cls]{fixfloatplacement}[]{}

\newcommand*{\linkcolor}{black}
\define@key{idcthesis.cls}{linkcolor}{\renewcommand*{\linkcolor}{#1}}

\global\def\@lang{}

\newif{\ifgermanlang}

\define@choicekey*+{idcthesis.cls}{language}[\val\nr]{de,en}{%
  \ifcase\nr\relax
     \germanlangtrue
     \global\def\@lang{german}
  \or
     \germanlangfalse
     \global\def\@lang{english}
  \fi
}{\PackageError{idcthesis.cls}{Invalid language '#1'}}

\ProcessOptionsX


\LoadClass[12pt,final,openright,chapterprefix]{scrreprt}
%% ----------------------------------------------------------------------------
\RequirePackage[automark,plainheadtopline,plainheadsepline]{scrpage2}
\defpagestyle{idcplain}{{\pagemark\hfill}{\hfill\pagemark}{\hfill\pagemark}}{{}{}{}}

\KOMAoptions{paper=a4,cleardoublepage=idcplain,index=totoc,bibliography=totoc,listof=totoc,headinclude=false,footinclude=false}
\setlength{\headheight}{1.5\baselineskip}

\newcommand*{\IDCThesisOptions}[1]{\setkeys{idcthesis.cls}{#1}}

%% language selection ---------------------------------------------------------
\RequirePackage[english,german]{babel}
\AtBeginDocument{\expandafter\selectlanguage\expandafter{\@lang}}

\addto\extrasgerman{%
  \def\titlename{Titel}%
  \def\abbreviationsname{Abk\"{u}rzungen}%
  \def\symbolsname{Formelzeichen}%
  \def\operatorsname{Operatoren}%
  \def\inputdecl{\input{decl_de}}%
}
\addto\extrasenglish{%
  \def\titlename{Title}%
  \def\abbreviationsname{Abbreviations}%
  \def\symbolsname{Symbols}%
  \def\operatorsname{Operators}%
  \def\inputdecl{\input{decl_en}}%
}
%% ----------------------------------------------------------------------------

%% load fonts -----------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern} % only use lmsans font
\RequirePackage[bitstream-charter]{mathdesign}
\RequirePackage{pifont}
\RequirePackage[scaled=0.9]{couriers}
%% ----------------------------------------------------------------------------

%% switch between one/two sided output ----------------------------------------
\newcommand{\@geometry@options}{%
    paper=a4paper,
    top=5\baselineskip,
    outer=30mm,
    marginparwidth=23mm,
    hmarginratio=1:1,
    vmarginratio=3:4
}
\iftwosided%
  \let\@geometry@options@orig\@geometry@options%
    \renewcommand{\@geometry@options}{%
      \@geometry@options@orig,
      twoside,
      outer=30mm,
      marginparwidth=23mm,
      bindingoffset=\@bcor,
      hmarginratio=1:2
    }
\fi%
\RequirePackage[\@geometry@options]{geometry}%
%% ----------------------------------------------------------------------------

%% for changing the line spread -----------------------------------------------
\RequirePackage{setspace}
\AtBeginDocument{%
    \ifclsonehalfspacing%
        \onehalfspacing%
    \fi%
}
%% ----------------------------------------------------------------------------

%% page headers ---------------------------------------------------------------
\clearscrheadfoot
\renewcommand*{\chapterpagestyle}{idcplain}
\renewcommand*{\partpagestyle}{idcplain}
\setheadwidth{textwithmarginpar}
\setheadsepline{.4pt}[\color{black}]
\ohead[\pagemark]{\pagemark}
\ihead[\headmark]{\headmark}
\pagestyle{scrheadings}
%% ----------------------------------------------------------------------------

%% change appearance of section headers ---------------------------------------
\addtokomafont{section}{\rmfamily}
\addtokomafont{subsection}{\rmfamily}
\addtokomafont{subsubsection}{\rmfamily}
\addtokomafont{paragraph}{\rmfamily}
\addtokomafont{subparagraph}{\rmfamily}
%% ----------------------------------------------------------------------------

%% toc shows three levels -----------------------------------------------------
\setcounter{secnumdepth}{3} \setcounter{tocdepth}{3}
%% ----------------------------------------------------------------------------

%% make the toc appear in the pdf navigation ----------------------------------
\renewcommand*{\toc@heading}{%
  \addchap*{\contentsname}
  \addcontentsline{toc}{chapter}{\contentsname}
  \@mkboth{\contentsname}{\contentsname}
}
%% ----------------------------------------------------------------------------

%% automatic handling of acronyms ---------------------------------------------
\usepackage[printonlyused]{acronym}

%% modify appearance if the acronym list
\def\rmlabel#1{#1\hfill}
\renewenvironment{AC@deflist}[1]%
        {\ifAC@nolist%
         \else%
            \raggedright\begin{list}{}%
                {\setlength{\itemsep}{-\parsep}
                \settowidth{\labelwidth}{{#1}}%
                \setlength{\leftmargin}{\labelwidth}%
                \addtolength{\leftmargin}{\labelsep}%
                \let\makelabel=\rmlabel}%
          \fi}%
        {\ifAC@nolist%
         \else%
            \end{list}%
         \fi}%
%% ----------------------------------------------------------------------------

%% a modified list environment to create the symbol and operator lists --------
\newcommand{\sym}[2]{\item[\makelabel{#1}]{#2}}%
\newenvironment{symbollist}[1]%
  {%
    \raggedright%
    \begin{list}{}%
    {%
      \setlength{\itemsep}{-\parsep}%
      \settowidth{\labelwidth}{#1}%
      \setlength{\leftmargin}{\labelwidth}%
      \addtolength{\leftmargin}{\labelsep}%
      \let\makelabel=\rmlabel%
    }%
  }%
  {\end{list}}%
%% ----------------------------------------------------------------------------

%% never without ... ----------------------------------------------------------
%\RequirePackage{fixltx2e}
\RequirePackage[
  plainpages=false,
  bookmarksopen=false,
  bookmarksnumbered=true,
  linktocpage=true,
  colorlinks=true,
  unicode=true,
  hyperfootnotes=false,
  breaklinks=true,
]{hyperref}
%% ----------------------------------------------------------------------------

\RequirePackage{\stypath/todo-o-matic}

%% Math -----------------------------------------------------------------------
\RequirePackage[centertags]{amsmath}
\RequirePackage{amsthm,amsbsy}
%%-----------------------------------------------------------------------------


%% nice referencing -----------------------------------------------------------
% this MUST be after amsmath! otherwise ends in a severe fuck-up...
\RequirePackage[english,german,capitalize]{cleveref}
\cref@addto\extrasgerman{%
  \crefname{figure}{Abb.}{Abb.}
  \crefname{equation}{Gl.}{Gl.}
  \crefname{table}{Tab.}{Tab.}
}
% tables are not abbreviated by default
\if@cref@capitalise
  \cref@addto\extrasenglish{%
    \crefname{table}{Tab.}{Tab.}
  }
\else
  \cref@addto\extrasenglish{%
    \crefname{table}{tab.}{tab.}
  }
\fi

%% ----------------------------------------------------------------------------


%% Graphics -------------------------------------------------------------------
\RequirePackage{tikz}
\RequirePackage{subfig}      % subfigures in one figure: a,b,c...
\RequirePackage{psfrag}      % Textsubstitution in eps/ps figures
%% ----------------------------------------------------------------------------

%% Miscellaneous --------------------------------------------------------------
\setlength{\unitlength}{1cm}
\RequirePackage{longtable}
\RequirePackage{booktabs}
\RequirePackage[alsoload=binary,loctolang={UK:english,DE:ngerman}]{siunitx}

\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\vbadness=10000
\vfuzz=3pt
%% ----------------------------------------------------------------------------

%% ----------------------------------------------------------------------------
%% Alter some LaTeX defaults for better treatment of figures:
%% See p.105 of "TeX Unbound" for suggested values.
%% See pp. 199-200 of Lamport's "LaTeX" book for details.
\AtBeginDocument{
  \ifclsfixfloatplacement
    % General parameters, for ALL pages:
    \renewcommand{\topfraction}{0.9}	% max fraction of floats at top
    \renewcommand{\bottomfraction}{0.8}	% max fraction of floats at bottom
    %  Parameters for TEXT pages (not float pages):
    \setcounter{topnumber}{2}
    \setcounter{bottomnumber}{2}
    \setcounter{totalnumber}{4}     % 2 may work better
    \setcounter{dbltopnumber}{2}    % for 2-column pages
    \renewcommand{\dbltopfraction}{0.9}	% fit big float above 2-col. text
    \renewcommand{\textfraction}{0.07}	% allow minimal text w. figs
    %   Parameters for FLOAT pages (not text pages):
    \renewcommand{\floatpagefraction}{0.7}	% require fuller float pages
    % N.B.: floatpagefraction MUST be less than topfraction !!
    \renewcommand{\dblfloatpagefraction}{0.7}	% require fuller float pages
    % remember to use [htp] or [htpb] for placement
  \fi
}
%% ----------------------------------------------------------------------------------

\newcommand*{\setupthesis}[5]{%
  \AtBeginDocument{%
    \newcommand*{\Thesis}{#1}
    \newcommand*{\Title}{#2}
    \newcommand*{\Author}{#3}
    \newcommand*{\Subject}{#4}
    \newcommand*{\Date}{#5}
    \title{\Title}
    \author{\Author}
    \subject{\Subject}
    \date{\Date}
    \hypersetup{
      pdftitle   = {\Title\ (\Thesis)},
      pdfsubject = {\Subject},
      pdfauthor  = {\Author},
      linkcolor=\linkcolor,
      citecolor=\linkcolor,
      filecolor=\linkcolor,
      menucolor=\linkcolor,
      urlcolor=\linkcolor
    }
  }
}
