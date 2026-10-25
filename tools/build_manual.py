"""Build docs/manual.pdf from the maintained Markdown sources."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='main', help='Git revision used in PDF repository links')
    options = parser.parse_args()
    try:
        import pypandoc
    except ImportError as exc:
        raise SystemExit('Install the docs extra: python -m pip install ".[docs]"') from exc
    if shutil.which('pdflatex') is None:
        raise SystemExit('pdflatex is required; install a LaTeX distribution.')
    root = Path(__file__).resolve().parents[1]
    sources = [root / 'README.md'] + [root / 'docs' / name for name in (
        'methods.md', 'tutorial.md', 'api.md', 'development.md')]
    with tempfile.TemporaryDirectory(prefix='kernelmix-manual-') as folder:
        build = Path(folder)
        parts = []
        for source in sources:
            text = source.read_text(encoding='utf-8')
            # Resolve repository links for readers of the standalone PDF.
            def link(match):
                target = match.group(1)
                if '://' in target or target.startswith('#'):
                    return match.group(0)
                resolved = (source.parent / target).resolve().relative_to(root)
                return f'](https://github.com/nilson01/kernelmix/blob/{options.revision}/{resolved.as_posix()})'
            text = re.sub(r'\]\(([^)]+)\)', link, text)
            # Use the local original image when compiling.
            text = text.replace(
                f'https://github.com/nilson01/kernelmix/blob/{options.revision}/docs/figures/response.png',
                str(root / 'docs/figures/response.pdf'))
            text = text.replace(
                f'https://github.com/nilson01/kernelmix/blob/{options.revision}/docs/figures/count-survival.png',
                str(root / 'docs/figures/count-survival.pdf'))
            parts.append(text)
        markdown = build / 'manual.md'
        markdown.write_text('\n\n'.join(parts), encoding='utf-8')
        header = build / 'header.tex'
        header.write_text(r'''\usepackage{fvextra}
\usepackage{needspace}
\usepackage{etoolbox}
\BeforeBeginEnvironment{Shaded}{\Needspace{8\baselineskip}}
\AddToHook{cmd/section/before}{\Needspace{16\baselineskip}}
\AddToHook{cmd/subsection/before}{\Needspace{12\baselineskip}}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},breaklines=true,breakanywhere=true,samepage=true,fontsize=\small}
\AtBeginDocument{\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=blue}}
''', encoding='utf-8')
        command = [pypandoc.get_pandoc_path(), str(markdown), '--standalone',
                   '--toc', '--toc-depth=2', '--number-sections', '--pdf-engine=pdflatex',
                   '-V', 'geometry:margin=0.85in', '-V', 'fontsize=10pt',
                   '-M', 'title:KernelMix reference manual', '-M', 'author:Nilson Chapagain',
                   '-M', 'date:', '-H', str(header), '-o', str(root / 'docs/manual.pdf')]
        subprocess.run(command, cwd=build, check=True)
    print(root / 'docs/manual.pdf')


if __name__ == '__main__':
    main()
