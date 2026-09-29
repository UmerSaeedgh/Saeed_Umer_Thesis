"""
Structural sanity checker for the thesis LaTeX sources.

There is no LaTeX toolchain on this machine (the document is compiled on Overleaf), so this
script catches the errors that would otherwise only surface at compile time: unbalanced braces,
unclosed environments, duplicate labels, and references pointing at labels that do not exist.

Run: python check_tex.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
FILES = ['title.tex', 'consent.tex'] + [f'content/{n}.tex' for n in
         ('abstract', 'intro', 'literature', 'methodology',
          'implementation', 'result', 'discussion', 'conclusion')]

problems = []
labels, refs, cites = {}, [], []


def strip_comments(text):
    out = []
    for line in text.split('\n'):
        idx = None
        for m in re.finditer(r'%', line):
            if m.start() == 0 or line[m.start() - 1] != '\\':
                idx = m.start()
                break
        out.append(line if idx is None else line[:idx])
    return '\n'.join(out)


for rel in FILES:
    p = ROOT / rel
    if not p.exists():
        problems.append(f'{rel}: MISSING')
        continue
    raw = p.read_text(encoding='utf-8')
    src = strip_comments(raw)

    # --- brace balance ---
    depth = 0
    for ch_i, ch in enumerate(src):
        if ch == '{' and (ch_i == 0 or src[ch_i - 1] != '\\'):
            depth += 1
        elif ch == '}' and (ch_i == 0 or src[ch_i - 1] != '\\'):
            depth -= 1
            if depth < 0:
                problems.append(f'{rel}: unmatched closing brace')
                depth = 0
    if depth != 0:
        problems.append(f'{rel}: {depth} unclosed brace(s)')

    # --- environment balance ---
    stack = []
    for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', src):
        kind, env = m.group(1), m.group(2)
        if kind == 'begin':
            stack.append(env)
        else:
            if not stack:
                problems.append(f'{rel}: \\end{{{env}}} with no matching \\begin')
            elif stack[-1] != env:
                problems.append(f'{rel}: \\end{{{env}}} closes \\begin{{{stack[-1]}}}')
                stack.pop()
            else:
                stack.pop()
    for env in stack:
        problems.append(f'{rel}: unclosed environment {env}')

    # --- labels / refs / cites ---
    for m in re.finditer(r'\\label\{([^}]+)\}', src):
        lb = m.group(1)
        if lb in labels:
            problems.append(f'{rel}: DUPLICATE label "{lb}" (also in {labels[lb]})')
        else:
            labels[lb] = rel
    refs += [(m.group(1), rel) for m in re.finditer(r'\\(?:page)?ref\{([^}]+)\}', src)]
    for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]+)\}', src):
        cites += [(k.strip(), rel) for k in m.group(1).split(',')]

# --- dangling refs ---
for r, rel in refs:
    if r not in labels:
        problems.append(f'{rel}: \\ref to undefined label "{r}"')

# --- missing bib keys ---
bib = (ROOT / 'database.bib').read_text(encoding='utf-8', errors='replace')
bibkeys = set(re.findall(r'@[a-zA-Z]+\{([^,]+),', bib))
for c, rel in cites:
    if c not in bibkeys:
        problems.append(f'{rel}: \\cite to missing bib key "{c}"')

print(f'files checked : {len([f for f in FILES if (ROOT/f).exists()])}/{len(FILES)}')
print(f'labels defined: {len(labels)}')
print(f'refs used     : {len(refs)}   cites used: {len(cites)}   bib keys: {len(bibkeys)}')
print()
if problems:
    print(f'PROBLEMS ({len(problems)}):')
    for p_ in problems:
        print('  -', p_)
    sys.exit(1)
print('No structural problems found.')
