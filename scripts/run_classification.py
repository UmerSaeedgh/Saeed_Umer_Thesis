"""
Model-experiment runner.

Calls the OpenAI / Anthropic / Gemini / Moonshot (Kimi K3) chat-completions
APIs to fill in the <Model>_label / <Model>_reason columns of the 3
experiment CSVs (experiment_results_unambiguous.csv,
experiment_results_author_independent.csv, experiment_results_author_relevant.csv).

Reads API keys from environment variables -- NEVER hardcode a key in this
file or paste one into a chat/terminal that gets logged. Set them in your
own shell session before running:

  OPENAI_API_KEY      (openai)
  ANTHROPIC_API_KEY   (claude)
  GOOGLE_API_KEY       (gemini)
  MOONSHOT_API_KEY    (kimi)

A model is skipped entirely (with a warning) if its key isn't set.

Safety default: without --all, only the first --limit (default 5) EMPTY
rows per (file, model) are processed -- a cheap pilot to sanity-check
parsing/format before spending real budget on the full ~5,400 rows per
model per file (16,200 rows x number of models after the numbered/inline
label-format split). Pass --all to run everything. Pass --dry-run to
exercise the whole pipeline (row selection, checkpointing, parsing) with a
canned fake response and zero network calls / zero cost.

Resumable: only rows where <Model>_label is empty are sent; progress is
saved to the CSV every --checkpoint-every rows (default 20) and again at
the end, so an interrupted run loses at most a few rows of work, not the
whole run. Safe to stop (Ctrl-C) and rerun the same command later.

Usage examples:
  python run_classification.py --dry-run --models openai --files unambiguous --limit 5
  python run_classification.py --models openai --files unambiguous --limit 10
  python run_classification.py --models openai,claude,gemini,kimi --all
"""

import argparse
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

KEY_COLS = ['text_id', 'framing_condition', 'prompt_variant', 'label_format']

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'

FILES = {
    'unambiguous': BASE / 'experiment_results_unambiguous.csv',
    'author_independent': BASE / 'experiment_results_author_independent.csv',
    'author_relevant': BASE / 'experiment_results_author_relevant.csv',
}

SYSTEM_PROMPT = (
    "You are an emotion annotation assistant. Read the text and identify the "
    "single emotion the author felt. Choose only from the provided labels."
)

MODEL_CONFIG = {
    'openai': {
        'label_col': 'ChatGPT_label',
        'reason_col': 'ChatGPT_reason',
        'env_key': 'OPENAI_API_KEY',
        'model_name': os.environ.get('OPENAI_MODEL', 'gpt-5-nano'),
    },
    'claude': {
        'label_col': 'Claude_label',
        'reason_col': 'Claude_reason',
        'env_key': 'ANTHROPIC_API_KEY',
        'model_name': os.environ.get('ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001'),
    },
    'gemini': {
        'label_col': 'Gemini_label',
        'reason_col': 'Gemini_reason',
        'env_key': 'GOOGLE_API_KEY',
        'model_name': os.environ.get('GEMINI_MODEL', 'gemini-3.6-flash'),
    },
    'kimi': {
        'label_col': 'KimiK3_label',
        'reason_col': 'KimiK3_reason',
        'env_key': 'MOONSHOT_API_KEY',
        'model_name': os.environ.get('MOONSHOT_MODEL', 'kimi-k3'),
    },
    'qwen3': {
        'label_col': 'Qwen3_label',
        'reason_col': 'Qwen3_reason',
        'env_key': None,
        'model_name': os.environ.get('QWEN3_MODEL', 'qwen3:8b'),
    },
    'gemma3': {
        'label_col': 'Gemma3_label',
        'reason_col': 'Gemma3_reason',
        'env_key': None,
        'model_name': os.environ.get('GEMMA3_MODEL', 'gemma3:4b'),
    },
    'ministral': {
        'label_col': 'Ministral_label',
        'reason_col': 'Ministral_reason',
        'env_key': None,
        'model_name': os.environ.get('MINISTRAL_MODEL', 'ministral-3:8b'),
    },
    'llama31': {
        'label_col': 'Llama31_label',
        'reason_col': 'Llama31_reason',
        'env_key': None,
        'model_name': os.environ.get('LLAMA31_MODEL', 'llama3.1:8b'),
    },
}

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

LABEL_RE = re.compile(r'label\s*:\s*(.+?)\s*(?:\n|$)', re.IGNORECASE)
REASON_RE = re.compile(r'reason\s*:\s*(.+)', re.IGNORECASE | re.DOTALL)


def parse_response(text):
    """Returns (label, reason). If the expected 'Label: ... / Reason: ...'
    pattern isn't found, saves the raw response as-is in both fields, per
    the is_valid_label policy documented in standardized_prompting.docx --
    invalid/unparseable output is kept for analysis, not discarded."""
    text = (text or "").strip()
    label_m = LABEL_RE.search(text)
    reason_m = REASON_RE.search(text)
    if not label_m:
        return text[:200], text
    label = label_m.group(1).strip()
    reason = reason_m.group(1).strip() if reason_m else ""
    return label, reason


def call_openai(prompt, cfg, key):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg['model_name'],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def call_claude(prompt, cfg, key):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg['model_name'],
            "max_tokens": 200,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()['content'][0]['text']


def call_gemini(prompt, cfg, key):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model_name']}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text']


def call_kimi(prompt, cfg, key):
    r = requests.post(
        "https://api.moonshot.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg['model_name'],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def call_ollama(prompt, cfg, key):
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": cfg['model_name'],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()['message']['content']


CALLERS = {
    'openai': call_openai, 'claude': call_claude, 'gemini': call_gemini, 'kimi': call_kimi,
    'qwen3': call_ollama, 'gemma3': call_ollama, 'ministral': call_ollama, 'llama31': call_ollama,
}


class RateLimiter:
    """Proactively paces requests to at most 1 per min_interval seconds,
    shared across all worker threads via a lock. Cheaper than firing at
    full speed and reacting to 429s after the fact -- a free-tier quota
    (e.g. Gemini's 20 req/min) gets hit immediately at any concurrency > the
    quota, wasting the whole retry/backoff cycle on requests that were
    always going to be rejected."""
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
                now = time.monotonic()
            self.next_ok = max(now, self.next_ok) + self.min_interval


def call_with_retry(caller, prompt, cfg, key, max_retries=4, dry_run=False, limiter=None):
    if dry_run:
        return f"Label: joy\nReason: [DRY RUN -- no API call made for model={cfg['model_name']}]"
    delay = 2
    last_err = None
    for attempt in range(max_retries):
        if limiter:
            limiter.wait()
        try:
            return caller(prompt, cfg, key)
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2
    return f"Label: \nReason: [ERROR after {max_retries} retries: {last_err}]"


def partial_path(path, model_key):
    """Per-(model,file) results file. Writing here instead of the shared
    master CSV lets multiple models run as separate concurrent processes
    against the same 3 master files without one process's checkpoint
    overwriting another's -- each process only ever writes its own file."""
    return path.parent / f"{path.stem}.{model_key}.partial.csv"


def load_partial(ppath):
    if ppath.exists():
        return pd.read_csv(ppath)
    return pd.DataFrame(columns=KEY_COLS + ['label', 'reason'])


def process_file(model_key, file_key, path, limit, all_rows, workers, checkpoint_every, dry_run,
                  min_interval=0):
    cfg = MODEL_CONFIG[model_key]
    api_key = os.environ.get(cfg['env_key']) if cfg['env_key'] else None
    if cfg['env_key'] and not api_key and not dry_run:
        print(f"[skip] {model_key}/{file_key}: {cfg['env_key']} not set")
        return

    df = pd.read_csv(path)
    label_col = cfg['label_col']
    if label_col not in df.columns:
        print(f"[skip] {model_key}/{file_key}: column {label_col} not found")
        return

    ppath = partial_path(path, model_key)
    partial = load_partial(ppath)
    done_keys = set(map(tuple, partial[KEY_COLS].values.tolist())) if len(partial) else set()

    # Rows already filled directly in the master (e.g. earlier pilots written
    # before partial-file mode existed) count as done too, so they aren't re-paid for.
    already = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != '')]
    done_keys |= set(map(tuple, already[KEY_COLS].values.tolist()))

    df['_key'] = list(zip(*(df[c] for c in KEY_COLS)))
    pending_idx = [i for i, k in zip(df.index, df['_key']) if k not in done_keys]
    if not all_rows:
        pending_idx = pending_idx[:limit]

    if not pending_idx:
        print(f"[done] {model_key}/{file_key}: no pending rows (limit={limit}, all={all_rows})")
        return

    print(f"[run]  {model_key}/{file_key}: {len(pending_idx)} rows to process "
          f"(model={cfg['model_name']}, dry_run={dry_run}) -> {ppath.name}")

    caller = CALLERS[model_key]
    completed = 0
    new_rows = []
    lock = threading.Lock()
    limiter = RateLimiter(min_interval) if min_interval > 0 else None
    if limiter:
        print(f"  pacing at 1 request per {min_interval}s (~{60/min_interval:.1f}/min)")

    def work(idx):
        prompt = df.at[idx, 'full_prompt']
        raw = call_with_retry(caller, prompt, cfg, api_key, dry_run=dry_run, limiter=limiter)
        label, reason = parse_response(raw)
        key = tuple(df.at[idx, c] for c in KEY_COLS)
        return key, label, reason

    def flush():
        nonlocal partial
        if new_rows:
            add = pd.DataFrame(list(new_rows), columns=KEY_COLS + ['label', 'reason'])
            partial = pd.concat([partial, add], ignore_index=True)
            partial.to_csv(ppath, index=False)
            new_rows.clear()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, idx): idx for idx in pending_idx}
        for fut in as_completed(futures):
            try:
                key, label, reason = fut.result()
                with lock:
                    new_rows.append((*key, label, reason))
            except Exception:
                traceback.print_exc()
            completed += 1
            if completed % checkpoint_every == 0:
                with lock:
                    flush()
                print(f"  ... {completed}/{len(pending_idx)} checkpointed -> {ppath.name}")

    with lock:
        flush()
    print(f"[saved] {model_key}/{file_key}: {completed}/{len(pending_idx)} rows written to {ppath.name}")


def merge_partials(file_key, path, models):
    df = pd.read_csv(path)
    changed = False
    for model_key in models:
        cfg = MODEL_CONFIG[model_key]
        label_col, reason_col = cfg['label_col'], cfg['reason_col']
        if label_col not in df.columns:
            continue
        ppath = partial_path(path, model_key)
        if not ppath.exists():
            print(f"[merge] {model_key}/{file_key}: no partial file found, skipping")
            continue
        partial = pd.read_csv(ppath)
        df[label_col] = df[label_col].astype(object)
        df[reason_col] = df[reason_col].astype(object)
        partial_map = {
            tuple(r[c] for c in KEY_COLS): (r['label'], r['reason'])
            for _, r in partial.iterrows()
        }
        merged = 0
        for i, row in df.iterrows():
            k = tuple(row[c] for c in KEY_COLS)
            if k in partial_map:
                cur = row[label_col]
                if pd.isna(cur) or str(cur).strip() == '':
                    label, reason = partial_map[k]
                    df.at[i, label_col] = label
                    df.at[i, reason_col] = reason
                    merged += 1
                    changed = True
        print(f"[merge] {model_key}/{file_key}: merged {merged} new rows from {ppath.name} "
              f"({len(partial) - merged} were already in master)")
    if changed:
        df.to_csv(path, index=False)
        print(f"[merge] {file_key}: master file saved")
    else:
        print(f"[merge] {file_key}: nothing new to merge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='openai,claude,gemini,kimi',
                     help="comma-separated subset of: openai,claude,gemini,kimi")
    ap.add_argument('--files', default='unambiguous,author_independent,author_relevant',
                     help="comma-separated subset of: unambiguous,author_independent,author_relevant")
    ap.add_argument('--limit', type=int, default=5,
                     help="max empty rows to process per (model,file) unless --all is passed")
    ap.add_argument('--all', action='store_true', help="process ALL pending rows, not just --limit")
    ap.add_argument('--workers', type=int, default=5, help="concurrent requests per (model,file)")
    ap.add_argument('--min-interval', type=float, default=0,
                     help="seconds between successive request starts, to proactively stay under "
                          "a rate limit (e.g. 3.5 for Gemini's free-tier 20 req/min with margin) "
                          "instead of firing at full speed and reacting to 429s")
    ap.add_argument('--checkpoint-every', type=int, default=20)
    ap.add_argument('--dry-run', action='store_true',
                     help="exercise the full pipeline with no network calls / no cost")
    ap.add_argument('--merge', action='store_true',
                     help="don't call any API -- merge existing *.partial.csv files for "
                          "--models into the master CSVs for --files (single-writer, safe "
                          "to run after parallel process_file runs finish)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    files = [f.strip() for f in args.files.split(',') if f.strip()]

    if args.merge:
        for file_key in files:
            if file_key not in FILES:
                print(f"Unknown file '{file_key}', skipping. Valid: {list(FILES)}")
                continue
            merge_partials(file_key, FILES[file_key], models)
        return

    for model_key in models:
        if model_key not in MODEL_CONFIG:
            print(f"Unknown model '{model_key}', skipping. Valid: {list(MODEL_CONFIG)}")
            continue
        for file_key in files:
            if file_key not in FILES:
                print(f"Unknown file '{file_key}', skipping. Valid: {list(FILES)}")
                continue
            process_file(model_key, file_key, FILES[file_key], args.limit, args.all,
                         args.workers, args.checkpoint_every, args.dry_run,
                         min_interval=args.min_interval)


if __name__ == '__main__':
    main()
