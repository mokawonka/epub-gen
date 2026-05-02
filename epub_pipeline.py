#!/usr/bin/env python3
"""
epub_pipeline.py
────────────────
EPUB processing pipeline (summarise → cover → inject → embed → metadata).

For every .epub in INPUT_FOLDER it runs, in order:

  Phase 1  – Extract    : strip front/back matter, pull clean body text
  Phase 1b – Statistics : compute French-aware text statistics
  Phase 2  – Summarise  : hierarchical chunk summarisation (Ollama)
  Phase 3  – Generate   : image prompt + French back-cover blurb (Ollama)
  Phase 4  – Image      : cover image via Google Gemini Imagen
  Phase 5  – Composite  : overlay title/author band on cover image
  Phase 6  – Inject     : rebuild EPUB with new cover (EPUB 3)
  Phase 6b – Downscale  : overwrite cover_final.jpg with 500 px-wide web version
  Phase 7  – Embed      : embed chunk summaries with qwen3-embedding → metadata.json

Output layout (one sub-folder per book):

  <output_folder>/
    <stem>/
      <stem>_final.epub                 ← shareable: original text + full-res cover
      <stem>_cover_final.jpg            ← 500 px wide web cover (overwritten after EPUB build)
      cover_raw.png                     ← raw Imagen output
      <stem>_master_summary.txt
      <stem>_image_prompt.txt
      <stem>_quatrieme_de_couverture.txt
      <stem>_stats.json                 ← text statistics checkpoint
      metadata.json                     ← title, author, blurb, summary,
                                           cover path, epub path,
                                           chunk_vectors, mean_vector,
                                           stats (text statistics)
      chunk_checkpoints/
        chunk_0000_summary.txt          ← per-chunk summary (resume)

Requirements:
    pip install ebooklib beautifulsoup4 lxml pillow requests tqdm google-genai numpy

External services:
    • Ollama        – http://localhost:11434  (qwen2.5:7b recommended for text)
                      qwen3-embedding         (must be pulled for embeddings)
    • Google AI key – env var GEMINI_API_KEY
                      Get one at https://aistudio.google.com/app/apikey

Font (optional but recommended):
    League Spartan Bold from https://fonts.google.com/specimen/League+Spartan
    Place LeagueSpartan-Bold.ttf next to this script (or set FONT_PATH below).

Usage:
    python epub_pipeline.py input_folder/ output_folder/
    python epub_pipeline.py input_folder/ output_folder/ --model qwen2.5:7b
    python epub_pipeline.py input_folder/ output_folder/ --only-blurb
    python epub_pipeline.py input_folder/ output_folder/ -r   # recursive search
    python epub_pipeline.py input_folder/ output_folder/ --cover-image existing.png
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    sys.exit("❌  numpy not found.  Run: pip install numpy")

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    sys.exit("❌  ebooklib not found.  Run: pip install ebooklib")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("❌  google-genai not found.  Run: pip install google-genai")

# ══════════════════════════════════════════════════════════════════════════════
# Global configuration  (all overridable via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL          = "http://localhost:11434"
OLLAMA_MODEL        = "mistral-nemo"      # synthesis / final steps / blurb
OLLAMA_CHUNK_MODEL  = "mistral-nemo"      # per-chunk summarise
OLLAMA_EMBED_MODEL  = "qwen3-embedding" # embedding model for chunk summaries
OLLAMA_TIMEOUT      = 1800              # seconds per LLM call
RETRY_ATTEMPTS      = 4
RETRY_DELAY         = 10.0              # seconds for first retry; multiplied by attempt#

# Google Gemini Imagen
IMAGEN_MODEL        = "imagen-4.0-ultra-generate-001"
IMAGEN_ASPECT_RATIO = "3:4"             # portrait — closest to book cover ratio

COVER_W             = 1400
COVER_H             = 2100
IMAGE_HEIGHT_RATIO  = 0.75          # top 75% = image, bottom 25% = text band
TEXT_BAND_COLOR     = (15, 20, 45)
TEXT_COLOR          = (255, 255, 255)
TEXT_SHADOW_COLOR   = (0, 0, 0)
TEXT_SHADOW_OFFSET  = 3
TEXT_PADDING        = 40            # horizontal padding for text
TEXT_V_PADDING      = 30           # vertical padding top of band
TEXT_BOTTOM_PADDING = 24           # minimum gap between last text line and band bottom

# Imprint ("Éditions Skookoo") rendered at the very bottom of the cover
IMPRINT_TEXT        = "Éditions Skookoo"
IMPRINT_COLOR       = (212, 175, 55)   # golden
IMPRINT_SHADOW      = (0, 0, 0)
IMPRINT_FONT_SIZE   = 36               # px — adjust to taste
IMPRINT_BOTTOM_GAP  = 28              # pixels above the very bottom edge

CHUNK_WORDS         = 2000
MAX_CHUNK_TOKENS    = 3000

# Web cover thumbnail — overwrites cover_final.jpg after EPUB is built
WEB_COVER_WIDTH     = 500           # px; height is computed to preserve aspect ratio
WEB_COVER_QUALITY   = 93            # JPEG quality for the web thumbnail

FONT_PATH     = Path(__file__).parent / "LeagueSpartan-Bold.ttf"
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Average reading speed in words per minute (French prose)
READING_WPM = 200

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''} {s:02d} second{'s' if s != 1 else ''}"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Ollama helpers
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama() -> None:
    """Verify Ollama is reachable and the required models are pulled."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as exc:
        sys.exit(f"❌  Cannot reach Ollama at {OLLAMA_URL}: {exc}\n"
                 "    Make sure Ollama is running.")
    names = [m["name"] for m in r.json().get("models", [])]
    for model in set([OLLAMA_MODEL, OLLAMA_CHUNK_MODEL, OLLAMA_EMBED_MODEL]):
        base = model.split(":")[0]
        if not any(base in n for n in names):
            sys.exit(f"❌  Model '{model}' not found in Ollama.\n"
                     f"    Run: ollama pull {model}\n"
                     f"    Available: {names}")


# ══════════════════════════════════════════════════════════════════════════════
# ollama_generate  (with aggressive retry logic)
# ══════════════════════════════════════════════════════════════════════════════

def ollama_generate(
    prompt: str,
    model: str,
    system: str = "",
) -> str:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_thread": 6,
        }
    }
    
    if system:
        payload["system"] = system

    last_exc: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        session = requests.Session()
        timer   = None
        try:
            def _force_close():
                print(f"\n  ⚠ Hard-closing hung Ollama connection "
                      f"(attempt {attempt}/{RETRY_ATTEMPTS})…")
                session.close()

            timer = threading.Timer(OLLAMA_TIMEOUT, _force_close)
            timer.daemon = True
            timer.start()

            r = session.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT + 5,
            )
            r.raise_for_status()
            return r.json()["response"].strip()

        except Exception as exc:
            last_exc = exc
            if attempt == RETRY_ATTEMPTS:
                break
            wait = RETRY_DELAY * attempt
            print(f"\n  ⚠ Ollama error (attempt {attempt}/{RETRY_ATTEMPTS}): "
                  f"{type(exc).__name__} — retrying in {wait:.0f}s…")
            time.sleep(wait)

        finally:
            if timer is not None:
                timer.cancel()
            session.close()

    raise RuntimeError(
        f"Ollama failed after {RETRY_ATTEMPTS} attempts. Last error: {last_exc}"
    )


def ollama_unload(model: str) -> None:
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        print(f"  → Unloaded '{model}' from Ollama VRAM.")
    except Exception as exc:
        print(f"  ⚠ Could not unload '{model}': {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – EPUB text extraction
# ══════════════════════════════════════════════════════════════════════════════

def load_epub_text_and_meta(epub_path: Path) -> tuple[str, str, str, int]:
    """Return (body_text, title, author, total_words)."""
    from html.parser import HTMLParser

    FRONTBACK_FILENAME_RE = re.compile(
        r"""
        cover | toc | table.?of.?contents | contents |
        copyright | copyrights | legal | licence | license |
        preface | pr[eé]face | foreword | avant.?propos |
        introduction | intro(?!duc) |
        dedication | d[eé]dicace |
        epigraph |
        about.?author | colophon | acknowledgement | acknowledgment |
        bibliography | index | glossary | appendix |
        back.?matter | front.?matter | half.?title | title.?page |
        note | notes(?!\.) |
        errata | permissions
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    GUIDE_SKIP_ROLES = {
        "cover", "toc", "copyright-page", "title-page",
        "dedication", "acknowledgements", "preface", "foreword",
        "bibliography", "index", "glossary", "colophon",
    }
    FRONTBACK_HEADING_RE = re.compile(
        r"^(table (of )?contents|avant.?propos|pr[eé]face|foreword|"
        r"d[eé]dicace|dedication|remerciements|acknowledgements?|"
        r"introduction|[ée]pilogue|postface|about the author|"
        r"bibliograph|index|glossaire|notes?|copyright|colophon)s?$",
        re.IGNORECASE,
    )
    MIN_BODY_WORDS       = 80
    MIN_PROSE_LINE_WORDS = 6

    class _StripHTML(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self.first_heading = ""
            self._in_heading   = False
            self._heading_buf: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in ("h1", "h2", "h3", "h4"):
                self._in_heading  = True
                self._heading_buf = []
            elif tag == "hr":
                self.parts.append("\n\n--- SCENE ---\n\n")

        def handle_endtag(self, tag):
            if tag in ("h1", "h2", "h3", "h4") and self._in_heading:
                ht = " ".join(self._heading_buf).strip()
                if ht:
                    self.parts.append(f"\n\n=== HEADING: {ht} ===\n\n")
                    if not self.first_heading:
                        self.first_heading = ht
                self._in_heading  = False
                self._heading_buf = []

        def handle_data(self, data):
            s = data.strip()
            if not s:
                return
            if self._in_heading:
                self._heading_buf.append(s)
            else:
                self.parts.append(s)

        def get_text(self) -> str:
            return " ".join(self.parts)

    book   = epub.read_epub(str(epub_path))
    title  = book.get_metadata("DC", "title")
    title  = title[0][0] if title else epub_path.stem
    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else "Auteur inconnu"

    guide_skip_hrefs: set[str] = set()
    try:
        for item in book.guide:
            role = (item.get("type") or item.get("role") or "").lower()
            href = (item.get("href") or item.get("uri") or "").split("#")[0]
            if role in GUIDE_SKIP_ROLES and href:
                guide_skip_hrefs.add(href.lstrip("/"))
    except Exception:
        pass

    id_to_item = {
        item.id: item
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    }
    spine_ids   = [idref for idref, _ in book.spine]
    spine_items = (
        [id_to_item[i] for i in spine_ids if i in id_to_item]
        if spine_ids
        else list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    )

    def _is_frontback(item) -> tuple[bool, str]:
        href     = item.get_name().lstrip("/")
        basename = Path(href).stem.lower()
        if FRONTBACK_FILENAME_RE.search(basename):
            return True, f"filename '{basename}'"
        for gh in guide_skip_hrefs:
            if href.endswith(gh) or gh.endswith(href):
                return True, "guide role"
        try:
            raw = item.get_body_content().decode("utf-8", errors="ignore")
        except Exception:
            return True, "unreadable"
        parser = _StripHTML()
        parser.feed(raw)
        text = parser.get_text().strip()
        if len(text.split()) < MIN_BODY_WORDS:
            return True, f"too short ({len(text.split())} words)"
        if parser.first_heading and FRONTBACK_HEADING_RE.match(
            parser.first_heading.strip()
        ):
            return True, f"heading '{parser.first_heading.strip()}'"
        return False, ""

    body_items = []
    skipped    = []
    for item in spine_items:
        skip, reason = _is_frontback(item)
        if skip:
            skipped.append((item.get_name(), reason))
        else:
            body_items.append(item)

    if skipped:
        print(f"  ℹ  Skipped {len(skipped)} front/back-matter document(s).")
        for name, reason in skipped:
            print(f"       – {name}  ({reason})")

    if not body_items:
        print("  ⚠  All documents filtered as front/back-matter — using full text.")
        body_items = spine_items

    def _extract(item) -> str:
        raw = item.get_body_content().decode("utf-8", errors="ignore")
        parser = _StripHTML()
        parser.feed(raw)
        return parser.get_text().strip()

    texts = [_extract(item) for item in body_items]

    # Trim leading boilerplate from first chapter
    if texts:
        first_lines = texts[0].splitlines()
        start = next(
            (i for i, l in enumerate(first_lines) if len(l.split()) >= MIN_PROSE_LINE_WORDS),
            0,
        )
        texts[0] = "\n".join(first_lines[start:])

    # Trim trailing boilerplate from last chapter
    if texts:
        last_lines = texts[-1].splitlines()
        end = next(
            (i + 1 for i in range(len(last_lines) - 1, -1, -1)
             if len(last_lines[i].split()) >= MIN_PROSE_LINE_WORDS),
            len(last_lines),
        )
        texts[-1] = "\n".join(last_lines[:end])

    full_text   = "\n\n".join(t for t in texts if t)
    total_words = len(full_text.split())
    return full_text, title, author, total_words


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1b – Text Statistics (French-aware)
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_french_syllables(word: str) -> int:
    """
    Heuristic French syllable counter.
    Counts vowel groups (including accented vowels), with common French rules:
      – silent final 'e' after a consonant is not counted
      – 'eau', 'au', 'ai', 'ei', 'oi', 'ou', 'eu', 'oeu' count as 1 syllable
      – minimum 1 syllable per word
    """
    w = word.lower().strip(".,;:!?\"'«»—–-()[]{}")
    if not w:
        return 0

    VOWELS = "aeiouyàâäéèêëîïôöùûüœæ"

    # Merge common French digraphs / trigraphs into a single placeholder
    for digraph in ("eau", "oeu", "oei", "au", "ai", "ei", "oi", "ou", "eu", "ae", "oe"):
        w = w.replace(digraph, "V")

    # Now replace remaining single vowels
    w = re.sub(f"[{VOWELS}]", "V", w)

    # Remove silent final 'e' (not after another vowel)
    if w.endswith("Ve") or (w.endswith("e") and len(w) > 1 and w[-2] != "V"):
        w = w[:-1]

    count = w.count("V")
    return max(count, 1)


def _tokenize_words_fr(text: str) -> list[str]:
    """Return lowercase alphabetic tokens (handles French apostrophes)."""
    # Normalise typographic apostrophes
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # Keep only alphabetic chars and apostrophes, split on everything else
    tokens = re.findall(r"[a-zA-ZÀ-ÿœæ]+(?:'[a-zA-ZÀ-ÿœæ]+)*", text)
    return [t.lower() for t in tokens]


def _split_sentences_fr(text: str) -> list[str]:
    """
    Split French prose into sentences.
    Handles '.' '!' '?' as terminators; skips common French abbreviations
    (M. Mme. Dr. etc.) to avoid false splits.
    """
    # Common French honorifics / abbreviations that should NOT end a sentence
    ABBREVS = re.compile(
        r"\b(M|Mme|Mlle|MM|Dr|Pr|Me|St|Ste|Sr|Jr|vol|art|chap|fig|ex|cf|"
        r"éd|trad|ibid|op|cit|p|pp|vs|env|env|apr|av|etc)\.$",
        re.IGNORECASE,
    )

    # Protect abbreviations by temporarily replacing their period
    protected = re.sub(
        r"\b(M|Mme|Mlle|MM|Dr|Pr|Me|St|Ste|Sr|Jr|vol|art|chap|fig|ex|cf|"
        r"éd|trad|ibid|op|cit|p|pp|vs|env|apr|av|etc)\.",
        r"\1<DOT>",
        text,
        flags=re.IGNORECASE,
    )
    # Split on sentence-ending punctuation followed by space/newline + uppercase
    parts = re.split(r"(?<=[.!?…])\s+(?=[«\"'\u2018\u2019A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŒÆ])", protected)
    # Restore protected dots
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences if sentences else [text]


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; filter very short artefacts (headings, scene breaks)."""
    paras = re.split(r"\n{2,}", text)
    return [
        p.strip() for p in paras
        if p.strip()
        and not re.match(r"^(===|---).*", p.strip())
        and len(p.split()) >= 5
    ]


def _count_chapters(text: str) -> int:
    """Count HEADING markers inserted by the extractor."""
    return max(len(re.findall(r"^=== HEADING:", text, re.MULTILINE)), 1)


def _dialogue_ratio(text: str) -> float:
    """
    Estimate the fraction of words inside dialogue markers.
    Handles four French conventions:
      1. Guillemets  «…»
      2. Em-dash lines  – / — at line start
      3. Play scripts with proper line breaks (SPEAKER on its own line)
      4. Play scripts collapsed into a single line per scene:
         "Gorgibus Bonjour. Sganarelle Oui, merci." — speaker names inline
    """
    total_words = len(text.split())
    if total_words == 0:
        return 0.0

    dialogue_words = 0

    # Pre-compile patterns
    SPEAKER_RE = re.compile(
        r"^[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŒÆ\s\-\.']{2,40}[.:\,]?\s*$"
    )
    STAGE_DIR_RE = re.compile(r"^\s*\(.*\)\s*$")
    SCENE_HDR_RE = re.compile(
        r"^(acte|scène|scene|tableau|prologue|épilogue|fin)\b",
        re.IGNORECASE,
    )
    # Matches an inline speaker name: one or more capitalised words (possibly
    # hyphenated or with apostrophe) that appear mid-line before speech.
    # e.g. "Gorgibus", "Gros-René", "L'Avocat"
    INLINE_SPEAKER_RE = re.compile(
        r"(?<![a-zàâäéèêëîïôöùûüœæ])"   # not preceded by a lowercase letter
        r"([A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŒÆ][a-zA-ZÀ-ÿœæ\-']*"
        r"(?:\s+[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŒÆ][a-zA-ZÀ-ÿœæ\-']*){0,3})"
        r"(?=\s+[«A-ZÀ-ÿœæa-z\"\-–—])"  # followed by speech
    )

    # ── Convention 1 : guillemets ─────────────────────────────────────────
    for span in re.finditer(r"«[^»]{1,2000}»", text):
        dialogue_words += len(span.group().split())

    lines = text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and structural markers
        if not line or SCENE_HDR_RE.match(line) or line.startswith("==="):
            i += 1
            continue

        # ── Convention 2 : em-dash ────────────────────────────────────────
        if line.startswith(("–", "—", "- ")):
            dialogue_words += len(line.split())
            i += 1
            continue

        # ── Convention 3 : proper line-break script ───────────────────────
        if SPEAKER_RE.match(line) and not STAGE_DIR_RE.match(line):
            i += 1
            while i < len(lines):
                speech = lines[i].strip()
                if not speech:
                    i += 1
                    continue
                if (
                    SPEAKER_RE.match(speech)
                    or STAGE_DIR_RE.match(speech)
                    or SCENE_HDR_RE.match(speech)
                ):
                    break
                dialogue_words += len(speech.split())
                i += 1
            continue

        # ── Convention 4 : collapsed scene line ───────────────────────────
        # Split the line on inline speaker name boundaries, then count
        # everything after each speaker name as dialogue.
        #
        # Strategy: find all speaker-name positions in the line, then treat
        # the text between consecutive speaker positions as speech.
        matches = list(INLINE_SPEAKER_RE.finditer(line))
        if matches:
            # Collect (start_of_speech, end_of_speech) spans
            for idx, m in enumerate(matches):
                speech_start = m.end()
                speech_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
                speech_text  = line[speech_start:speech_end].strip()
                # Strip leading stage directions in parentheses
                speech_text  = re.sub(r"^\(.*?\)\s*", "", speech_text)
                dialogue_words += len(speech_text.split())

        i += 1

    return round(min(dialogue_words / total_words * 100, 100.0), 2)


def _lexical_diversity_msttr(tokens: list[str], window: int = 500) -> float:
    """
    Mean Segmental Type-Token Ratio (MSTTR) — more stable than raw TTR for
    long texts.  Splits the token list into non-overlapping windows of `window`
    tokens and averages the TTR of each window.
    """
    if not tokens:
        return 0.0
    ttrs = []
    for i in range(0, len(tokens) - window + 1, window):
        seg = tokens[i : i + window]
        ttrs.append(len(set(seg)) / len(seg))
    if not ttrs:
        # corpus shorter than one window — plain TTR
        ttrs = [len(set(tokens)) / len(tokens)]
    return round(sum(ttrs) / len(ttrs) * 100, 2)


def _hapax_legomena_ratio(tokens: list[str]) -> float:
    """Words appearing exactly once / total vocabulary × 100."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax_count = sum(1 for f in freq.values() if f == 1)
    return round(hapax_count / len(freq) * 100, 2)


def _kandel_moles(avg_sentence_len: float, avg_syllables_per_word: float) -> float:
    """
    Kandel & Moles (1958) French readability formula:
        FK = 209 − 1.15 × L − 68.48 × C
    where L = average sentence length in words,
          C = average number of syllables per word.

    Score interpretation (higher = easier):
      ≥ 100  : very easy (children's books)
        60–100: standard / easy
        30–60 : moderately difficult
         0–30 : difficult (academic / technical)
       < 0    : very difficult
    """
    score = 209 - 1.15 * avg_sentence_len - 68.48 * avg_syllables_per_word
    return round(score, 2)


def compute_text_statistics(body_text: str, verbose: bool = True) -> dict:
    """
    Compute French-aware text statistics from the extracted body text.

    Returns a dict with all metrics ready to be serialised to JSON.
    """
    if verbose:
        print("  → Tokenising words…", flush=True)
    tokens = _tokenize_words_fr(body_text)
    word_count = len(tokens)

    if verbose:
        print("  → Splitting sentences…", flush=True)
    sentences = _split_sentences_fr(body_text)
    sentence_count = len(sentences)

    if verbose:
        print("  → Splitting paragraphs…", flush=True)
    paragraphs = _split_paragraphs(body_text)
    paragraph_count = len(paragraphs)

    chapter_count = _count_chapters(body_text)

    # Average lengths
    avg_sentence_len  = round(word_count / sentence_count, 2)   if sentence_count  else 0.0
    avg_paragraph_len = round(word_count / paragraph_count, 2)  if paragraph_count else 0.0
    avg_chapter_len   = round(word_count / chapter_count, 2)    if chapter_count   else 0.0

    # Syllable stats (sample up to 20 000 tokens for speed)
    if verbose:
        print("  → Computing syllable counts (Kandel-Moles)…", flush=True)
    sample_tokens = tokens[:20_000]
    if sample_tokens:
        total_syllables = sum(_count_french_syllables(w) for w in sample_tokens)
        avg_syllables   = total_syllables / len(sample_tokens)
    else:
        avg_syllables = 0.0

    kandel_moles_score = _kandel_moles(avg_sentence_len, avg_syllables)

    # Reading time
    reading_time_minutes = round(word_count / READING_WPM, 1) if word_count else 0.0

    # Dialogue ratio
    if verbose:
        print("  → Computing dialogue ratio…", flush=True)
    dialogue_pct = _dialogue_ratio(body_text)

    # Lexical diversity (MSTTR)
    if verbose:
        print("  → Computing lexical diversity (MSTTR-500)…", flush=True)
    lexical_diversity = _lexical_diversity_msttr(tokens, window=500)

    # Hapax legomena
    if verbose:
        print("  → Computing hapax legomena…", flush=True)
    hapax_pct = _hapax_legomena_ratio(tokens)

    stats: dict = {
        # Counts
        "word_count":              word_count,
        "sentence_count":          sentence_count,
        "paragraph_count":         paragraph_count,
        "chapter_count":           chapter_count,
        # Averages
        "avg_sentence_length":     avg_sentence_len,
        "avg_paragraph_length":    avg_paragraph_len,
        "avg_chapter_length":      avg_chapter_len,
        # Readability
        "kandel_moles_score":      kandel_moles_score,
        "avg_syllables_per_word":  round(avg_syllables, 3),
        # Reader-facing
        "estimated_reading_time_minutes": reading_time_minutes,
        "dialogue_ratio_pct":      dialogue_pct,
        # Vocabulary richness
        "lexical_diversity_msttr": lexical_diversity,
        "hapax_legomena_pct":      hapax_pct,
    }

    if verbose:
        _print_stats(stats)

    return stats


def _print_stats(stats: dict) -> None:
    """Pretty-print the computed statistics."""
    km   = stats["kandel_moles_score"]
    if km >= 100:
        km_label = "très facile"
    elif km >= 60:
        km_label = "facile / standard"
    elif km >= 30:
        km_label = "modérément difficile"
    elif km >= 0:
        km_label = "difficile"
    else:
        km_label = "très difficile"

    rt = stats["estimated_reading_time_minutes"]
    if rt < 60:
        rt_str = f"{rt:.0f} min"
    else:
        h = int(rt // 60)
        m = int(rt % 60)
        rt_str = f"{h}h {m:02d}min"

    print(f"\n  ┌── Text Statistics ─────────────────────────────────┐")
    print(f"  │  Words              : {stats['word_count']:>10,}                   │")
    print(f"  │  Sentences          : {stats['sentence_count']:>10,}                   │")
    print(f"  │  Paragraphs         : {stats['paragraph_count']:>10,}                   │")
    print(f"  │  Chapters           : {stats['chapter_count']:>10,}                   │")
    print(f"  │  Avg sentence len   : {stats['avg_sentence_length']:>10.1f} words               │")
    print(f"  │  Avg paragraph len  : {stats['avg_paragraph_length']:>10.1f} words               │")
    print(f"  │  Avg chapter len    : {stats['avg_chapter_length']:>10.1f} words               │")
    print(f"  │  Kandel-Moles score : {km:>10.1f}  ({km_label})    │")
    print(f"  │  Avg syllables/word : {stats['avg_syllables_per_word']:>10.3f}                   │")
    print(f"  │  Est. reading time  : {rt_str:>10}                   │")
    print(f"  │  Dialogue ratio     : {stats['dialogue_ratio_pct']:>9.1f} %                   │")
    print(f"  │  Lexical diversity  : {stats['lexical_diversity_msttr']:>9.1f} %  (MSTTR-500)    │")
    print(f"  │  Hapax legomena     : {stats['hapax_legomena_pct']:>9.1f} %  of vocabulary    │")
    print(f"  └────────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 – Hierarchical summarisation
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks_smart(
    text: str,
    target_words: int = 2000,
    min_words: int    = 500,
    max_words: int    = 3500,
) -> list[str]:
    HEADING_RE = re.compile(r'^=== HEADING:.*===$')
    SCENE_RE   = re.compile(r'^--- SCENE ---$')

    units = [u.strip() for u in re.split(r'\n{2,}', text) if u.strip()]
    chunks: list[str] = []
    buf:    list[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        block = "\n\n".join(buf).strip()
        if block:
            chunks.append(block)
        buf, buf_words = [], 0

    for unit in units:
        is_heading    = bool(HEADING_RE.match(unit))
        is_scene      = bool(SCENE_RE.match(unit))
        is_structural = is_heading or is_scene
        unit_words    = 0 if is_structural else len(unit.split())

        if is_heading:
            if buf_words >= min_words:
                flush()
            buf.append(unit)
            continue

        if is_scene:
            if buf_words >= target_words * 0.75:
                flush()
            continue

        if buf_words + unit_words > max_words and buf_words >= min_words:
            flush()

        buf.append(unit)
        buf_words += unit_words

        if buf_words >= target_words:
            flush()

    flush()

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk.split()) < min_words:
            merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


def summarise_chunk(chunk: str, idx: int, total: int, model: str) -> str:
    words = chunk.split()
    if len(words) > MAX_CHUNK_TOKENS:
        chunk = " ".join(words[:MAX_CHUNK_TOKENS])

    system = (
        "Tu es un analyste littéraire. "
        "Tu dois OBLIGATOIREMENT écrire en français. "
        "Toute réponse en anglais est une erreur grave. "
        "Lis l'extrait fourni et rédige un résumé thématique concis en français "
        "(100–200 mots). Concentre-toi sur : les personnages principaux introduits "
        "ou développés, les événements clés, l'atmosphère dominante, "
        "les symboles ou motifs récurrents. "
        "INTERDICTIONS ABSOLUES : n'ajoute aucune formule de politesse, aucune invite "
        "à poser des questions, aucune mention de 'suite', aucun commentaire méta, "
        "aucune phrase du type 'n'hésitez pas', 'à suivre', 'si vous voulez en savoir plus'. "
        "Réponds UNIQUEMENT avec le résumé, sans préambule, en français."
    )
    try:
        return ollama_generate(
            f"[Fragment {idx + 1} sur {total}]\n\n{chunk}\n\n"
            "Rédige le résumé EN FRANÇAIS UNIQUEMENT.",
            model,
            system
        )
    except Exception as exc:
        placeholder = (
            f"[Résumé indisponible pour le fragment {idx + 1} — "
            f"erreur Ollama : {type(exc).__name__}]"
        )
        print(f"\n  ✗ Chunk {idx + 1} permanently failed: {exc}")
        print(f"    → Placeholder saved to checkpoint so pipeline can continue.")
        return placeholder
    

def hierarchical_summarise(
    full_text:   str,
    chunk_words: int,
    chunk_model: str,
    final_model: str,
    ckpt_dir,
    verbose:     bool = True,
) -> str:
    from pathlib import Path

    chunks = split_into_chunks_smart(
        full_text,
        target_words=chunk_words,
        min_words = max(400, int(chunk_words * 0.20)),
        max_words=chunk_words + 800,
    )
    n = len(chunks)
    if verbose:
        print(f"  → {n} chunk(s) of ~{chunk_words} words each.")

    chunk_summaries: list[str] = []
    failed_chunks:   list[int] = []

    for i, chunk in enumerate(chunks):
        ckpt_summary = Path(ckpt_dir) / f"chunk_{i:04d}_summary.txt"

        if ckpt_summary.exists():
            summary = ckpt_summary.read_text(encoding="utf-8")
            if verbose:
                print(f"  → Chunk {i+1}/{n}: ✓ loaded from checkpoint.")
            chunk_summaries.append(summary)
            continue

        word_count = len(chunk.split())
        if verbose:
            print(
                f"  → Summarising chunk {i+1}/{n} ({word_count:,} words)…",
                flush=True,
            )

        t0      = time.perf_counter()
        summary = summarise_chunk(chunk, i, n, chunk_model)
        elapsed = time.perf_counter() - t0

        ckpt_summary.write_text(summary, encoding="utf-8")
        chunk_summaries.append(summary)

        if "[Résumé indisponible" in summary:
            failed_chunks.append(i + 1)
        elif verbose:
            print(f"     ✓ Done in {fmt_duration(elapsed)}.")

    if failed_chunks and verbose:
        print(
            f"\n  ⚠ {len(failed_chunks)} chunk(s) could not be summarised "
            f"and were replaced with placeholders: {failed_chunks}"
        )

    if n == 1:
        return chunk_summaries[0]

    if verbose:
        print(f"  → Synthesising {n} summaries into master…", flush=True)

    numbered = "\n\n".join(
        f"[Chunk {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    system = (
        "Tu es un éditeur littéraire senior. "
        "Tu dois OBLIGATOIREMENT écrire en français. "
        "Toute réponse en anglais est une erreur grave. "
        "Synthétise les résumés suivants en un résumé maître cohérent en français "
        "(200–300 mots) qui capture l'arc narratif global, les personnages principaux, "
        "les thèmes centraux, l'atmosphère et les motifs visuels de l'œuvre. "
        "Si un résumé de fragment est marqué comme indisponible, ignore-le simplement. "
        "INTERDICTIONS ABSOLUES : n'ajoute aucune formule de politesse, aucune invite "
        "à poser des questions, aucune mention de 'suite', aucun commentaire méta, "
        "aucune phrase du type 'n'hésitez pas', 'à suivre', 'si vous voulez en savoir plus'. "
        "Réponds UNIQUEMENT avec le résumé maître, sans préambule, en français."
    )
    t0     = time.perf_counter()
    master = ollama_generate(
        f"Voici les résumés des fragments :\n\n{numbered}\n\n"
        "Rédige maintenant le résumé maître EN FRANÇAIS UNIQUEMENT.",
        final_model,
        system,
    )
    if verbose:
        print(f"     ✓ Master summary done in {fmt_duration(time.perf_counter()-t0)}.")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 – Image prompt + back-cover blurb
# ══════════════════════════════════════════════════════════════════════════════

def build_image_prompt(master_summary: str, title: str, author: str, model: str) -> str:
    english_summary = ollama_generate(
        prompt=master_summary,
        model=model,
        system="You are a translator. Translate the following French text to English. "
               "Output ONLY the translation, no preamble.",
    )

    system = (
        "You are an expert image-generation prompt engineer for book covers.\n"
        "You must describe scenes in a STRUCTURED, VISUAL way that image models can follow.\n\n"
        "RULES:\n"
        "- Explicitly state the number of characters.\n"
        "- Describe each character separately.\n"
        "- Specify spatial relationships (foreground, background, left, right).\n"
        "- Ensure all important characters are visible.\n"
        "- Avoid abstract or symbolic descriptions.\n"
        "- Keep it visually concrete and physically grounded.\n"
        "- If multiple important characters exist, NEVER reduce them to one.\n\n"
        "STYLE:\n"
        "- Realistic, cinematic, detailed\n"
        "- Period-accurate (clothing, lighting, architecture)\n"
        "- No text or typography in the image\n"
        "- Portrait orientation (taller than wide)\n"
    )

    user = (
        f"Book title: {title}\n"
        f"Author: {author}\n\n"
        f"Here is a master thematic summary:\n\n{english_summary}\n\n"
        "Write ONE image generation prompt (max 120 words).\n\n"
        "MANDATORY REQUIREMENTS:\n"
        "- State the number of characters explicitly (e.g., 'two men', 'one woman').\n"
        "- Describe each character separately.\n"
        "- Specify spatial layout (foreground, background, left, right).\n"
        "- Ensure all important characters are visible.\n"
        "- Use concrete visual descriptions only.\n"
        "- Respect historical accuracy.\n"
        "- No text, no typography.\n\n"
        "Output ONLY the final prompt as a single paragraph."
    )

    raw = ollama_generate(user, model, system)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[-1] if lines else raw


def build_back_cover(
    master_summary: str, title: str, author: str, model: str
) -> str:
    system = (
        "Tu es un éditeur littéraire français expert en rédaction de quatrièmes "
        "de couverture. "
        "Tu écris en français natif, irréprochable et élégant — sans aucun "
        "anglicisme, sans mots inventés, sans calques de l'anglais. "
        "Tout mot doit exister dans le dictionnaire français standard. "
        "Ne révèle pas la fin. Suscite la curiosité du lecteur."
    )
    user = (
        f"Titre : {title}\nAuteur : {author}\n\n"
        f"Résumé thématique complet :\n\n{master_summary}\n\n"
        "Rédige une quatrième de couverture convaincante (entre 150 et 250 mots). "
        "Utilise uniquement des mots existant dans le dictionnaire français. "
        "N'inclus pas le titre ni le nom de l'auteur dans le texte. "
        "Réponds UNIQUEMENT avec le texte de la quatrième de couverture."
    )
    raw_blurb = ollama_generate(user, model, system)

    correction_system = (
        "Tu es un correcteur littéraire français. "
        "Corrige UNIQUEMENT les fautes de langue dans le texte suivant : "
        "anglicismes, mots inventés, calques de l'anglais, fautes d'orthographe, "
        "accords incorrects. Ne réécris pas le texte — corrige seulement ce qui "
        "est fautif. "
        "Réponds UNIQUEMENT avec le texte corrigé, sans explication."
    )
    return ollama_generate(raw_blurb, model, correction_system)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 – Google Gemini Imagen image generation
# ══════════════════════════════════════════════════════════════════════════════

def check_google_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(
            "❌  GEMINI_API_KEY environment variable is not set.\n"
            "    Export it before running:\n"
            "      export GEMINI_API_KEY=AIza...\n"
            "    Get a key at: https://aistudio.google.com/app/apikey"
        )

def regenerate_image_prompt(
    master_summary: str, title: str, author: str, model: str,
    previous_prompt: str, attempt: int,
) -> str:
    system = (
        "You are an expert image-generation prompt engineer for book covers.\n"
        "A previous prompt was blocked by an image generation safety filter.\n"
        "You must rewrite it to be accepted while keeping the same scene and mood.\n\n"
        "STRICT SAFETY RULES — violations cause blocks:\n"
        "- NEVER describe any character as a child, boy, girl, kid, young, youth, minor.\n"
        "- Replace any young/small character with 'a slight ethereal figure', "
        "'a slender mysterious traveller', or 'a small cloaked figure' — no age references.\n"
        "- NEVER use: child, boy, girl, kid, young, little, small person, youth, minor.\n"
        "- NEVER mention weapons, blood, gore, violence, death, war, battle.\n"
        "- NEVER mention drugs, alcohol, nudity, or anything sexually suggestive.\n"
        "- Describe clothing, posture, and setting only — never age or physical vulnerability.\n\n"
        "STYLE:\n"
        "- Cinematic, painterly, detailed\n"
        "- Period-accurate clothing and architecture\n"
        "- No text or typography in the image\n"
        "- Portrait orientation (taller than wide)\n"
        f"- This is attempt {attempt} — be more conservative than the previous version.\n"
    )
    user = (
        f"Book title: {title}\n"
        f"Author: {author}\n\n"
        f"Master summary:\n{master_summary}\n\n"
        f"Previous blocked prompt:\n{previous_prompt}\n\n"
        "Rewrite the prompt (max 100 words) to pass safety filters "
        "while preserving the scene, mood, and characters.\n"
        "Output ONLY the rewritten prompt as a single paragraph."
    )
    raw = ollama_generate(user, model, system)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[-1] if lines else raw


def generate_image_imagen(
    prompt: str,
    out_dir: Path,
    model: str               = IMAGEN_MODEL,
    aspect_ratio: str        = IMAGEN_ASPECT_RATIO,
    master_summary: str      = "",
    title: str               = "",
    author: str              = "",
    ollama_model: str        = "",
    prompt_path: Path | None = None,
    max_prompt_attempts: int = 5,
) -> Path:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    words = prompt.split()
    if len(words) > 400:
        prompt = " ".join(words[:400])

    model_chain = [model, "imagen-4.0-generate-001", "imagen-4.0-fast-generate-001"]
    seen = set()
    model_chain = [m for m in model_chain if not (m in seen or seen.add(m))]

    current_prompt = prompt

    for prompt_attempt in range(1, max_prompt_attempts + 1):
        print(f"  → Prompt attempt {prompt_attempt}/{max_prompt_attempts}…")

        for current_model in model_chain:
            print(f"  → Sending to {current_model} (aspect={aspect_ratio})…")
            try:
                response = client.models.generate_images(
                    model=current_model,
                    prompt=current_prompt,
                    config=genai_types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=aspect_ratio,
                        person_generation="allow_adult",
                    ),
                )

                if not response.generated_images:
                    print(f"  ⚠ {current_model} blocked the prompt.")
                    continue

                raw_path = out_dir / "cover_raw.png"
                img_obj = response.generated_images[0].image

                if isinstance(img_obj, Image.Image):
                    img_obj.save(raw_path)
                else:
                    raw_bytes = None
                    for attr in ("_image_bytes", "image_bytes", "data", "content", "raw", "bytes"):
                        if hasattr(img_obj, attr):
                            raw_bytes = getattr(img_obj, attr)
                            break
                    if isinstance(img_obj, bytes):
                        raw_bytes = img_obj
                    if raw_bytes is None:
                        raise RuntimeError(f"Cannot extract image bytes from {type(img_obj)}")
                    Image.open(io.BytesIO(raw_bytes)).save(raw_path)

                print(f"  ✓  Raw image saved → {raw_path.name}  "
                      f"(model: {current_model}, prompt attempt: {prompt_attempt})")
                return raw_path

            except Exception as exc:
                if "blocked" in str(exc).lower() or "safety" in str(exc).lower():
                    print(f"  ⚠ {current_model} raised a safety error.")
                    continue
                raise

        if prompt_attempt < max_prompt_attempts:
            if master_summary and title and ollama_model:
                print(f"  ⚠ All models blocked prompt attempt {prompt_attempt} "
                      f"— regenerating prompt from master summary…")
                current_prompt = regenerate_image_prompt(
                    master_summary=master_summary,
                    title=title,
                    author=author,
                    model=ollama_model,
                    previous_prompt=current_prompt,
                    attempt=prompt_attempt + 1,
                )
                print(f"  → New prompt: {current_prompt[:120]}…")

                if prompt_path is not None:
                    prompt_path.write_text(current_prompt, encoding="utf-8")
                    print(f"  → Saved → {prompt_path.name}")
            else:
                print("  ⚠ Cannot regenerate prompt — master_summary or ollama_model not provided.")
                break

    raise RuntimeError(
        f"Imagen blocked all {max_prompt_attempts} prompt attempts across all models.\n"
        f"Last prompt tried: {current_prompt}\n"
        "Try running with --cover-image to supply your own cover."
    )


def enhance_raw_image(
    raw_path: Path,
    brightness: float = 1.25,
    contrast: float   = 1.15,
) -> Path:
    img = Image.open(raw_path).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img.save(raw_path, format="PNG")
    print(f"  → Enhanced brightness ×{brightness} / contrast ×{contrast}")
    return raw_path


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 – Cover compositing
# ══════════════════════════════════════════════════════════════════════════════

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    if Path(FALLBACK_FONT).exists():
        print("  ⚠  League Spartan not found — using fallback font.")
        return ImageFont.truetype(FALLBACK_FONT, size)
    print("  ⚠  No TTF font found — using PIL default (pixelated).")
    return ImageFont.load_default()


def _wrap_text(
    text: str, font: ImageFont.FreeTypeFont,
    max_width: int, draw: ImageDraw.ImageDraw,
) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def composite_cover(
    raw_image: Path, title: str, author: str,
    output_path: Path, target_w: int = COVER_W, target_h: int = COVER_H,
) -> None:
    img = Image.open(raw_image).convert("RGB")
    src_w, src_h = img.size

    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    full_img = img.crop((left, top, left + target_w, top + target_h))

    image_h = int(target_h * IMAGE_HEIGHT_RATIO)

    cover = full_img.copy()

    overlay = Image.new("RGBA", (target_w, target_h - image_h), TEXT_BAND_COLOR + (179,))
    cover = cover.convert("RGBA")
    cover.paste(overlay, (0, image_h), mask=overlay)
    cover = cover.convert("RGB")

    text_h = target_h - image_h
    text_y = image_h

    draw = ImageDraw.Draw(cover)

    max_text_w       = target_w - 2 * TEXT_PADDING
    available_text_h = text_h - TEXT_V_PADDING - TEXT_BOTTOM_PADDING

    title_lines = author_lines = []
    title_font  = author_font  = None

    def _line_height(line: str, font: ImageFont.FreeTypeFont) -> int:
        bb = draw.textbbox((0, 0), line, font=font)
        return bb[3] - bb[1]

    def _block_h(lines: list[str], font: ImageFont.FreeTypeFont, gap: int = 12) -> int:
        if not lines:
            return 0
        return sum(_line_height(l, font) for l in lines) + gap * (len(lines) - 1)

    for attempt in range(10):
        ts  = max(42, 85 - attempt * 7)
        as_ = max(28, 55 - attempt * 5)
        title_font  = _load_font(ts)
        author_font = _load_font(as_)
        title_lines  = _wrap_text(title.upper(), title_font, max_text_w, draw)
        author_lines = _wrap_text(author, author_font, max_text_w, draw)
        title_h  = _block_h(title_lines, title_font)
        author_h = _block_h(author_lines, author_font)
        total_block_h = title_h + 25 + author_h
        if total_block_h <= available_text_h:
            break

    block_y = text_y + TEXT_V_PADDING + (available_text_h - total_block_h) // 2

    def _draw_lines(
        lines: list[str],
        font: ImageFont.FreeTypeFont,
        start_y: int,
        gap: int = 12,
        color: tuple = TEXT_COLOR,
        shadow_color: tuple = TEXT_SHADOW_COLOR,
    ) -> int:
        y = start_y
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            lw   = bbox[2] - bbox[0]
            lh   = bbox[3] - bbox[1]
            x    = (target_w - lw) // 2
            draw.text(
                (x + TEXT_SHADOW_OFFSET, y + TEXT_SHADOW_OFFSET),
                line, font=font, fill=shadow_color,
            )
            draw.text((x, y), line, font=font, fill=color)
            y += lh + gap
        return y

    y_after_title = _draw_lines(title_lines, title_font, block_y)
    _draw_lines(author_lines, author_font, y_after_title + 25)

    imprint_font = _load_font(IMPRINT_FONT_SIZE)
    imp_bbox     = draw.textbbox((0, 0), IMPRINT_TEXT, font=imprint_font)
    imp_w        = imp_bbox[2] - imp_bbox[0]
    imp_h        = imp_bbox[3] - imp_bbox[1]
    imp_x        = (target_w - imp_w) // 2
    imp_y        = target_h - imp_h - IMPRINT_BOTTOM_GAP
    draw.text(
        (imp_x + TEXT_SHADOW_OFFSET, imp_y + TEXT_SHADOW_OFFSET),
        IMPRINT_TEXT, font=imprint_font, fill=IMPRINT_SHADOW,
    )
    draw.text((imp_x, imp_y), IMPRINT_TEXT, font=imprint_font, fill=IMPRINT_COLOR)

    cover.save(output_path, format="JPEG", quality=92)
    print(f"  ✓  Full-res cover → {output_path.name}  ({target_w}×{target_h}px)")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6b – Downscale cover_final.jpg to web size
# ══════════════════════════════════════════════════════════════════════════════

def downscale_cover_to_web(cover_path: Path, web_width: int = WEB_COVER_WIDTH) -> None:
    """
    Overwrite cover_path with a web-optimised version scaled to web_width px wide.
    The aspect ratio is preserved.  Called AFTER the EPUB has been built so the
    full-resolution image is safely embedded in the EPUB before we shrink it.
    """
    img = Image.open(cover_path).convert("RGB")
    orig_w, orig_h = img.size

    if orig_w <= web_width:
        print(f"  ℹ  Cover is already ≤ {web_width}px wide — no downscale needed.")
        return

    ratio    = web_width / orig_w
    web_h    = int(orig_h * ratio)
    img      = img.resize((web_width, web_h), Image.LANCZOS)
    img.save(cover_path, format="JPEG", quality=WEB_COVER_QUALITY, optimize=True)

    size_kb = cover_path.stat().st_size / 1024
    print(f"  ✓  Web cover → {cover_path.name}  "
          f"({web_width}×{web_h}px, {size_kb:.1f} KB)  "
          f"[was {orig_w}×{orig_h}px]")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 – Inject cover into EPUB  (EPUB 3)
# ══════════════════════════════════════════════════════════════════════════════

FRONTMATTER_SKIP_RE = re.compile(
    r"""
    cover | toc | table.?of.?contents | contents |
    copyright | copyrights | legal | licence | license |
    preface | pr[eé]face | foreword | avant.?propos |
    introduction | intro(?!duc) |
    dedication | d[eé]dicace | epigraph |
    about.?author | colophon | acknowledgement | acknowledgment |
    bibliography | index | glossary | appendix |
    back.?matter | front.?matter | half.?title | title.?page |
    note | notes(?!\.) | errata | permissions |
    a.?propos | page.?titre | titlepage | halftitle
    """,
    re.VERBOSE | re.IGNORECASE,
)


def inject_cover_into_epub(
    epub_path: Path, cover_jpg: Path, output_path: Path,
    title: str = "", author: str = "",
) -> None:
    """Rebuild the EPUB as a valid EPUB 3 file with the new cover."""
    import zipfile
    from lxml import etree

    src = epub.read_epub(str(epub_path))

    lang = "fr"
    try:
        lm = src.get_metadata("DC", "language")
        if lm:
            lang = lm[0][0]
    except Exception:
        pass

    uid = f"skookoo-{epub_path.stem}"

    src_id_to_item = {
        item.id: item
        for item in src.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    }
    spine_ids = [idref for idref, _ in src.spine]

    COLOPHON_RE = re.compile(
        r"""
        ebooks?\s+libres?\s+et\s+gratuits | ebooksgratuits\.com |
        gutenberg\.org | wikisource\.org | \bfeedbooks\b |
        [eé]dition\s+[eé]lectronique | libre\s+de\s+droits |
        conversion\s+informatique | [eé]laboration\s+de\s+ce\s+livre |
        [àa]\s+propos\s+de\s+cette\s+[eé]dition |
        produced\s+by | this\s+ebook\s+was\s+produced |
        project\s+gutenberg | transcribed\s+from |
        num[eé]ris[eé] | correcteurs?\s+b[eé]n[eé]voles |
        vous\s+avez\s+aim[eé]\s+ce\s+livre |
        nos\s+utilisateurs\s+ont\s+aussi |
        generation\.feedbooks\.com
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def _skip_by_name(item) -> bool:
        name = Path(item.get_name()).stem.lower()
        return bool(FRONTMATTER_SKIP_RE.search(name)) or \
               name in ("cover", "cover_page", "coverpage")

    def _skip_by_content(item) -> bool:
        try:
            raw = item.get_body_content()
            if not raw:
                return True
            plain = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="ignore"))
            return bool(COLOPHON_RE.search(plain))
        except Exception:
            return True

    def _lxml_ok(item) -> bool:
        try:
            from lxml import html as lh
            body = item.get_body_content()
            if not body or not body.strip():
                return False
            lh.document_fromstring(body)
            return True
        except Exception:
            return False

    chapter_items = []
    skipped_content = []
    for sid in spine_ids:
        if sid not in src_id_to_item:
            continue
        item = src_id_to_item[sid]
        if _skip_by_name(item):
            continue
        if _skip_by_content(item):
            skipped_content.append(item.get_name())
            continue
        if not _lxml_ok(item):
            continue
        chapter_items.append(item)

    if skipped_content:
        print(f"  ℹ  Removed {len(skipped_content)} colophon/boilerplate page(s):")
        for n in skipped_content:
            print(f"       – {n}")

    if not chapter_items:
        raise RuntimeError("No valid chapter content found after filtering.")

    asset_items = [
        item for item in src.get_items()
        if not isinstance(item, (epub.EpubHtml, epub.EpubNcx, epub.EpubNav))
        and item.get_type() != ebooklib.ITEM_DOCUMENT
    ]

    cover_jpg_bytes = cover_jpg.read_bytes()

    cover_xhtml = b"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="fr">
<head>
  <title>Couverture</title>
  <meta charset="utf-8"/>
  <style>
    html, body { margin: 0; padding: 0; background: #000; }
    img { display: block; width: 100%; height: auto; }
  </style>
</head>
<body epub:type="cover">
  <img src="../images/cover.jpg" alt="Couverture"/>
</body>
</html>"""

    title_xhtml = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}">
<head>
  <title>{title}</title>
  <meta charset="utf-8"/>
  <style>
    body {{
      margin: 0; padding: 0;
      background-color: #0f1420;
      font-family: Georgia, serif;
      text-align: center;
    }}
    .wrapper {{ padding: 10% 8%; }}
    .author {{
      font-size: 1.1em; font-weight: bold;
      letter-spacing: 0.25em; text-transform: uppercase;
      color: #a89060; margin-bottom: 3em;
    }}
    .title {{
      font-size: 2.4em; font-weight: bold;
      line-height: 1.25; color: #ffffff;
      margin-bottom: 3em; letter-spacing: 0.05em;
    }}
    .rule {{ width: 4em; height: 2px; background-color: #a89060; margin: 0 auto; }}
  </style>
</head>
<body epub:type="frontmatter titlepage">
  <div class="wrapper">
    <p class="author">{author}</p>
    <h1 class="title">{title}</h1>
    <div class="rule"></div>
  </div>
</body>
</html>""".encode("utf-8")

    imprint_xhtml = """<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="fr">
<head>
  <title>Mentions légales</title>
  <meta charset="utf-8"/>
  <style>
    body {
      margin: 0; padding: 0;
      background-color: #0f1420;
      font-family: Georgia, serif;
      color: #cccccc; font-size: 0.85em; line-height: 1.8;
    }
    .wrapper { padding: 15% 10%; }
    .publisher {
      font-size: 1.3em; font-weight: bold;
      letter-spacing: 0.2em; text-transform: uppercase;
      color: #a89060; margin-bottom: 0.3em;
    }
    .publisher-sub {
      font-size: 0.8em; letter-spacing: 0.15em;
      color: #888888; margin-bottom: 3em; text-transform: uppercase;
    }
    .rule { width: 3em; height: 1px; background-color: #a89060; margin: 2em 0; }
    .ai-notice { color: #aaaaaa; font-style: italic; font-size: 0.9em; line-height: 1.9; }
    .ai-notice strong { color: #c8aa70; font-style: normal; }
  </style>
</head>
<body epub:type="frontmatter">
  <div class="wrapper">
    <p class="publisher">&#201;ditions Skookoo</p>
    <p class="publisher-sub">Collection num&#233;rique &amp; patrimoniale</p>
    <div class="rule"></div>
    <p class="ai-notice">
      Cette &#233;dition a &#233;t&#233;
      <strong>constitu&#233;e, enrichie et mise en forme par une intelligence artificielle</strong>.
      La s&#233;lection du texte, la conception de la couverture,
      la r&#233;daction de la quatri&#232;me de couverture
      ainsi que la structuration de l&#8217;ouvrage ont &#233;t&#233; enti&#232;rement
      r&#233;alis&#233;es par des syst&#232;mes d&#8217;IA,
      sous la supervision des &#233;quipes &#233;ditoriales des
      <em>&#201;ditions Skookoo</em>.
    </p>
    <p class="ai-notice">
      Nous croyons que la technologie, lorsqu&#8217;elle est mise au service de la culture,
      peut rendre la litt&#233;rature plus accessible, plus vivante et plus belle.
      Ce livre en est la preuve.
    </p>
    <div class="rule"></div>
    <p style="color:#666666;font-size:0.8em;">
      &#169; &#201;ditions Skookoo &#8212; Tous droits r&#233;serv&#233;s.<br/>
      &#201;dition num&#233;rique.
      Toute reproduction interdite sans autorisation &#233;crite de l&#8217;&#233;diteur.
    </p>
  </div>
</body>
</html>""".encode("utf-8")

    front_pages = [
        ("OEBPS/text/cover.xhtml",    "application/xhtml+xml", cover_xhtml,   "cover-page"),
        ("OEBPS/text/titlepage.xhtml","application/xhtml+xml", title_xhtml,   "titlepage"),
        ("OEBPS/text/imprint.xhtml",  "application/xhtml+xml", imprint_xhtml, "imprint"),
    ]

    asset_slots = []
    asset_href_map: dict[str, str] = {}
    for item in asset_items:
        fname = Path(item.get_name()).name
        arc   = f"OEBPS/assets/{fname}"
        mid   = re.sub(r"[^a-zA-Z0-9_\-]", "_", fname)
        asset_slots.append((arc, item.media_type or "application/octet-stream",
                             item.get_content(), mid))
        asset_href_map[fname.lower()] = f"../assets/{fname}"

    def _rewrite_hrefs(content: bytes) -> bytes:
        try:
            text = content.decode("utf-8", errors="ignore")
            def _replacer(m: re.Match) -> str:
                attr  = m.group(1)
                quote = m.group(2)
                url   = m.group(3)
                fname = Path(url).name.lower()
                if fname in asset_href_map:
                    return f'{attr}{quote}{asset_href_map[fname]}{quote}'
                return m.group(0)
            text = re.sub(
                r'(src=|href=)(["\'])([^"\']+)\2',
                _replacer,
                text,
                flags=re.IGNORECASE,
            )
            return text.encode("utf-8")
        except Exception:
            return content

    chap_slots = []
    for i, item in enumerate(chapter_items):
        body = item.get_body_content()
        if body:
            body = _rewrite_hrefs(body)
        arc   = f"OEBPS/text/chap{i:04d}.xhtml"
        mid   = f"chap{i:04d}"
        label = (item.title or "").strip() or Path(item.get_name()).stem
        label = re.sub(r"[_\-]+", " ", label).strip()
        label = re.sub(r"^\d+\s*", "", label).strip().title() or f"Chapitre {i+1}"
        chap_slots.append((arc, "application/xhtml+xml", body, mid, label))

    cover_img_arc = "OEBPS/images/cover.jpg"

    def opf_xml() -> bytes:
        NS  = "http://www.idpf.org/2007/opf"
        DC  = "http://purl.org/dc/elements/1.1/"
        XML_NS = "http://www.w3.org/XML/1998/namespace"
        root = etree.Element(
            f"{{{NS}}}package",
            attrib={
                "version":           "3.0",
                "unique-identifier": "bookid",
                f"{{{XML_NS}}}lang": lang,
            },
            nsmap={None: NS, "dc": DC},
        )

        meta = etree.SubElement(root, f"{{{NS}}}metadata",
                                nsmap={"dc": DC, "opf": NS})
        etree.SubElement(meta, f"{{{DC}}}title").text    = title
        etree.SubElement(meta, f"{{{DC}}}creator").text  = author
        etree.SubElement(meta, f"{{{DC}}}language").text = lang
        etree.SubElement(meta, f"{{{DC}}}identifier",
                         attrib={"id": "bookid"}).text   = uid

        etree.SubElement(meta, f"{{{NS}}}meta",
                         attrib={"name": "cover", "content": "cover-img"})

        mf = etree.SubElement(root, f"{{{NS}}}manifest")

        def _item(mid, href, mt, **kw):
            etree.SubElement(mf, f"{{{NS}}}item",
                             attrib={"id": mid, "href": href,
                                     "media-type": mt, **kw})

        _item("nav", "text/nav.xhtml", "application/xhtml+xml",
              properties="nav")

        _item("cover-img", "images/cover.jpg", "image/jpeg",
              properties="cover-image")

        for arc, mt, _, mid in front_pages:
            href = arc.replace("OEBPS/", "")
            _item(mid, href, mt)

        for arc, mt, _, mid, _label in chap_slots:
            href = arc.replace("OEBPS/", "")
            _item(mid, href, mt)

        for arc, mt, _, mid in asset_slots:
            href = arc.replace("OEBPS/", "")
            _item(mid, href, mt)

        sp = etree.SubElement(root, f"{{{NS}}}spine")
        etree.SubElement(sp, f"{{{NS}}}itemref",
                         attrib={"idref": "cover-page", "linear": "no"})
        for _, _, _, mid in front_pages[1:]:
            etree.SubElement(sp, f"{{{NS}}}itemref", attrib={"idref": mid})
        for _, _, _, mid, _ in chap_slots:
            etree.SubElement(sp, f"{{{NS}}}itemref", attrib={"idref": mid})

        return etree.tostring(root, xml_declaration=True,
                              encoding="utf-8", pretty_print=True)

    def nav_xhtml() -> bytes:
        def _chap_href(arc: str) -> str:
            return Path(arc).name

        toc_items = "\n    ".join(
            f'<li><a href="{_chap_href(arc)}">{label}</a></li>'
            for arc, _, _, _, label in chap_slots
        )
        landmarks = (
            f'<li><a epub:type="cover"      href="cover.xhtml">Couverture</a></li>\n    '
            f'<li><a epub:type="titlepage"  href="titlepage.xhtml">Page de titre</a></li>\n    '
            f'<li><a epub:type="bodymatter" href="{_chap_href(chap_slots[0][0])}">'
            f'D&#233;but</a></li>'
            if chap_slots else ""
        )
        return f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}">
<head>
  <title>{title}</title>
  <meta charset="utf-8"/>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table des matières</h1>
    <ol>
    {toc_items}
    </ol>
  </nav>
  <nav epub:type="landmarks" hidden="">
    <ol>
    {landmarks}
    </ol>
  </nav>
</body>
</html>""".encode("utf-8")

    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )

        zf.writestr("META-INF/container.xml", """<?xml version='1.0' encoding='utf-8'?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""")

        zf.writestr("OEBPS/content.opf",      opf_xml())
        zf.writestr("OEBPS/text/nav.xhtml",   nav_xhtml())
        zf.writestr(cover_img_arc,             cover_jpg_bytes)

        for arc, _, content, _ in front_pages:
            zf.writestr(arc, content)

        for arc, _, content, _, _ in chap_slots:
            if content:
                zf.writestr(arc, content)

        for arc, _, content, _ in asset_slots:
            if content:
                zf.writestr(arc, content)

    print(f"  ✓  Final EPUB 3 → {output_path.name}  ({len(chap_slots)} chapter(s) in TOC)")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7 – Embed chunk summaries + write metadata.json
# ══════════════════════════════════════════════════════════════════════════════

def embed_texts(texts: list[str], model: str = OLLAMA_EMBED_MODEL) -> list[list[float]]:
    if not texts:
        return []

    vectors: list[list[float] | None] = [None] * len(texts)
    dim: int | None = None

    for i, text in enumerate(texts):
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": model, "input": text, "options": {"num_thread": 6}},
                    timeout=OLLAMA_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                if "embeddings" in data:
                    vec = data["embeddings"][0]
                elif "embedding" in data:
                    vec = data["embedding"]
                else:
                    raise ValueError(f"Unexpected embed response keys: {list(data.keys())}")

                vectors[i] = vec
                if dim is None:
                    dim = len(vec)
                break

            except Exception as exc:
                if attempt == RETRY_ATTEMPTS:
                    print(f"\n  ✗ Embedding chunk {i+1} permanently failed: {exc}")
                    break
                wait = RETRY_DELAY * attempt
                print(
                    f"\n  ⚠ Embed error (chunk {i+1}, attempt {attempt}/{RETRY_ATTEMPTS}): "
                    f"{type(exc).__name__} — retrying in {wait:.0f}s…"
                )
                time.sleep(wait)

    if dim is None:
        dim = 0
    for i, v in enumerate(vectors):
        if v is None:
            vectors[i] = [0.0] * dim

    return vectors  # type: ignore[return-value]


def compute_mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors or not vectors[0]:
        return []
    arr = np.array(vectors, dtype=np.float64)
    return arr.mean(axis=0).tolist()


def load_chunk_summaries_from_checkpoints(ckpt_dir: Path) -> list[str]:
    files = sorted(ckpt_dir.glob("chunk_*_summary.txt"))
    summaries = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        summaries.append(text)
    return summaries


def generate_embeddings(
    ckpt_dir: Path,
    embed_model: str,
    verbose: bool = True,
) -> tuple[list[list[float]], list[float]]:
    summaries = load_chunk_summaries_from_checkpoints(ckpt_dir)

    if not summaries:
        print("  ⚠  No chunk summaries found — skipping embedding.")
        return [], []

    valid   = [(i, s) for i, s in enumerate(summaries)
               if "[Résumé indisponible" not in s]
    skipped = len(summaries) - len(valid)

    if verbose:
        print(f"  → Embedding {len(valid)} chunk summary/summaries "
              f"({skipped} placeholder(s) skipped)…")

    if not valid:
        print("  ⚠  All summaries are placeholders — no embeddings generated.")
        return [], []

    indices, texts = zip(*valid)

    chunk_vectors_valid: list[list[float]] = []
    for j, (orig_i, text) in enumerate(zip(indices, texts)):
        if verbose:
            print(
                f"  → Embedding chunk {j+1}/{len(texts)} "
                f"(original index {orig_i+1})…",
                flush=True,
            )
        t0   = time.perf_counter()
        vecs = embed_texts([text], model=embed_model)
        elapsed = time.perf_counter() - t0
        chunk_vectors_valid.append(vecs[0])
        if verbose:
            print(f"     ✓ Done in {fmt_duration(elapsed)} (dim={len(vecs[0])}).")

    mean_vec = compute_mean_vector(chunk_vectors_valid)

    if verbose:
        print(f"  ✓  Embeddings done. "
              f"Dimension: {len(mean_vec) if mean_vec else 0}. "
              f"Mean vector computed from {len(chunk_vectors_valid)} chunk(s).")

    dim = len(mean_vec) if mean_vec else 0
    all_vectors: list[list[float]] = [[0.0] * dim] * len(summaries)
    valid_iter = iter(chunk_vectors_valid)
    for orig_i in indices:
        all_vectors[orig_i] = next(valid_iter)

    return all_vectors, mean_vec


def write_metadata_json(
    out_dir:        Path,
    epub_path:      Path,
    stem:           str,
    title:          str,
    author:         str,
    blurb:          str,
    master_summary: str,
    mean_vector:    list[float],
    stats:          dict | None = None,
) -> Path:
    cover_final = out_dir / f"{stem}_cover_final.jpg"
    epub_final  = out_dir / f"{stem}_final.epub"

    cover_rel = (
        f"./{cover_final.name}" if cover_final.exists()
        else "./cover_raw.png"
    )
    epub_rel = (
        f"./{epub_final.name}" if epub_final.exists()
        else f"./{epub_path.name}"
    )

    metadata: dict = {
        "title":         title,
        "author":        author,
        "blurb":         blurb,
        "summary":       master_summary,
        "cover":         cover_rel,
        "epub":          epub_rel,
        "embed_model":   OLLAMA_EMBED_MODEL,
        "chunk_count":   0,
        "vector_dim":    len(mean_vector) if mean_vector else 0,
        "mean_vector":   mean_vector,
    }

    if stats is not None:
        metadata["stats"] = stats

    out_path = out_dir / "metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓  metadata.json → {out_path.name}  ({size_kb:.1f} KB)")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Per-book orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def process_epub(epub_path: Path, output_root: Path, args: argparse.Namespace) -> None:
    stem    = epub_path.stem
    out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "chunk_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    timings: dict[str, float] = {}
    t_book  = time.perf_counter()

    section(f"📖  {epub_path.name}")

    # ── Phase 1 : Extract text ────────────────────────────────────────────────
    print("\n[1/8] Extracting body text from EPUB…")
    t0 = time.perf_counter()
    body_text, title, author, word_count = load_epub_text_and_meta(epub_path)
    timings["1 · Extract text"] = time.perf_counter() - t0
    print(f"  Title      : {title}")
    print(f"  Author     : {author}")
    print(f"  Body words : {word_count:,}  (front/back matter excluded)")
    print(f"  ⏱  {fmt_duration(timings['1 · Extract text'])}")

    # ── Phase 1b : Text Statistics ────────────────────────────────────────────
    stats_path = out_dir / f"{stem}_stats.json"

    if stats_path.exists():
        print(f"\n[1b/8] Text statistics: loaded from checkpoint.")
        with open(stats_path, encoding="utf-8") as f:
            text_stats = json.load(f)
        timings["1b· Text statistics"] = 0.0
        _print_stats(text_stats)
    else:
        print(f"\n[1b/8] Computing French text statistics…")
        t0 = time.perf_counter()
        text_stats = compute_text_statistics(body_text, verbose=True)
        timings["1b· Text statistics"] = time.perf_counter() - t0

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(text_stats, f, ensure_ascii=False, indent=2)
        size_kb = stats_path.stat().st_size / 1024
        print(f"  ✓  Stats checkpoint → {stats_path.name}  ({size_kb:.1f} KB)")
        print(f"  ⏱  {fmt_duration(timings['1b· Text statistics'])}")

    # ── Phase 2 : Hierarchical summarisation ─────────────────────────────────
    summary_path = out_dir / f"{stem}_master_summary.txt"

    if summary_path.exists():
        print(f"\n[2/8] Master summary: loaded from checkpoint.")
        master_summary = summary_path.read_text(encoding="utf-8")
        timings["2 · Hierarchical summarise"] = 0.0
    else:
        print(f"\n[2/8] Hierarchical summarisation…")
        print(f"  Chunk model : {OLLAMA_CHUNK_MODEL}")
        print(f"  Final model : {OLLAMA_MODEL}")
        t0 = time.perf_counter()
        master_summary = hierarchical_summarise(
            full_text=body_text,
            chunk_words=args.chunk_words,
            chunk_model=OLLAMA_CHUNK_MODEL,
            final_model=OLLAMA_MODEL,
            ckpt_dir=ckpt_dir,
            verbose=True,
        )
        timings["2 · Hierarchical summarise"] = time.perf_counter() - t0
        summary_path.write_text(master_summary, encoding="utf-8")
        print(f"  ✓  Master summary → {summary_path.name}")
        print(f"  ⏱  {fmt_duration(timings['2 · Hierarchical summarise'])}")

    # ── Phase 3 : Image prompt + back-cover blurb ────────────────────────────
    print(f"\n[3/8] Building image prompt + back-cover blurb ({OLLAMA_MODEL})…")

    prompt_path = out_dir / f"{stem}_image_prompt.txt"

    if prompt_path.exists():
        print("  → Image prompt: loaded from checkpoint.")
        image_prompt = prompt_path.read_text(encoding="utf-8")
        timings["3a· Image prompt"] = 0.0
    else:
        print("  → Image prompt…", flush=True)
        t0 = time.perf_counter()
        image_prompt = build_image_prompt(master_summary, title, author, OLLAMA_MODEL)
        timings["3a· Image prompt"] = time.perf_counter() - t0
        print(f"     ✓ Done in {fmt_duration(timings['3a· Image prompt'])}.")
        prompt_path.write_text(image_prompt, encoding="utf-8")

    print(f"  Image prompt: {image_prompt[:120]}…")

    blurb_path = out_dir / f"{stem}_quatrieme_de_couverture.txt"

    if blurb_path.exists():
        print("  → Back-cover blurb: loaded from checkpoint.")
        raw_blurb_file = blurb_path.read_text(encoding="utf-8")
        separator = "─" * 60
        if separator in raw_blurb_file:
            blurb = raw_blurb_file.split(separator, 1)[-1].strip()
        else:
            blurb = raw_blurb_file.strip()
        timings["3b· Back-cover blurb"] = 0.0
    else:
        print("  → Back-cover blurb (+ self-correction)…", flush=True)
        t0 = time.perf_counter()
        blurb = build_back_cover(master_summary, title, author, OLLAMA_MODEL)
        timings["3b· Back-cover blurb"] = time.perf_counter() - t0
        print(f"     ✓ Done in {fmt_duration(timings['3b· Back-cover blurb'])}.")
        blurb_path.write_text(
            f"Titre  : {title}\nAuteur : {author}\n\n{'─'*60}\n\n{blurb}\n",
            encoding="utf-8",
        )

    print(f"  ✓  Blurb → {blurb_path.name}")
    print(f"  ⏱  {fmt_duration(timings['3a· Image prompt'] + timings['3b· Back-cover blurb'])}")

    if args.only_blurb:
        print("\n  --only-blurb set: skipping image generation and EPUB rebuild.")
        print("\n[8/8] Generating embeddings + writing metadata.json…")
        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists():
            print(f"  → metadata.json already exists — skipping embeddings.")
            timings["8 · Embed + metadata"] = 0.0
        else:
            t0 = time.perf_counter()
            chunk_vectors, mean_vector = generate_embeddings(
                ckpt_dir=ckpt_dir,
                embed_model=args.embed_model,
                verbose=True,
            )
            write_metadata_json(
                out_dir=out_dir,
                epub_path=epub_path,
                stem=stem,
                title=title,
                author=author,
                blurb=blurb,
                master_summary=master_summary,
                mean_vector=mean_vector,
                stats=text_stats,
            )
            timings["8 · Embed + metadata"] = time.perf_counter() - t0
            print(f"  ⏱  {fmt_duration(timings['8 · Embed + metadata'])}")
        _print_timings(timings, t_book, out_dir, stem)
        return

    # ── Phase 4 : Image generation (Imagen) ──────────────────────────────────
    raw_cover: Path | None = args.cover_image
    cached_raw = out_dir / "cover_raw.png"

    if raw_cover is None and cached_raw.exists():
        print(f"\n[4/8] cover_raw.png found in output folder — skipping image generation.")
        raw_cover = cached_raw
        timings["4 · Image gen (Imagen)"] = 0.0
    elif raw_cover is None:
        print("\n  → Freeing Ollama models before image generation…")
        for m in set([OLLAMA_CHUNK_MODEL, OLLAMA_MODEL]):
            ollama_unload(m)
        time.sleep(2)

        print(f"\n[4/8] Generating cover image with Imagen…")
        t0 = time.perf_counter()
        raw_cover = generate_image_imagen(
            prompt=image_prompt,
            out_dir=out_dir,
            model=args.imagen_model,
            aspect_ratio=args.imagen_aspect,
            master_summary=master_summary,
            title=title,
            author=author,
            ollama_model=OLLAMA_MODEL,
            prompt_path=prompt_path,
        )
        raw_cover = enhance_raw_image(raw_cover)
        timings["4 · Image gen (Imagen)"] = time.perf_counter() - t0
        print(f"  ⏱  {fmt_duration(timings['4 · Image gen (Imagen)'])}")
    else:
        print(f"\n[4/8] Using provided cover image: {raw_cover}")
        timings["4 · Image gen (Imagen)"] = 0.0

    # ── Phase 5 : Composite cover ─────────────────────────────────────────────
    print("\n[5/8] Compositing cover…")
    t0 = time.perf_counter()
    final_cover = out_dir / f"{stem}_cover_final.jpg"
    composite_cover(raw_cover, title, author, final_cover, args.width, args.height)
    timings["5 · Composite cover"] = time.perf_counter() - t0
    print(f"  ⏱  {fmt_duration(timings['5 · Composite cover'])}")

    # ── Phase 6 : Inject cover into the original EPUB ────────────────────────
    print("\n[6/8] Rebuilding EPUB 3 with new cover…")
    t0 = time.perf_counter()
    final_epub = out_dir / f"{stem}_final.epub"
    inject_cover_into_epub(epub_path, final_cover, final_epub, title=title, author=author)
    timings["6 · EPUB rebuild"] = time.perf_counter() - t0
    print(f"  ⏱  {fmt_duration(timings['6 · EPUB rebuild'])}")

    # ── Phase 6b : Downscale cover_final.jpg to web size ─────────────────────
    print(f"\n[6b/8] Downscaling cover to {WEB_COVER_WIDTH}px web version…")
    t0 = time.perf_counter()
    downscale_cover_to_web(final_cover, web_width=args.web_cover_width)
    timings["6b· Web cover downscale"] = time.perf_counter() - t0
    print(f"  ⏱  {fmt_duration(timings['6b· Web cover downscale'])}")

    # ── Phase 7 : Embed chunk summaries + write metadata.json ────────────────
    print("\n[7/8] Generating embeddings…")
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists():
        print(f"  → metadata.json already exists — skipping embeddings.")
        timings["7 · Embed"] = 0.0
    else:
        t0 = time.perf_counter()
        chunk_vectors, mean_vector = generate_embeddings(
            ckpt_dir=ckpt_dir,
            embed_model=args.embed_model,
            verbose=True,
        )
        timings["7 · Embed"] = time.perf_counter() - t0
        print(f"  ⏱  {fmt_duration(timings['7 · Embed'])}")

        print("\n[8/8] Writing metadata.json…")
        t0 = time.perf_counter()
        write_metadata_json(
            out_dir=out_dir,
            epub_path=epub_path,
            stem=stem,
            title=title,
            author=author,
            blurb=blurb,
            master_summary=master_summary,
            mean_vector=mean_vector,
            stats=text_stats,
        )
        timings["8 · Metadata"] = time.perf_counter() - t0
        print(f"  ⏱  {fmt_duration(timings['8 · Metadata'])}")

    _print_timings(timings, t_book, out_dir, stem)


def _print_timings(
    timings: dict[str, float], t_start: float, out_dir: Path, stem: str
) -> None:
    total = time.perf_counter() - t_start
    col   = max((len(k) for k in timings), default=10) + 2

    section("Timings")
    for step, elapsed in timings.items():
        bar_units = round((elapsed / total) * 20) if total > 0 else 0
        bar       = "█" * bar_units + "░" * (20 - bar_units)
        print(f"  {step:<{col}} {bar}  {fmt_duration(elapsed)}")
    print(f"{'─'*60}")
    print(f"  {'Total':<{col}} {'─'*20}  {fmt_duration(total)}")
    print(f"{'─'*60}")

    print("\n  Output files:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    • {f.name:<50}  {size_kb:>7.1f} KB")
    ckpt_dir = out_dir / "chunk_checkpoints"
    if ckpt_dir.exists():
        n = len(list(ckpt_dir.iterdir()))
        print(f"    • chunk_checkpoints/  ({n} file(s))")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EPUB summarise → cover → inject → embed → metadata pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input_folder",  type=Path,
                   help="Folder containing input .epub files.")
    p.add_argument("output_folder", type=Path,
                   help="Root output folder (one sub-folder created per book).")

    p.add_argument("-r", "--recursive", action="store_true",
                   help="Search input_folder recursively for .epub files.")

    # Ollama
    p.add_argument("--model",       default=OLLAMA_MODEL,
                   help=f"Ollama model for synthesis/blurb (default: {OLLAMA_MODEL}).")
    p.add_argument("--chunk-model", default=OLLAMA_CHUNK_MODEL,
                   help=f"Ollama model for chunk summarisation "
                        f"(default: {OLLAMA_CHUNK_MODEL}).")
    p.add_argument("--chunk-words", type=int, default=CHUNK_WORDS,
                   help=f"Words per chunk (default: {CHUNK_WORDS}).")
    p.add_argument("--embed-model", default=OLLAMA_EMBED_MODEL,
                   help=f"Ollama embedding model for chunk summaries "
                        f"(default: {OLLAMA_EMBED_MODEL}).")

    # Imagen
    p.add_argument("--imagen-model",  default=IMAGEN_MODEL,
                   help=f"Imagen model ID (default: {IMAGEN_MODEL}).")
    p.add_argument("--imagen-aspect", default=IMAGEN_ASPECT_RATIO,
                   choices=["1:1", "3:4", "4:3", "9:16", "16:9"],
                   help=f"Output aspect ratio (default: {IMAGEN_ASPECT_RATIO}). "
                        "Use 3:4 or 9:16 for portrait book covers.")

    # Cover dimensions
    p.add_argument("--width",  type=int, default=COVER_W,
                   help=f"Final cover width px (default: {COVER_W}).")
    p.add_argument("--height", type=int, default=COVER_H,
                   help=f"Final cover height px (default: {COVER_H}).")

    # Web cover
    p.add_argument("--web-cover-width", type=int, default=WEB_COVER_WIDTH,
                   help=f"Width in px of the web-optimised cover_final.jpg "
                        f"(default: {WEB_COVER_WIDTH}). "
                        "Overrides the full-res file AFTER the EPUB is built.")

    # Shortcuts
    p.add_argument("--cover-image", type=Path, default=None,
                   help="Use a pre-made cover PNG instead of calling Imagen.")
    p.add_argument("--only-blurb",  action="store_true",
                   help="Stop after generating the blurb (no image, no EPUB rebuild). "
                        "Embeddings and metadata.json are still generated.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    global OLLAMA_MODEL, OLLAMA_CHUNK_MODEL, OLLAMA_EMBED_MODEL
    OLLAMA_MODEL       = args.model
    OLLAMA_CHUNK_MODEL = args.chunk_model
    OLLAMA_EMBED_MODEL = args.embed_model

    if not args.input_folder.is_dir():
        sys.exit(f"❌  Not a directory: {args.input_folder}")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    glob_fn = args.input_folder.rglob if args.recursive else args.input_folder.glob
    epubs   = sorted(glob_fn("*.epub"))
    if not epubs:
        sys.exit(
            f"❌  No .epub files found in {args.input_folder}"
            + (" (try -r)" if not args.recursive else "")
        )

    section(f"EPUB Pipeline  —  {len(epubs)} book(s) found")
    print(f"  Input       : {args.input_folder.resolve()}")
    print(f"  Output      : {args.output_folder.resolve()}")
    print(f"  Text models : {OLLAMA_CHUNK_MODEL} (chunk)  /  {OLLAMA_MODEL} (final)")
    print(f"  Embed model : {OLLAMA_EMBED_MODEL}")
    print(f"  Web cover   : {args.web_cover_width}px wide")

    check_ollama()

    if not args.only_blurb and args.cover_image is None:
        check_google_key()

    t_all = time.perf_counter()
    for i, ep in enumerate(epubs, 1):
        print(f"\n{'═'*60}")
        print(f"  Book {i}/{len(epubs)}")
        print(f"{'═'*60}")
        try:
            process_epub(ep, args.output_folder, args)
        except KeyboardInterrupt:
            print("\n\n  ⚠  Interrupted by user.")
            sys.exit(1)
        except Exception as exc:
            print(f"\n  ✗  Failed to process {ep.name}: {exc}")
            import traceback; traceback.print_exc()
            print("  → Continuing with next book…\n")

    section(f"🎉  All done!  Total: {fmt_duration(time.perf_counter() - t_all)}")
    print(f"  Results in: {args.output_folder.resolve()}\n")


if __name__ == "__main__":
    main()