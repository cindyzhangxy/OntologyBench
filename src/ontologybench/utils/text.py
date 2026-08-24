# ontologybench/utils/text.py

import re
from typing import List, Tuple
import Levenshtein


# ---------------------------------------------------------
# Basic text normalization
# ---------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(normalize(a).split())
    b_tokens = set(normalize(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def levenshtein_similarity(a: str, b: str) -> float:
    a_p, b_p = normalize(a), normalize(b)
    if not a_p or not b_p:
        return 0.0
    dist = Levenshtein.distance(a_p, b_p)
    max_len = max(len(a_p), len(b_p))
    return 1 - (dist / max_len) if max_len > 0 else 0.0


# ---------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------

def cleanup_masked_text(text: str) -> str:
    """Fix doubled determiners, sentence starts, and redundant spacing."""
    if not isinstance(text, str):
        return text

    # Remove duplicated determiners
    text = re.sub(r"(?i)\bthe the phenotype\b", "the phenotype", text)
    text = re.sub(r"(?i)\bthe the disorder\b", "the disorder", text)
    text = re.sub(r"(?i)\bthe the gene\b", "the gene", text)

    # Normalize "so-called"
    text = re.sub(
        r"(?i)\b(?:the\s+)?so[-\s]?called\s+(the phenotype|the disorder|the gene)\b",
        r"\1",
        text
    )

    # Capitalize entity markers at sentence start
    text = re.sub(
        r"(?i)(^|[\.!?]\s+)(the phenotype|the disorder|the gene)",
        lambda m: m.group(1) + m.group(2).capitalize(),
        text
    )

    return re.sub(r"\s{2,}", " ", text).strip()

def strip_type_suffix(alias: str) -> str:
    """
    Remove trailing 'type X' patterns from aliases.
    Matches:
      - type I, type II, type 1, type 2A, type-1, type_III
    """
    pattern = r"\s*type[\s\-_]*[ivx0-9]+$"
    return re.sub(pattern, "", alias, flags=re.IGNORECASE).strip()



def mask_exact_alias(alias: str, text: str, replacement="the entity") -> Tuple[str, bool]:
    """
    Mask:
       (1) the full alias including any 'type X' suffix
       (2) the core alias without the 'type X' suffix

    Does NOT mask standalone 'type X' unless it appears as part of the alias.
    """

    if not isinstance(alias, str) or not isinstance(text, str):
        return text, False

    original_text = text
    masked = text
    any_mask = False

    # --------------------------
    # 1. Mask full alias FIRST
    # --------------------------
    alias_full = alias.strip()
    if alias_full:
        # Escape special characters and enforce word boundaries
        pattern_full = r"(?i)\b" + re.escape(alias_full) + r"\b"
        masked_new, n_full = re.subn(pattern_full, replacement, masked)
        if n_full > 0:
            masked = masked_new
            any_mask = True

    # --------------------------
    # 2. Mask core alias (alias minus 'type X')
    # --------------------------
    alias_core = strip_type_suffix(alias_full)
    if alias_core and alias_core != alias_full:
        # Token-aware flexible pattern
        alias_tokens = re.split(r"[\s\-/]+", alias_core)
        alias_tokens = [t for t in alias_tokens if t]

        if alias_tokens:
            flexible = r"[\s\-/]*".join(map(re.escape, alias_tokens))
            pattern_core = r"(?i)\b" + flexible + r"\b"

            masked_new, n_core = re.subn(pattern_core, replacement, masked)
            if n_core > 0:
                masked = masked_new
                any_mask = True

    # --------------------------
    # 3. Cleanup final text
    # --------------------------
    masked = cleanup_masked_text(masked)

    return masked, any_mask


def mask_all_aliases(
    aliases: List[str],
    text: str,
    replacement="the entity"
) -> Tuple[str, bool]:
    """
    Mask ALL aliases appearing in the text using the same robust logic as mask_exact_alias.

    Ensures:
      • No alias leaks into the final document.
      • Handles punctuation / whitespace variations.
      • Returns one unified masked definition per concept.

    Returns:
        masked_text, any_alias_masked
    """
    if not isinstance(text, str):
        return text, False
    if not isinstance(aliases, (list, tuple)):
        return text, False

    masked_text = text
    any_mask = False

    for alias in aliases:
        if not isinstance(alias, str):
            continue
        alias = alias.strip()
        if not alias:
            continue

        alias_tokens = re.split(r"[\s\-/]+", alias)
        alias_tokens = [t for t in alias_tokens if t]
        if not alias_tokens:
            continue

        # Build flexible alias pattern
        flexible = r"[\s\-/]*".join(map(re.escape, alias_tokens))
        pattern = r"(?i)\b" + flexible + r"\b"

        # Replace all occurrences
        masked_text_new, n = re.subn(pattern, replacement, masked_text)
        if n > 0:
            any_mask = True
            masked_text = masked_text_new

    masked_text = cleanup_masked_text(masked_text)
    return masked_text, any_mask


# ---------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------

def join_with_and(items: List[str]) -> str:
    """Turn list into human-readable English phrase."""
    if not items:
        return ""
    items = [str(x).strip() for x in items if str(x).strip()]
    n = len(items)
    if n == 0:
        return ""
    if n == 1:
        return items[0]
    if n == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
