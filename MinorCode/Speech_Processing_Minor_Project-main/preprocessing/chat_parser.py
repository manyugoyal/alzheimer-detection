"""
chat_parser.py
==============
Parses DementiaBank Pitt Corpus CHAT (.cha) files to extract clean patient
(*PAR:) speech, and computes a suite of interpretable linguistic features.

CHAT Format Overview
---------------------
*PAR:   patient utterance.
*INV:   interviewer utterance (ignored).
%xxx:   tier lines (ignored).
@Header lines (ignored).
Multi-line utterances continue until a new * speaker starts.

Codes stripped during cleaning
-------------------------------
[/]   repetition marker
[//]  retracing marker
[?]   uncertain transcription
[*]   error marker
[+ ...] postcodes
&word unintelligible/filler fragments
+...  trailing off
<...> grouped text (angle brackets removed, text kept)
(.)   filled / unfilled pauses
xxx   transcriber insertion of unintelligible
0word omitted morphemes
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal CHAT code cleaning patterns
# ---------------------------------------------------------------------------

# Compiled once at module import for efficiency
_CHAT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Remove % tier lines (handled at line level, but kept as safety net)
    (re.compile(r"^%.*$", re.MULTILINE), ""),
    # Remove @ header lines
    (re.compile(r"^@.*$", re.MULTILINE), ""),
    # Remove [/] repetition markers
    (re.compile(r"\[/+\]"), ""),
    # Remove [?] uncertain transcription
    (re.compile(r"\[\?\]"), ""),
    # Remove [*] error markers
    (re.compile(r"\[\*\]"), ""),
    # Remove [+ ...] postcodes
    (re.compile(r"\[\+[^\]]*\]"), ""),
    # Remove [= ...] explanations
    (re.compile(r"\[=[^\]]*\]"), ""),
    # Remove [<] and [>] overlapping speech markers
    (re.compile(r"\[<[^\]]*\]"), ""),
    (re.compile(r"\[>[^\]]*\]"), ""),
    # Remove all remaining bracketed codes
    (re.compile(r"\[[^\]]*\]"), ""),
    # Remove & fillers (e.g., &uh, &-um)
    (re.compile(r"&[-\w]+"), ""),
    # Remove +... trailing-off markers
    (re.compile(r"\+\.\.\."), ""),
    # Remove +/. and similar continuation markers
    (re.compile(r"\+[/\.,!?]+"), ""),
    # Remove <...> angle brackets but keep content inside
    (re.compile(r"<([^>]*)>"), r"\1"),
    # Remove (.) pause markers
    (re.compile(r"\(\.\.\.\)|\(\.\.\)|\(\.\)"), ""),
    # Remove numeric repetition counts like [x 3]
    (re.compile(r"\[x\s+\d+\]"), ""),
    # Remove CHAT special characters: 0 (omitted words prefix), ^ caret
    (re.compile(r"\b0\w+"), ""),
    # Remove \x15 CHAT timing code markers and content between them
    (re.compile(r"\x15\d+_\d+\x15"), ""),
    # Remove trailing terminal markers  . ! ? (kept for sentence boundaries)
    # Actually keep them — they help with tokenisation
    # Remove double/triple spaces
    (re.compile(r" {2,}"), " "),
]


def _clean_chat_text(raw_text: str) -> str:
    """
    Apply all CHAT cleaning patterns to a raw utterance string.

    Parameters
    ----------
    raw_text : str
        A single raw CHAT utterance (already extracted from *PAR: line).

    Returns
    -------
    str
        Clean plain-text utterance with CHAT codes removed.
    """
    text = raw_text.strip()
    for pattern, replacement in _CHAT_PATTERNS:
        text = pattern.sub(replacement, text)
    # Final pass: collapse whitespace
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# CHAT file parser
# ---------------------------------------------------------------------------

def parse_chat_file(filepath: str) -> str:
    """
    Parse a DementiaBank CHAT (.cha) file and extract the complete
    patient (*PAR:) speech as a single clean plain-text string.

    Algorithm
    ---------
    1. Read the file with UTF-8 encoding (fall back to latin-1).
    2. Join continuation lines (lines starting with tab) to their parent.
    3. Collect all *PAR: speaker turns into a list of raw utterances.
    4. Clean each utterance using :func:`_clean_chat_text`.
    5. Concatenate all clean utterances separated by a space.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .cha file.

    Returns
    -------
    str
        Complete clean patient speech as a single string.
        Returns an empty string if no *PAR: turns are found.

    Raises
    ------
    FileNotFoundError
        If the .cha file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CHAT file not found: {filepath}")

    # Try UTF-8 first, fall back to latin-1
    try:
        raw_content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_content = path.read_text(encoding="latin-1")
        logger.debug("Fell back to latin-1 encoding for: %s", filepath)

    # Step 1: Normalise line endings
    raw_content = raw_content.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2: Join continuation lines (tab-indented) to their parent utterance
    # CHAT uses a tab character to indicate continuation of the previous line
    lines: List[str] = []
    for raw_line in raw_content.split("\n"):
        if raw_line.startswith("\t") and lines:
            # Append continuation text (strip leading tab) to last line
            lines[-1] = lines[-1] + " " + raw_line.strip()
        else:
            lines.append(raw_line)

    # Step 3: Collect *PAR: turns
    par_utterances: List[str] = []
    par_pattern = re.compile(r"^\*PAR:\s*(.+)$")

    for line in lines:
        match = par_pattern.match(line.strip())
        if match:
            raw_utterance = match.group(1)
            par_utterances.append(raw_utterance)

    if not par_utterances:
        logger.warning("No *PAR: utterances found in: %s", filepath)
        return ""

    # Step 4: Clean utterances
    clean_utterances = [_clean_chat_text(utt) for utt in par_utterances]

    # Remove empty strings after cleaning
    clean_utterances = [utt for utt in clean_utterances if utt.strip()]

    # Step 5: Join into one string
    patient_speech = " ".join(clean_utterances)
    logger.debug(
        "Parsed %d PAR utterances from %s → %d chars",
        len(par_utterances),
        path.name,
        len(patient_speech),
    )
    return patient_speech


def parse_chat_file_to_utterances(filepath: str) -> List[str]:
    """
    Parse a CHAT file and return a list of individual clean patient utterances
    (one string per *PAR: turn) instead of a single concatenated string.

    Useful for computing turn-level statistics like mean utterance length.

    Parameters
    ----------
    filepath : str
        Path to the .cha file.

    Returns
    -------
    List[str]
        List of individual clean patient utterance strings.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CHAT file not found: {filepath}")

    try:
        raw_content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_content = path.read_text(encoding="latin-1")

    raw_content = raw_content.replace("\r\n", "\n").replace("\r", "\n")

    lines: List[str] = []
    for raw_line in raw_content.split("\n"):
        if raw_line.startswith("\t") and lines:
            lines[-1] = lines[-1] + " " + raw_line.strip()
        else:
            lines.append(raw_line)

    par_pattern = re.compile(r"^\*PAR:\s*(.+)$")
    utterances: List[str] = []

    for line in lines:
        match = par_pattern.match(line.strip())
        if match:
            clean = _clean_chat_text(match.group(1))
            if clean.strip():
                utterances.append(clean)

    return utterances


# ---------------------------------------------------------------------------
# Linguistic feature extraction
# ---------------------------------------------------------------------------

# POS tagging patterns — basic heuristic sets used when NLTK is unavailable
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "up",
    "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "not", "only", "same", "than", "too", "very", "just",
    "it", "its", "he", "she", "they", "we", "i", "you", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "whose",
}

_COMMON_NOUNS = {
    "picture", "picture", "girl", "boy", "woman", "man", "child", "children",
    "cookie", "jar", "water", "sink", "kitchen", "stool", "window", "curtain",
    "mother", "lady", "dishes", "plate", "cup", "cloth", "dishcloth", "floor",
    "water", "outside", "yard", "garden", "tree", "house", "cat", "dog",
    "bird", "chair", "table", "door", "hand", "hand", "thing", "something",
    "everything", "nothing", "one", "two", "people", "person", "time",
    "way", "day", "year", "work", "part", "place", "case", "week", "company",
    "system", "program", "question", "government", "number", "night",
    "point", "home", "water", "room", "mother", "area", "money", "story",
    "fact", "month", "lot", "right", "study", "book", "eye", "job",
    "word", "world", "side", "family", "head", "church", "others", "heart",
    "air", "night", "war", "city", "hospital", "back", "top", "face",
    "end", "service", "state", "school", "kind", "light", "ground", "fall",
}

_COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "say", "go", "get", "make", "know", "think",
    "take", "see", "come", "look", "want", "give", "use", "find", "tell",
    "ask", "seem", "feel", "try", "leave", "call", "put", "keep", "let",
    "begin", "show", "hear", "play", "run", "move", "live", "believe",
    "hold", "bring", "happen", "write", "provide", "sit", "stand", "lose",
    "reach", "kill", "remain", "suggest", "raise", "pass", "sell", "require",
    "report", "open", "pick", "remember", "turn", "learn", "change", "grow",
    "fall", "draw", "set", "done", "wash", "overflow", "climb", "reach",
    "hand", "talk", "mean", "help", "eat", "drink", "drive", "send", "read",
}


def extract_linguistic_features(
    transcript_path: Optional[str] = None,
    text: Optional[str] = None,
    utterances: Optional[List[str]] = None,
    filler_words: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute a set of interpretable linguistic features from patient speech.

    Exactly one of ``transcript_path``, ``text``, or ``utterances`` must be
    provided as the speech source.

    Features computed
    -----------------
    - type_token_ratio (TTR): |unique_words| / |total_words|
    - mean_utterance_length: mean number of words per *PAR: turn
    - total_utterances: number of *PAR: turns
    - total_words: total word count
    - filler_count: occurrences of filler words (uh, um, er, hmm, etc.)
    - filler_rate: filler_count / total_words
    - lexical_density: content_words / total_words
      where content_words = total_words - function_words
    - unique_noun_count: approximate count of unique nouns
    - unique_verb_count: approximate count of unique verbs
    - brunet_w_index: Brunét's W index (vocabulary richness, lower = richer)
    - honore_r_statistic: Honoré's R (higher = richer vocabulary)

    Parameters
    ----------
    transcript_path : str, optional
        Path to a .cha file. If provided, the file is parsed first.
    text : str, optional
        Pre-parsed plain text string of patient speech.
    utterances : list of str, optional
        Pre-split list of individual utterances.
    filler_words : list of str, optional
        Custom list of filler words to count. Defaults to
        ['uh', 'um', 'er', 'hmm', 'hm', 'ah'].

    Returns
    -------
    dict
        Mapping from feature name (str) to feature value (float).
    """
    if filler_words is None:
        filler_words = ["uh", "um", "er", "hmm", "hm", "ah", "like", "well"]

    # Resolve source
    if transcript_path is not None:
        utterances = parse_chat_file_to_utterances(transcript_path)
        text = " ".join(utterances)
    elif text is not None and utterances is None:
        # Approximate utterance split by sentence-ending punctuation
        utterances = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    elif utterances is not None and text is None:
        text = " ".join(utterances)
    elif text is None and utterances is None:
        raise ValueError(
            "Provide at least one of: transcript_path, text, or utterances."
        )

    # -----------------------------------------------------------------------
    # Tokenise (simple whitespace + punctuation split)
    # -----------------------------------------------------------------------
    token_pattern = re.compile(r"\b[a-zA-Z']+\b")
    all_tokens: List[str] = token_pattern.findall(text.lower())

    if not all_tokens:
        logger.warning("No tokens found in the provided text/utterances.")
        return {
            "type_token_ratio": 0.0,
            "mean_utterance_length": 0.0,
            "total_utterances": float(len(utterances)),
            "total_words": 0.0,
            "filler_count": 0.0,
            "filler_rate": 0.0,
            "lexical_density": 0.0,
            "unique_noun_count": 0.0,
            "unique_verb_count": 0.0,
            "brunet_w_index": 0.0,
            "honore_r_statistic": 0.0,
        }

    total_words = len(all_tokens)
    unique_words = set(all_tokens)
    vocab_size = len(unique_words)

    # -----------------------------------------------------------------------
    # TTR
    # -----------------------------------------------------------------------
    ttr = vocab_size / total_words if total_words > 0 else 0.0

    # -----------------------------------------------------------------------
    # Utterance-level statistics
    # -----------------------------------------------------------------------
    if utterances:
        utterance_lengths = []
        for utt in utterances:
            utt_tokens = token_pattern.findall(utt.lower())
            utterance_lengths.append(len(utt_tokens))
        mean_utt_length = float(np.mean(utterance_lengths)) if utterance_lengths else 0.0
        total_utterances = float(len(utterances))
    else:
        mean_utt_length = 0.0
        total_utterances = 0.0

    # -----------------------------------------------------------------------
    # Filler words
    # -----------------------------------------------------------------------
    filler_set = set(filler_words)
    filler_count = sum(1 for tok in all_tokens if tok in filler_set)
    filler_rate = filler_count / total_words if total_words > 0 else 0.0

    # -----------------------------------------------------------------------
    # Lexical density (content vs function words)
    # -----------------------------------------------------------------------
    function_word_count = sum(1 for tok in all_tokens if tok in _FUNCTION_WORDS)
    content_word_count = total_words - function_word_count
    lexical_density = content_word_count / total_words if total_words > 0 else 0.0

    # -----------------------------------------------------------------------
    # Unique noun / verb count (heuristic — lexicon lookup)
    # -----------------------------------------------------------------------
    unique_nouns = unique_words.intersection(_COMMON_NOUNS)
    unique_verbs = unique_words.intersection(_COMMON_VERBS)

    # -----------------------------------------------------------------------
    # Vocabulary richness indices
    # -----------------------------------------------------------------------
    # Brunét's W: N^(V^-0.165), where N = total_words, V = vocab_size
    brunet_w = (total_words ** (vocab_size ** -0.165)) if vocab_size > 0 else 0.0

    # Honoré's R = 100 * log(N) / (1 - V1/V)
    # V1 = number of words that appear exactly once (hapax legomena)
    from collections import Counter
    freq: Counter = Counter(all_tokens)
    hapax_count = sum(1 for cnt in freq.values() if cnt == 1)
    if vocab_size > 0 and hapax_count < vocab_size and total_words > 1:
        honore_r = 100.0 * np.log(total_words) / (1.0 - hapax_count / vocab_size)
    else:
        honore_r = 0.0

    features = {
        "type_token_ratio": float(ttr),
        "mean_utterance_length": float(mean_utt_length),
        "total_utterances": float(total_utterances),
        "total_words": float(total_words),
        "filler_count": float(filler_count),
        "filler_rate": float(filler_rate),
        "lexical_density": float(lexical_density),
        "unique_noun_count": float(len(unique_nouns)),
        "unique_verb_count": float(len(unique_verbs)),
        "brunet_w_index": float(brunet_w),
        "honore_r_statistic": float(honore_r),
    }

    logger.debug("Linguistic features: %s", features)
    return features
