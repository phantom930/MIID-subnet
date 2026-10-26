# MIID/miner/word_check.py
#
# Spoken-word check for voice clones (P5 C1).
#
# The subnet owner grades a clone in two steps: SpeechBrain ECAPA similarity
# above 0.6 to the reference (voice_generator gates on that), then a reviewer
# listens for every target word, and one missing, swapped or garbled word
# fails the whole clip. This module is the second step done by machine:
# Whisper transcribes each take, the transcript and the target words are
# brought to one canonical form, and the two sequences are aligned token by
# token. A take passes only with no missing, wrong or extra token.
#
# The targets are 3 digit words and 7 pool words (MIID/validator/
# voice_words.py) in English or Spanish, wrapped in the first and last four
# characters of the miner's hotkey as single tokens: "5 H P O ... R D 8 L".
# A digit and its word ("8", "eight", "ocho") and a letter and its spoken
# name ("H", "aitch", "hache") are the same token on both sides, so the
# check does not depend on how Whisper chooses to write them.
#
# spoken_text() is what the TTS model is given instead of the raw target
# text: the same tokens, with the single characters written as the
# language names them (a bare "H" is read as a letter only some of the
# time, and a Spanish voice reads "8" in English as often as not) and a
# comma between tokens, so each one is said on its own and none run together.
#
# Whisper (MIID_WORD_CHECK_MODEL, default openai/whisper-large-v3-turbo) is
# loaded lazily in this process and parked in CPU memory between rounds like
# the voice models. MIID_WORD_CHECK=0 switches the check off.

import os
import re
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bittensor as bt


ENABLED = os.environ.get("MIID_WORD_CHECK", "1").strip().lower() not in ("0", "false", "no")
MODEL_ID = os.environ.get("MIID_WORD_CHECK_MODEL", "openai/whisper-large-v3-turbo")
ASR_SR = 16000

DIGIT_WORDS: Dict[str, Tuple[str, ...]] = {
    "en": ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    "es": ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"),
}

# How each language names its letters: the first spelling is what
# spoken_text() hands the TTS model, every spelling is read back as the letter.
LETTER_NAMES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "en": {
        "A": ("ay",), "B": ("bee", "be"), "C": ("see", "sea", "cee"), "D": ("dee",),
        "E": ("ee",), "F": ("eff", "ef"), "G": ("gee", "jee"), "H": ("aitch", "haitch", "eych"),
        "I": ("eye",), "J": ("jay",), "K": ("kay",), "L": ("el", "ell", "elle"),
        "M": ("em",), "N": ("en",), "O": ("oh",), "P": ("pee", "pea"),
        "Q": ("cue", "queue"), "R": ("ar", "are", "arr"), "S": ("ess", "es"),
        "T": ("tee", "tea"), "U": ("you", "yu"), "V": ("vee",),
        "W": ("double-you", "double u"), "X": ("ex",), "Y": ("why", "wye"),
        "Z": ("zee", "zed"),
    },
    "es": {
        "A": ("a",), "B": ("be",), "C": ("ce",), "D": ("de",), "E": ("e",),
        "F": ("efe",), "G": ("ge",), "H": ("hache", "ache"), "I": ("i",),
        "J": ("jota",), "K": ("ka", "ca"), "L": ("ele",), "M": ("eme",),
        "N": ("ene",), "O": ("o",), "P": ("pe",), "Q": ("cu",),
        "R": ("erre", "ere"), "S": ("ese",), "T": ("te",), "U": ("u",),
        "V": ("uve", "ve"), "W": ("uve doble", "doble ve"), "X": ("equis",),
        "Y": ("ye", "i griega"), "Z": ("zeta", "ceta"),
    },
}


def _plain(word: str) -> str:
    """Lower case without accents: Whisper drops or adds them at will."""
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _language(language: str) -> str:
    lang = (language or "en").strip().lower()[:2]
    return lang if lang in DIGIT_WORDS else "en"


def _lookup(language: str) -> Dict[str, str]:
    """Spelling (plain) -> canonical token, for digit words and letter names."""
    table: Dict[str, str] = {}
    for digit, word in enumerate(DIGIT_WORDS[language]):
        table[_plain(word)] = f"#{digit}"
    for letter, names in LETTER_NAMES[language].items():
        for name in names:
            table.setdefault(_plain(name), f"@{letter}")
    return table


def canonical_targets(target_words: Sequence[str], language: str) -> List[str]:
    """The target tokens in canonical form: "#8" for a digit or digit word,
    "@H" for a single letter, the plain word otherwise."""
    lang = _language(language)
    digits = {_plain(word): f"#{digit}" for digit, word in enumerate(DIGIT_WORDS[lang])}
    out = []
    for token in target_words:
        token = token.strip()
        if not token:
            continue
        if len(token) == 1 and token.isdigit():
            out.append(f"#{token}")
        elif len(token) == 1 and token.isalpha():
            out.append(f"@{token.upper()}")
        else:
            plain = _plain(token)
            out.append(digits.get(plain, plain))
    return out


def canonical_heard(transcript: str, language: str, vocabulary: Sequence[str] = ()) -> List[str]:
    """Whisper's transcript in the same canonical form as canonical_targets.

    Single characters, digits and letter names become tokens of their own;
    a run of capitals and digits that is not a word ("HPO", "RD8L", "5-H")
    is split into characters, since Whisper often writes a spelled-out
    sequence as one block. Whisper's number formatting is undone ("768th",
    "473" are three digits) and a digit 0 is the letter O: the targets never
    hold a zero (no digit word is zero, and SS58 has no "0") while Whisper
    writes "oh" as 0. Two target words run together ("Cloudbread") are
    split. Two-word letter names ("i griega", "uve doble") are joined first.
    """
    lang = _language(language)
    table = _lookup(lang)
    words = {_plain(w) for w in vocabulary if len(w.strip()) > 1}
    words |= {_plain(w) for w in DIGIT_WORDS[lang][1:]}
    letters = {w.strip().lower() for w in vocabulary if len(w.strip()) == 1 and w.strip().isalpha()}
    text = transcript
    for letter, names in LETTER_NAMES[lang].items():
        for name in names:
            if " " in name or "-" in name:
                text = re.sub(rf"\b{re.escape(name)}\b", f" {letter} ", text, flags=re.IGNORECASE)

    def char(c: str) -> str:
        if c == "0":
            return "@O"
        return f"#{c}" if c.isdigit() else f"@{c.upper()}"

    def word(plain: str) -> str:
        return table[plain] if table.get(plain, "").startswith("#") else plain

    pieces: List[str] = []
    for raw in re.findall(r"[^\W_]+", text, flags=re.UNICODE):
        # Whisper writes ordinals in lower case ("768th"); "1RD" is a digit
        # and two letters.
        if re.fullmatch(r"\d+(?:st|nd|rd|th|o|a)?", raw):
            pieces.append(raw)
        else:
            # "5One", "Clock1RDAL", "HP086": Whisper glues digits to words.
            pieces.extend(re.findall(r"\d+|[^\W\d_]+", raw, flags=re.UNICODE))
    out: List[str] = []
    for raw in pieces:
        plain = _plain(raw)
        digits = re.fullmatch(r"(\d+)(?:st|nd|rd|th|o|a)?", plain)
        if digits:
            out.extend(char(c) for c in digits.group(1))
        elif len(raw) == 1 and raw.isalpha():
            out.append(char(raw))
        elif plain in words:
            # A target word stays itself even where it is also a letter name.
            out.append(word(plain))
        elif re.fullmatch(r"[A-Z0-9]{2,8}", raw):
            out.extend(char(c) for c in raw)
        elif (parts := _split_run(plain, words, letters)) is not None:
            out.extend(char(p) if len(p) == 1 else word(p) for p in parts)
        else:
            out.append(table.get(plain, plain))
    return out


def _split_run(plain: str, words: set, letters: set, most: int = 3) -> Optional[List[str]]:
    """``plain`` as two or three of ``words`` run together, or None. A
    spelled letter of the targets may lead ("ocloud": O, cloud)."""
    for cut in range(1, len(plain) - 1):
        head, tail = plain[:cut], plain[cut:]
        if head not in words and not (cut == 1 and head in letters):
            continue
        if tail in words:
            return [head, tail]
        if most > 2:
            rest = _split_run(tail, words, letters, most - 1)
            if rest is not None:
                return [head] + rest
    return None


def align(expected: Sequence[str], heard: Sequence[str]) -> Dict[str, Any]:
    """Token alignment (Levenshtein): what was missing, wrong or extra.

    Returns {"ok", "errors", "missing", "wrong", "extra"}; "wrong" lists
    (expected, heard) pairs.
    """
    n, m = len(expected), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = expected[i - 1] == heard[j - 1]
            cost[i][j] = min(
                cost[i - 1][j - 1] + (0 if same else 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    missing, wrong, extra = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != heard[j - 1]):
            if expected[i - 1] != heard[j - 1]:
                wrong.append((expected[i - 1], heard[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            missing.append(expected[i - 1])
            i -= 1
        else:
            extra.append(heard[j - 1])
            j -= 1
    errors = cost[n][m]
    return {
        "ok": errors == 0,
        "errors": errors,
        "missing": missing[::-1],
        "wrong": wrong[::-1],
        "extra": extra[::-1],
    }


def spoken_text(target_words: Sequence[str], language: str) -> str:
    """The target tokens as the TTS model should read them: digits as words,
    single letters by their names, one comma-separated token at a time."""
    lang = _language(language)
    parts = []
    for token in target_words:
        token = token.strip()
        if not token:
            continue
        if len(token) == 1 and token.isdigit():
            parts.append(DIGIT_WORDS[lang][int(token)])
        elif len(token) == 1 and token.isalpha():
            parts.append(LETTER_NAMES[lang][token.upper()][0])
        else:
            parts.append(token)
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

class _Whisper:
    """Whisper loaded once, moved between the CPU and the GPU per request."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"
        self.failed: Optional[str] = None

    def ensure(self, device: str, run: bool = True) -> bool:
        """Load if needed and move to ``device``; ``run``: about to run there."""
        if self.failed:
            return False
        import torch

        if self.model is None:
            try:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                started = time.time()
                self.processor = WhisperProcessor.from_pretrained(MODEL_ID)
                # Half precision wherever it will run on a GPU: it is parked
                # in CPU memory between rounds at half the size, too.
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    MODEL_ID, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                ).eval()
                self.device = "cpu"
                bt.logging.info(f"Word check: {MODEL_ID} loaded in {time.time() - started:.1f}s")
            except Exception as e:  # noqa: BLE001
                self.failed = f"{type(e).__name__}: {e}"
                bt.logging.warning(f"Word check: Whisper unavailable ({self.failed}); clips go unchecked")
                return False
        if self.device != device:
            self.model.to(device)
            self.device = device
        if run and device == "cpu" and self.model.dtype == torch.float16:
            # Inference on the CPU (the voice models fell back there): half
            # precision is slow or unsupported on it.
            self.model.to(dtype=torch.float32)
        return True

    def transcribe(self, wav16k: Any, language: str, device: str) -> Optional[str]:
        import torch

        with self.lock:
            if not self.ensure(device):
                return None
            audio = wav16k.detach().float().cpu().numpy()
            features = self.processor(audio, sampling_rate=ASR_SR, return_tensors="pt").input_features
            features = features.to(self.device, dtype=self.model.dtype)
            with torch.inference_mode():
                ids = self.model.generate(features, language=_language(language), task="transcribe")
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def park(self) -> None:
        with self.lock:
            if self.model is not None and self.device != "cpu":
                self.model.to("cpu")
                self.device = "cpu"


_WHISPER = _Whisper()


def check(
    wav16k: Any, target_words: Sequence[str], language: str, device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Transcribe a 16 kHz mono clip and align it with the target words.

    Returns align()'s result plus "heard" (the transcript), or None when the
    check is off or Whisper could not run — an unchecked take is neither a
    pass nor a fail.
    """
    if not ENABLED:
        return None
    try:
        transcript = _WHISPER.transcribe(wav16k, language, device)
    except Exception as e:  # noqa: BLE001
        bt.logging.warning(f"Word check: transcription failed ({type(e).__name__}: {e})")
        return None
    if transcript is None:
        return None
    expected = canonical_targets(target_words, language)
    result = align(expected, canonical_heard(transcript, language, target_words))
    result["heard"] = transcript
    return result


def describe(result: Optional[Dict[str, Any]]) -> str:
    """One line for the log: "all 18 words heard" or what went wrong."""
    if result is None:
        return "words unchecked"
    if result["ok"]:
        return "every word heard"
    parts = []
    if result["missing"]:
        parts.append("missing " + " ".join(result["missing"]))
    if result["wrong"]:
        parts.append("wrong " + ", ".join(f"{a}→{b}" for a, b in result["wrong"]))
    if result["extra"]:
        parts.append("extra " + " ".join(result["extra"]))
    return f"{result['errors']} word error(s): " + "; ".join(parts) + f" (heard {result['heard']!r})"


def preload() -> None:
    """Load Whisper into CPU memory ahead of the first voice request."""
    if ENABLED:
        with _WHISPER.lock:
            _WHISPER.ensure("cpu", run=False)


def release() -> None:
    """Free Whisper's VRAM between rounds (it stays in CPU memory)."""
    _WHISPER.park()
