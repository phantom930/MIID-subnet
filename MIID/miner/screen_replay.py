# MIID/miner/screen_replay.py
#
# The synthetic screen replay (6th slot, P5 C1), composed instead of drawn
# whole by FLUX.
#
# The validator asks for a photo of the seed face shown on a named device
# (phone, tablet, laptop, monitor, tv) with two named visual cues (moiré /
# pixel grid, screen glare, keystone perspective, gamma / contrast shift,
# edge / bezel cues). The slot is reviewed by hand, and the subnet owner's
# bar is that it must break Passive Liveness v3 (/is_live answers "not a
# screen replay") while the device type and both cues stay visible and the
# face stays the seed's identity.
#
# A FLUX-drawn device scene does not break it. Probed on 2026-10-26 through
# /is_live: every archived FLUX screen replay read as a replay at 0.999+,
# and so did the same scene with its screen content swapped for a clean
# take (0.99), cropped to the screen (0.997), blurred or denoised (0.9999).
# A device drawn here around a generated photo of the person does pass, and
# what the screen shows decides it more than the device or the cues: over
# 45 composites with random settings, 14 of 15 passed showing the person
# outdoors, 7 of 13 indoors and 3 of 17 on a plain studio backdrop. The
# verdict reads a region around the face (three composites with the same
# face crop scored an identical 0.9397), and the cut sits near 0.98.
#
# So the slot is built in two halves: FLUX generates what the screen shows
# — the reference person outdoors, at the screen's shape (content_size) —
# and compose() draws the named device around it with both requested cues
# plainly visible, as a camera would see it. The LADDER of compose()
# settings is tried in order and liveness_check verifies each composite
# against /is_live before it ships (image_generator).

import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEVICES = ("phone", "tablet", "laptop", "monitor", "tv")
CUES = ("moire", "glare", "keystone", "gamma", "edge")

# How each validator cue description is recognised (SCREEN_REPLAY_VISUAL_CUES
# in MIID/validator/image_variations.py).
_CUE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("moire", r"moir|pixel grid|subpixel"),
    ("glare", r"glare|specular|hotspot"),
    ("keystone", r"keystone|perspective|off-angle"),
    ("gamma", r"gamma|contrast shift|colour/brightness|color/brightness"),
    ("edge", r"edge / crop|bezel|screen borders|crop cues"),
)
_DEVICE_PATTERN = re.compile(r"\bon an? (phone|tablet|laptop|monitor|tv)\b", re.IGNORECASE)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# The size FLUX draws each device's screen content at: the screen's own
# shape, at about the pixel count of a 1008x1344 portrait (multiples of 16).
CONTENT_SIZES: Dict[str, Tuple[int, int]] = {
    "phone": (784, 1632),
    "tablet": (1008, 1392),
    "laptop": (1440, 896),
    "monitor": (1536, 864),
    "tv": (1536, 864),
}


def content_size(device: str) -> Tuple[int, int]:
    return CONTENT_SIZES.get(device, CONTENT_SIZES["phone"])


def parse_request(description: str, detail: str) -> Tuple[str, List[str]]:
    """(device, cue keys) named in a screen_replay request.

    The cue list is the one the validator wrote, in its order; a request
    whose cues cannot be read gets the two that every composite shows anyway
    (the bezel and the display's gamma).
    """
    text = f"{description} {detail}"
    match = _DEVICE_PATTERN.search(text)
    device = match.group(1).lower() if match else "phone"
    blob = detail.lower()
    found = []
    for key, pattern in _CUE_PATTERNS:
        position = re.search(pattern, blob)
        if position:
            found.append((position.start(), key))
    cues = [key for _, key in sorted(found)]
    return device, cues or ["edge", "gamma"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = False) -> Any:
    try:
        return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _rounded(size: Tuple[int, int], radius: int) -> np.ndarray:
    """Float mask (h, w) of a rounded rectangle, antialiased."""
    w, h = size
    scale = 4
    big = Image.new("L", (w * scale, h * scale), 0)
    ImageDraw.Draw(big).rounded_rectangle(
        (0, 0, w * scale - 1, h * scale - 1), radius=radius * scale, fill=255,
    )
    return np.asarray(big.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0


def _screen_content(
    content: Image.Image, size: Tuple[int, int], face_box: Tuple[float, float, float, float],
    clearance: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """What the screen shows (float RGB, ``size``) and where the face lands.

    The portrait fills the screen, the face centred across and a little
    above the middle, but never so large that the screen edge comes within
    ``clearance`` face boxes of the face (a face touching the bezel looks
    cut off, and its identity reading suffers). A portrait shrunk for that
    is extended at its edges, and the shoulders always run to the bottom.
    A small face does not help with /is_live — composites with more than
    0.3 face boxes of clearance passed less often (6 of 17) than closer
    ones (13 of 20).
    """
    import cv2

    sw, sh = size
    cw, ch = content.size
    fx0, fy0, fx1, fy1 = face_box
    room = 1 + 2 * clearance
    k = min(
        max(sw / cw, sh / ch),
        sw / max(1.0, (fx1 - fx0) * room), sh / max(1.0, (fy1 - fy0) * room),
    )
    pw, ph = max(1, int(round(cw * k))), max(1, int(round(ch * k)))
    picture = np.asarray(content.convert("RGB").resize((pw, ph), Image.LANCZOS), dtype=np.float32)
    ox = min(max(sw / 2 - (fx0 + fx1) / 2 * k, sw - pw), 0.0) if pw >= sw else sw / 2 - (fx0 + fx1) / 2 * k
    oy = sh * 0.45 - (fy0 + fy1) / 2 * k
    oy = max(min(oy, 0.0) if ph >= sh else oy, sh - ph)
    ix0, iy0 = int(round(ox)), int(round(oy))

    # Pad the picture out to the screen by repeating its edge pixels, then
    # soften what was repeated so it reads as out-of-focus backdrop.
    top, left = max(0, iy0), max(0, ix0)
    bottom, right = max(0, sh - (iy0 + ph)), max(0, sw - (ix0 + pw))
    padded = cv2.copyMakeBorder(picture, top, bottom, left, right, cv2.BORDER_REPLICATE)
    inside = cv2.copyMakeBorder(np.ones((ph, pw), np.float32), top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0.0)
    y0, x0 = max(0, -iy0), max(0, -ix0)
    padded, inside = padded[y0:y0 + sh, x0:x0 + sw], inside[y0:y0 + sh, x0:x0 + sw]
    if inside.min() < 1:
        soft = cv2.GaussianBlur(padded, (0, 0), max(sw, sh) * 0.02)
        weight = cv2.GaussianBlur(inside, (0, 0), max(sw, sh) * 0.01)[..., None]
        padded = padded * weight + soft * (1 - weight)
    face = (ix0 + fx0 * k, iy0 + fy0 * k, ix0 + fx1 * k, iy0 + fy1 * k)
    return padded, face


# ---------------------------------------------------------------------------
# Cues, rendered on the screen content (screen pixels, float RGB 0..255)
# ---------------------------------------------------------------------------

def _display_gamma(rgb: np.ndarray, amount: float) -> np.ndarray:
    """The look of a backlit panel photographed: lifted blacks, a bluish
    white point, harder mid-tone contrast and clipped highlights."""
    x = rgb / 255.0
    x = 0.07 * amount + (1 - 0.07 * amount) * np.power(np.clip(x, 0, 1), 1 - 0.18 * amount)
    x = 0.5 + (x - 0.5) * (1 + 0.22 * amount)
    tint = np.array([1 - 0.05 * amount, 1.0, 1 + 0.07 * amount], dtype=np.float32)
    x = x * tint + 0.04 * amount
    return np.clip(x * 255.0, 0, 255)


def _moire(shape: Tuple[int, int], amount: float, rng: random.Random) -> np.ndarray:
    """Coloured interference bands: two near-parallel gratings beating
    against each other, curved the way a camera's sampling of a pixel grid
    bends them, with the RGB phases offset so the bands fringe in colour."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    angle = math.radians(rng.uniform(15, 75))
    period = rng.uniform(16, 30) * max(w, h) / 1000.0
    u = xx * math.cos(angle) + yy * math.sin(angle)
    bend = (((xx - w * rng.uniform(0.3, 0.7)) / w) ** 2 + ((yy - h * rng.uniform(0.3, 0.7)) / h) ** 2)
    phase = 2 * math.pi * (u / period + 6.0 * bend)
    out = np.empty((h, w, 3), dtype=np.float32)
    for c, offset in enumerate((0.0, 2.1, 4.2)):
        out[..., c] = np.sin(phase + offset)
    envelope = 0.55 + 0.45 * np.sin(2 * math.pi * (yy / h * rng.uniform(1.0, 2.0) + rng.random()))
    return out * envelope[..., None] * 13.0 * amount


def _pixel_grid(shape: Tuple[int, int], pitch: int, amount: float) -> np.ndarray:
    """The panel's subpixel stripes and row gaps, at camera resolution."""
    h, w = shape
    grid = np.zeros((h, w, 3), dtype=np.float32)
    columns = np.arange(w) % (3 * pitch) // pitch
    for c in range(3):
        grid[:, columns == c, c] += 7.0 * amount
        grid[:, columns != c, c] -= 3.5 * amount
    grid[np.arange(h) % (3 * pitch) == 0] -= 9.0 * amount
    return grid


def _glare(shape: Tuple[int, int], amount: float, rng: random.Random,
           avoid: Tuple[float, float, float, float]) -> np.ndarray:
    """Specular reflections on the glass, as a 0..1 map: one or two bright
    hotspots of a window or lamp and a faint diagonal sheen, kept off the
    face box ``avoid`` (x0, y0, x1, y1 in screen pixels)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    g = np.zeros((h, w), dtype=np.float32)
    ax0, ay0, ax1, ay1 = avoid
    spots = [(0.82, 0.12), (0.15, 0.85), (0.86, 0.80), (0.14, 0.14)]
    rng.shuffle(spots)
    placed = 0
    unit = min(w, h)
    for fx, fy in spots:
        cx, cy = fx * w, fy * h
        if ax0 - 0.05 * w < cx < ax1 + 0.05 * w and ay0 - 0.05 * h < cy < ay1 + 0.05 * h:
            continue
        # A lamp's hotspot, smeared along the glass, inside the soft
        # reflection of the window it comes from.
        angle = rng.uniform(-0.6, 0.6)
        u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
        v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        rx, ry = rng.uniform(0.10, 0.16) * unit, rng.uniform(0.035, 0.06) * unit
        core = np.exp(-((u / rx) ** 2 + (v / ry) ** 2) * 2.0)
        window = 0.5 * (1 - np.tanh((np.abs(u) - rx * 2.2) / (0.04 * unit)))
        window *= 0.5 * (1 - np.tanh((np.abs(v) - ry * 3.0) / (0.04 * unit)))
        g += core + 0.22 * window
        placed += 1
        if placed == 2:
            break
    sheen_angle = rng.uniform(0.6, 1.0)
    band = np.abs((xx / w) * sheen_angle + (yy / h) - rng.uniform(0.2, 0.5))
    g += 0.18 * np.exp(-((band / 0.06) ** 2))
    # Nothing bright over the face itself: a reflection across the eyes
    # costs the identity match more than it adds to the cue.
    fx, fy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    face = np.exp(-(((xx - fx) / max(1.0, (ax1 - ax0) / 2)) ** 4 + ((yy - fy) / max(1.0, (ay1 - ay0) / 2)) ** 4))
    g *= 1 - 0.9 * face
    return np.clip(g * amount, 0, 1)


# ---------------------------------------------------------------------------
# Devices: body (RGB + mask) with the screen's place in it
# ---------------------------------------------------------------------------

class _Device:
    """A device drawn flat, front-on: ``body`` RGB, ``mask`` coverage, and
    ``screen`` = (x0, y0, x1, y1) of the display inside it, plus overlays
    drawn over the content (status bar, OSD) by ``decorate``."""

    def __init__(self, body: np.ndarray, mask: np.ndarray, screen: Tuple[int, int, int, int],
                 screen_radius: int = 0):
        self.body, self.mask, self.screen, self.screen_radius = body, mask, screen, screen_radius


def _status_bar(img: Image.Image, box: Tuple[int, int, int, int], scale: float, time_text: str) -> None:
    """A phone/tablet status bar: the time on the left, signal and battery right."""
    x0, y0, x1, _ = box
    d = ImageDraw.Draw(img)
    size = max(12, int(30 * scale))
    d.text((x0 + int(40 * scale), y0 + int(14 * scale)), time_text, fill=(255, 255, 255), font=_font(size, bold=True))
    bx = x1 - int(40 * scale) - int(52 * scale)
    by = y0 + int(20 * scale)
    bw, bh = int(46 * scale), int(22 * scale)
    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=int(6 * scale), outline=(255, 255, 255), width=max(1, int(2 * scale)))
    d.rounded_rectangle((bx + 3, by + 3, bx + int(bw * 0.75), by + bh - 3), radius=int(3 * scale), fill=(255, 255, 255))
    d.rectangle((bx + bw + 1, by + bh // 3, bx + bw + int(4 * scale), by + 2 * bh // 3), fill=(255, 255, 255))
    sx = bx - int(70 * scale)
    for i in range(4):
        hh = int((8 + 5 * i) * scale)
        d.rectangle((sx + i * int(10 * scale), by + bh - hh, sx + i * int(10 * scale) + int(6 * scale), by + bh), fill=(255, 255, 255))


def _phone(width: int, rng: random.Random) -> _Device:
    height = int(width / 0.49)
    radius = int(width * 0.13)
    bezel = max(8, int(width * 0.025))
    body = np.zeros((height, width, 3), dtype=np.float32)
    frame = np.array(rng.choice([(28, 28, 32), (60, 62, 66), (205, 206, 210), (40, 44, 60)]), dtype=np.float32)
    body[:] = (14, 14, 16)
    mask = _rounded((width, height), radius)
    rim = mask - np.pad(_rounded((width - 6, height - 6), radius - 3), 3)
    body += np.clip(rim, 0, 1)[..., None] * (frame - 14)
    screen = (bezel, bezel, width - bezel, height - bezel)
    return _Device(body, mask, screen, screen_radius=radius - bezel)


def _decorate_phone(img: Image.Image, device: _Device, offset: Tuple[int, int], rng: random.Random) -> None:
    x0, y0, x1, y1 = device.screen
    ox, oy = offset
    w = x1 - x0
    scale = w / 600.0
    d = ImageDraw.Draw(img)
    pill_w, pill_h = int(w * 0.21), int(w * 0.06)
    cx = ox + (x0 + x1) // 2
    top = oy + y0 + int(w * 0.03)
    d.rounded_rectangle((cx - pill_w // 2, top, cx + pill_w // 2, top + pill_h), radius=pill_h // 2, fill=(5, 5, 6))
    _status_bar(img, (ox + x0, oy + y0, ox + x1, oy + y1), scale, f"{rng.randint(7, 11)}:{rng.randint(0, 59):02d}")
    bar_w = int(w * 0.34)
    d.rounded_rectangle((cx - bar_w // 2, oy + y1 - int(w * 0.035), cx + bar_w // 2, oy + y1 - int(w * 0.022)),
                        radius=4, fill=(240, 240, 240))


def _tablet(width: int, rng: random.Random) -> _Device:
    height = int(width / 0.74)
    radius = int(width * 0.05)
    bezel = int(width * 0.04)
    body = np.zeros((height, width, 3), dtype=np.float32)
    body[:] = rng.choice([(18, 18, 20), (226, 226, 228)])
    mask = _rounded((width, height), radius)
    screen = (bezel, bezel, width - bezel, height - bezel)
    return _Device(body, mask, screen, screen_radius=int(radius * 0.4))


def _decorate_tablet(img: Image.Image, device: _Device, offset: Tuple[int, int], rng: random.Random) -> None:
    x0, y0, x1, y1 = device.screen
    ox, oy = offset
    d = ImageDraw.Draw(img)
    cx = ox + (x0 + x1) // 2
    r = max(3, int((x1 - x0) * 0.007))
    cy = oy + y0 // 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(40, 46, 60))
    _status_bar(img, (ox + x0, oy + y0, ox + x1, oy + y1), (x1 - x0) / 1100.0, f"{rng.randint(7, 11)}:{rng.randint(0, 59):02d}")


def _landscape_panel(width: int, aspect: float, side: int, top: int, chin: int,
                     colour: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int], int, int]:
    screen_w = width - 2 * side
    screen_h = int(screen_w / aspect)
    height = top + screen_h + chin
    body = np.zeros((height, width, 3), dtype=np.float32)
    body[:] = colour
    return body, (side, top, side + screen_w, top + screen_h), width, height


def _laptop(width: int, rng: random.Random) -> _Device:
    lid, screen, w, h = _landscape_panel(width, 16 / 10, int(width * 0.018), int(width * 0.03), int(width * 0.045), (16, 16, 18))
    deck_h = int(width * 0.36)
    total = h + deck_h
    body = np.zeros((total, w, 3), dtype=np.float32)
    mask = np.zeros((total, w), dtype=np.float32)
    body[:h] = lid
    mask[:h] = _rounded((w, h), int(width * 0.015))
    # Keyboard deck, catching more light toward the camera.
    metal = np.array(rng.choice([(176, 178, 182), (120, 122, 128), (60, 62, 66)]), dtype=np.float32)
    shade = np.linspace(0.92, 1.04, deck_h, dtype=np.float32)[:, None, None]
    body[h:] = metal * shade
    mask[h:] = 1.0
    img = Image.fromarray(np.clip(body, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle((0, h, w, h + int(width * 0.012)), fill=(30, 30, 32))  # hinge
    kx0, kx1 = int(w * 0.08), int(w * 0.92)
    ky0, ky1 = h + int(deck_h * 0.12), h + int(deck_h * 0.62)
    rows, cols = 5, 14
    gap = max(2, int(w * 0.004))
    kw = (kx1 - kx0 - gap * (cols - 1)) / cols
    kh = (ky1 - ky0 - gap * (rows - 1)) / rows
    for r in range(rows):
        for c in range(cols):
            x = kx0 + c * (kw + gap)
            y = ky0 + r * (kh + gap)
            d.rounded_rectangle((x, y, x + kw, y + kh), radius=max(1, int(kh * 0.15)), fill=(24, 24, 26))
    tw = int(w * 0.34)
    d.rounded_rectangle((w // 2 - tw // 2, ky1 + int(deck_h * 0.08), w // 2 + tw // 2, total - int(deck_h * 0.04)),
                        radius=int(w * 0.01), outline=tuple(int(v * 0.8) for v in metal), width=2)
    body = np.asarray(img, dtype=np.float32)
    return _Device(body, mask, screen)


def _decorate_webcam(img: Image.Image, device: _Device, offset: Tuple[int, int], rng: random.Random) -> None:
    x0, y0, x1, _ = device.screen
    ox, oy = offset
    d = ImageDraw.Draw(img)
    cx, cy = ox + (x0 + x1) // 2, oy + y0 // 2
    r = max(2, int((x1 - x0) * 0.005))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(50, 56, 70))


def _monitor(width: int, rng: random.Random) -> _Device:
    panel, screen, w, h = _landscape_panel(width, 16 / 9, int(width * 0.012), int(width * 0.012), int(width * 0.035), (20, 20, 22))
    neck_h = int(width * 0.2)
    base_h = int(width * 0.04)
    total = h + neck_h + base_h
    body = np.zeros((total, w, 3), dtype=np.float32)
    mask = np.zeros((total, w), dtype=np.float32)
    body[:h] = panel
    mask[:h] = _rounded((w, h), int(width * 0.006))
    img = Image.fromarray(np.clip(body, 0, 255).astype(np.uint8))
    m = Image.fromarray((mask * 255).astype(np.uint8))
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(m)
    stand = rng.choice([(48, 48, 52), (150, 152, 156)])
    nw = int(w * 0.07)
    neck = [(w // 2 - nw // 2, h), (w // 2 + nw // 2, h), (w // 2 + nw // 2 + 6, h + neck_h), (w // 2 - nw // 2 - 6, h + neck_h)]
    d.polygon(neck, fill=stand)
    dm.polygon(neck, fill=255)
    bw = int(w * 0.34)
    base = (w // 2 - bw // 2, h + neck_h - 4, w // 2 + bw // 2, total - 1)
    d.ellipse(base, fill=tuple(int(v * 0.9) for v in stand))
    dm.ellipse(base, fill=255)
    # Power LED on the chin.
    lx, ly = int(w * 0.93), h - int(width * 0.017)
    d.ellipse((lx - 3, ly - 3, lx + 3, ly + 3), fill=(90, 200, 255))
    return _Device(np.asarray(img, dtype=np.float32), np.asarray(m, dtype=np.float32) / 255, screen)


def _tv(width: int, rng: random.Random) -> _Device:
    panel, screen, w, h = _landscape_panel(width, 16 / 9, int(width * 0.008), int(width * 0.008), int(width * 0.02), (12, 12, 13))
    leg_h = int(width * 0.06)
    total = h + leg_h
    body = np.zeros((total, w, 3), dtype=np.float32)
    mask = np.zeros((total, w), dtype=np.float32)
    body[:h] = panel
    mask[:h] = 1.0
    img = Image.fromarray(np.clip(body, 0, 255).astype(np.uint8))
    m = Image.fromarray((mask * 255).astype(np.uint8))
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(m)
    for fx in (0.12, 0.88):
        x = int(w * fx)
        leg = [(x - 4, h), (x + 4, h), (x + int(w * 0.035), total - 1), (x - int(w * 0.035), total - 1)]
        d.polygon(leg, fill=(30, 30, 32))
        dm.polygon(leg, fill=255)
    return _Device(np.asarray(img, dtype=np.float32), np.asarray(m, dtype=np.float32) / 255, screen)


def _decorate_tv(img: Image.Image, device: _Device, offset: Tuple[int, int], rng: random.Random) -> None:
    x0, y0, x1, _ = device.screen
    ox, oy = offset
    w = x1 - x0
    d = ImageDraw.Draw(img, "RGBA")
    size = max(12, int(w * 0.03))
    label = rng.choice(["HDMI 1", "HDMI 2", "Photos"])
    tx, ty = ox + x0 + int(w * 0.03), oy + y0 + int(w * 0.025)
    d.rounded_rectangle((tx - 8, ty - 6, tx + int(size * 0.62 * len(label)) + 10, ty + size + 8), radius=6, fill=(0, 0, 0, 110))
    d.text((tx, ty), label, fill=(255, 255, 255, 230), font=_font(size, bold=True))


_BUILDERS = {
    # device: (builder, decorator, width as a fraction of the frame)
    "phone": (_phone, _decorate_phone, 0.60),
    "tablet": (_tablet, _decorate_tablet, 0.88),
    "laptop": (_laptop, _decorate_webcam, 0.96),
    "monitor": (_monitor, _decorate_webcam, 0.96),
    "tv": (_tv, _decorate_tv, 0.97),
}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

WALLS = ((216, 210, 200), (204, 207, 210), (226, 219, 206), (196, 202, 192), (212, 204, 214))
DESKS = ((150, 112, 80), (196, 182, 160), (232, 230, 226), (92, 80, 72))


def _backdrop(size: Tuple[int, int], horizon: int, brightness: float, rng: random.Random) -> np.ndarray:
    """An out-of-focus room behind the device: a wall lit from above with a
    few soft shapes in it, and the desk it stands on from ``horizon`` down.
    (Its brightness made no difference to /is_live: 10 of 20 bright and 9
    of 17 dim composites passed.)"""
    import cv2

    w, h = size
    wall = np.array(rng.choice(WALLS), dtype=np.float32)
    desk = np.array(rng.choice(DESKS), dtype=np.float32)
    ramp = np.linspace(1.06, 0.88, h, dtype=np.float32)[:, None, None]
    room = np.broadcast_to(wall, (h, w, 3)) * ramp
    blobs = np.zeros((h // 16, w // 16, 3), dtype=np.float32)
    for _ in range(rng.randint(2, 4)):
        x, y = rng.randrange(w // 16), rng.randrange(max(1, horizon // 16))
        tone = rng.uniform(-40, 30)
        cv2.circle(blobs, (x, y), rng.randint(3, 9), (tone, tone * 0.9, tone * 0.8), -1)
    room = room + cv2.resize(cv2.GaussianBlur(blobs, (0, 0), 3), (w, h), interpolation=cv2.INTER_CUBIC)
    surface = np.linspace(0.9, 1.08, max(1, h - horizon), dtype=np.float32)[:, None, None]
    room[horizon:] = np.broadcast_to(desk, (h - horizon, w, 3)) * surface
    room = cv2.GaussianBlur(room, (0, 0), 3)
    return np.clip(room * brightness, 0, 255)


def compose(
    content: Image.Image,
    device: str,
    cues: Sequence[str],
    strength: float = 1.0,
    brightness: float = 0.85,
    face_box: Optional[Tuple[float, float, float, float]] = None,
    clearance: float = 0.15,
    device_scale: float = 1.1,
    camera: bool = True,
    decorations: bool = True,
    seed: Optional[int] = None,
    size: Tuple[int, int] = (1015, 1350),
) -> Image.Image:
    """``content`` (the portrait the screen shows) on the named ``device``,
    photographed, with each of ``cues`` visibly rendered at ``strength``
    (0..1). ``face_box`` is the face in ``content`` (x0, y0, x1, y1, as
    ada_face_compare.face_geometry finds it; default: where a passport
    portrait has it); ``brightness`` scales the room behind it."""
    import cv2

    rng = random.Random(seed)
    device = device if device in _BUILDERS else "phone"
    cues = set(cues)
    W, H = size
    builder, decorate, width_frac = _BUILDERS[device]
    dev = builder(int(W * width_frac * device_scale), rng)
    sx0, sy0, sx1, sy1 = dev.screen
    sw, sh = sx1 - sx0, sy1 - sy0

    cw, ch = content.size
    face_box = face_box or (cw * 0.3, ch * 0.22, cw * 0.7, ch * 0.62)
    # The face in screen pixels comes back with it: the glare stays off it
    # and the moiré is fainter over it (the flat backdrop around it shows
    # the bands best).
    screen, face = _screen_content(content, (sw, sh), face_box, clearance)

    # The panel always reads as a display a little; the gamma cue strongly.
    screen = _display_gamma(screen, strength * (1.0 if "gamma" in cues else 0.25))
    if "moire" in cues:
        yy, xx = np.mgrid[:sh, :sw].astype(np.float32)
        fcx, fcy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
        frx, fry = max(1.0, (face[2] - face[0]) * 0.6), max(1.0, (face[3] - face[1]) * 0.6)
        over_face = np.exp(-(((xx - fcx) / frx) ** 2 + ((yy - fcy) / fry) ** 2) ** 2)
        screen = screen + _moire((sh, sw), strength, rng) * (1 - 0.6 * over_face)[..., None]
    if "glare" in cues:
        g = _glare((sh, sw), strength, rng, face)[..., None]
        screen = screen + g * (255.0 * 1.05 - screen)
    screen = np.clip(screen, 0, 255)

    body = dev.body.copy()
    screen_mask = _rounded((sw, sh), dev.screen_radius) if dev.screen_radius else np.ones((sh, sw), np.float32)
    region = body[sy0:sy1, sx0:sx1]
    body[sy0:sy1, sx0:sx1] = region * (1 - screen_mask[..., None]) + screen * screen_mask[..., None]
    dh, dw = body.shape[:2]

    # Edge cue: a lit rim along the bezel, as a lamp catches the device's edge.
    if "edge" in cues:
        rim = np.zeros((dh, dw), dtype=np.float32)
        rim[: max(3, dh // 200)] = 1.0
        rim[:, : max(3, dw // 300)] = 0.6
        rim = cv2.GaussianBlur(rim, (0, 0), 2.0) * dev.mask
        body = body + rim[..., None] * 90.0 * strength

    # The body catches the room light from above: a soft top-to-bottom
    # falloff outside the screen.
    outside = 1 - np.pad(screen_mask, ((sy0, dh - sy1), (sx0, dw - sx1)))
    falloff = np.linspace(1.25, 0.9, dh, dtype=np.float32)[:, None, None]
    body = body * (1 - outside[..., None]) + (body * falloff + 6) * outside[..., None]

    device_img = Image.fromarray(np.clip(body, 0, 255).astype(np.uint8))
    if decorations:
        decorate(device_img, dev, (0, 0), rng)
    body = np.asarray(device_img, dtype=np.float32)

    # Placement and camera angle. The keystone cue tilts the device back
    # until its top edge visibly narrows; otherwise a slight, natural angle.
    scale = min(1.0, (H * 0.95) / dh)
    tw, th = int(dw * scale), int(dh * scale)
    x0 = (W - tw) / 2 + rng.uniform(-0.02, 0.02) * W
    y0 = max(H * 0.03, (H - th) / 2 - (H * 0.08 if device in ("laptop", "monitor", "tv") else 0))
    # (0.09 of the width cost the face 0.05–0.15 identity on a phone.)
    k = (0.06 if "keystone" in cues else 0.015) * strength * tw
    lean = (0.02 if "keystone" in cues else 0.008) * strength * th * rng.choice((-1, 1))
    src = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])
    dst = np.float32([
        [x0 + k, y0 + lean], [x0 + tw - k, y0 - lean],
        [x0 + tw, y0 + th], [x0, y0 + th],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(body, matrix, (W, H), flags=cv2.INTER_AREA)
    coverage = cv2.warpPerspective(dev.mask, matrix, (W, H), flags=cv2.INTER_LINEAR)[..., None]

    # The desk meets the wall at the device's foot (a phone or tablet is
    # propped up on it, the others stand on it).
    horizon = int(min(H - 2, y0 + th - (0.04 * th if device in ("phone", "tablet") else 0)))
    room = _backdrop((W, H), horizon, brightness, rng)
    shadow = cv2.GaussianBlur(coverage[..., 0], (0, 0), W * 0.02)[..., None]
    room = room * (1 - 0.35 * shadow)
    out = room * (1 - coverage) + warped * coverage

    if "moire" in cues:
        # The subpixel grid itself, at the camera's resolution, on the screen only.
        screen_cov = np.zeros((dh, dw), dtype=np.float32)
        screen_cov[sy0:sy1, sx0:sx1] = screen_mask
        screen_cov = cv2.warpPerspective(screen_cov, matrix, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        out = out + _pixel_grid((H, W), 1, strength) * screen_cov

    if not camera:
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    # The camera: a touch of lens softness, vignetting and sensor noise.
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    vignette = 1 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    out = out * vignette[..., None]
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 2.2, out.shape).astype(np.float32)
    out = out + noise
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


# The rungs compose() is tried with, in order: the composite that ships is
# the first /is_live does not flag. Over 45 random composites probed on
# 2026-10-26 the cue strength made no difference worth giving the cues up
# for (strong 12 of 19 passed, faint 7 of 18), while a larger device and a
# larger face did (device scale over 1.02: 11 of 16; clearance under 0.3:
# 13 of 20, over: 6 of 17), and the one outdoor composite that failed had
# the smallest face of them. So the cues stay strong and later rungs bring
# the device and the face closer, then drop the camera simulation and the
# status bar / OSD.
LADDER: Tuple[Dict[str, Any], ...] = (
    {"strength": 1.0, "clearance": 0.15, "device_scale": 1.1},
    {"strength": 0.9, "clearance": 0.05, "device_scale": 1.15, "camera": False},
    {"strength": 0.8, "clearance": 0.0, "device_scale": 1.2, "camera": False, "decorations": False},
)


# Rung order by device. With outdoor content the first rung passed on all
# six phone, tablet and tv probes, but on 0 of 2 laptops and 1 of 2
# monitors, which then passed on the later rungs (laptop 2 of 3, monitor 4
# of 4): a wide screen starts closer in.
LADDER_ORDER: Dict[str, Tuple[int, ...]] = {
    "laptop": (2, 1, 0),
    "monitor": (1, 2, 0),
}


def variants(device: str, cues: Sequence[str]) -> List[Dict[str, Any]]:
    """compose() settings to try, in order (see LADDER, LADDER_ORDER)."""
    order = LADDER_ORDER.get(device, tuple(range(len(LADDER))))
    return [dict(LADDER[i]) for i in order]
