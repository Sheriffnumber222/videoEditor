"""
generateCaptions.py
===================
Add AI-generated captions to a vertical video.

Pipeline:
  1. Locate the input video in the sibling "videoEditorAssets" folder.
  2. Extract the audio.
  3. Transcribe it locally with faster-whisper (phrase-level segments).
  4. Detect the black bars (letterbox) so captions can be centered on the
     bottom bar, then burn styled captions into the video with ffmpeg/libass.
  5. Write "<name>_captioned.mp4" back into videoEditorAssets.

The video is expected to be 1080 x 2350 with a 16:9 clip fit inside it and
black bars on the top and bottom. Captions are drawn centered on the bottom
black bar in #E7B95F (configurable below).

Requirements (already set up on this machine):
  - ffmpeg / ffprobe (installed via winget: Gyan.FFmpeg)
  - pip install faster-whisper
  - No API key and no network needed; transcription runs locally on the CPU
    (or GPU). The model downloads itself once on first run, then is cached.
"""

import os
import re
import sys
import json
import shutil
import tempfile
import textwrap
import subprocess
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG  --  tweak anything in this block
# ---------------------------------------------------------------------------

# --- Files -----------------------------------------------------------------
# Folder is the sibling "videoEditorAssets" next to this script's parent.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "videoEditorAssets"

# Name of the raw video inside ASSETS_DIR. Leave as None to auto-pick the most
# recently modified video file in that folder.
INPUT_VIDEO_NAME = None

# Output filename (written into ASSETS_DIR). None => "<input stem>_captioned.mp4".
OUTPUT_VIDEO_NAME = None

# --- Transcription (local faster-whisper) ---------------------------------
# Model size: "tiny", "base", "small", "medium", "large-v3".
#   small  -> fast, decent accuracy (good default for social clips)
#   medium -> slower, more accurate
# The model downloads once on first use and is cached under ~/.cache.
WHISPER_MODEL = "small"
# "cpu" works everywhere. Use "cuda" if you have an NVIDIA GPU + CUDA set up.
WHISPER_DEVICE = "cpu"
# "int8" is fast/low-memory on CPU. On GPU try "float16".
WHISPER_COMPUTE_TYPE = "int8"
TRANSCRIBE_LANGUAGE = None           # e.g. "en" to force a language, or None to auto-detect

# --- Caption appearance ----------------------------------------------------
CAPTION_COLOR = "#E7B95F"            # font color (hex)
FONT_NAME = "Arial"                  # any font installed on the system
FONT_SIZE = 64                       # in pixels (relative to the 1080x2350 canvas)
FONT_BOLD = True
OUTLINE_COLOR = "#000000"            # thin outline for legibility
OUTLINE_WIDTH = 2                    # 0 to disable
SHADOW_DEPTH = 0                     # drop-shadow distance in px (0 = none)
MAX_CHARS_PER_LINE = 28              # wrap captions to this width
MAX_LINES = 2                        # max lines shown at once

# --- Caption position ------------------------------------------------------
# The video canvas. Used as a fallback if ffprobe can't read the file.
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 2350

# How to find the bottom black bar:
#   AUTO_DETECT_BARS = True  -> sample the video and detect the actual content
#                               box, then center captions in the bottom bar.
#   AUTO_DETECT_BARS = False -> assume a width-fit 16:9 clip centered vertically.
AUTO_DETECT_BARS = True

# Force the vertical center (in px from the top) of the caption block. Overrides
# all detection when set. Leave as None to use detection / the 16:9 assumption.
CAPTION_CENTER_Y = None

# Nudge the auto-computed center up (-) or down (+) by this many pixels.
# Default lifts captions a little for extra breathing room above the bottom
# edge (the bottom is also where IG/Threads overlay their UI). Set to 0 for
# dead-center in the bottom bar, or make it more negative to raise further.
CAPTION_Y_OFFSET = -32

# Keep the caption block at least this many pixels from the bottom edge and
# from the bottom of the video content, so it never gets clipped or overlaps.
CAPTION_SAFE_MARGIN = 64

# Line height as a multiple of FONT_SIZE (used to size the caption block so it
# fits inside the bottom bar without running off the canvas).
LINE_HEIGHT_FACTOR = 1.25

# Aspect ratio of the inner clip (used only when AUTO_DETECT_BARS is False).
CLIP_ASPECT_W = 16
CLIP_ASPECT_H = 9

# ---------------------------------------------------------------------------
# END CONFIG
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------
def _find_tool(name: str) -> str:
    """Return a path to ffmpeg/ffprobe, checking PATH then the winget install."""
    found = shutil.which(name)
    if found:
        return found
    # winget (Gyan.FFmpeg) drops binaries under LOCALAPPDATA; the PATH update
    # only applies to new shells, so search for it directly as a fallback.
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.exists():
        matches = list(base.rglob(f"{name}.exe"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(
        f"Could not find '{name}'. Open a new terminal (so the PATH refresh "
        f"from the winget install takes effect) or install ffmpeg."
    )


FFMPEG = _find_tool("ffmpeg")
FFPROBE = _find_tool("ffprobe")


def _run(cmd, **kwargs):
    """Run a command, capturing output; raise with stderr on failure."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
            f"{result.stderr.strip()}"
        )
    return result


# ---------------------------------------------------------------------------
# Color handling (hex -> ASS &HAABBGGRR)
# ---------------------------------------------------------------------------
def hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    """Convert '#RRGGBB' to ASS color '&HAABBGGRR' (alpha 0 = opaque)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected #RRGGBB, got '{hex_color}'")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------
def pick_input_video() -> Path:
    if not ASSETS_DIR.exists():
        raise FileNotFoundError(f"Assets folder not found: {ASSETS_DIR}")
    if INPUT_VIDEO_NAME:
        path = ASSETS_DIR / INPUT_VIDEO_NAME
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        return path
    candidates = [
        p for p in ASSETS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        and "_captioned" not in p.stem
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No video files found in {ASSETS_DIR}. Drop your raw video there "
            f"or set INPUT_VIDEO_NAME."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Probing & letterbox detection
# ---------------------------------------------------------------------------
def get_dimensions(video: Path):
    try:
        out = _run([
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(video),
        ]).stdout
        stream = json.loads(out)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception as e:
        print(f"  ! ffprobe could not read dimensions ({e}); using canvas defaults.")
        return CANVAS_WIDTH, CANVAS_HEIGHT


def detect_content_box(video: Path, width: int, height: int):
    """
    Use ffmpeg's cropdetect to find the non-black content region.
    Returns (x, y, w, h) of the inner clip, or None if detection fails.
    """
    try:
        # Scan the whole clip at 2 fps. cropdetect (reset=0) accumulates the
        # bounding box of all non-black content across every sampled frame, so
        # the final crop is the union -- it won't shrink to a single frame where
        # the subject happens to be small/dark. round=2 keeps dims even.
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", str(video),
             "-vf", "fps=2,cropdetect=24:2:0", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
        if not crops:
            return None
        w, h, x, y = (int(v) for v in crops[-1])
        # Sanity: content must be inside the canvas and reasonably tall.
        if w <= 0 or h <= 0 or h > height or w > width:
            return None
        return x, y, w, h
    except Exception:
        return None


def compute_caption_center_y(video: Path, width: int, height: int) -> int:
    """Pixel y-coordinate to center the caption block on the bottom black bar."""
    if CAPTION_CENTER_Y is not None:
        return CAPTION_CENTER_Y + CAPTION_Y_OFFSET

    content_bottom = None
    if AUTO_DETECT_BARS:
        box = detect_content_box(video, width, height)
        if box:
            _, y, _, h = box
            content_bottom = y + h
            print(f"  detected clip content box bottom at y={content_bottom}")

    if content_bottom is None:
        # Assume a width-fit clip of the configured aspect, centered vertically.
        clip_h = round(width * CLIP_ASPECT_H / CLIP_ASPECT_W)
        top_bar = (height - clip_h) // 2
        content_bottom = top_bar + clip_h
        print(f"  assuming centered {CLIP_ASPECT_W}:{CLIP_ASPECT_H} clip; "
              f"content bottom at y={content_bottom}")

    # Vertically center the *caption block* (not just a baseline) within the
    # bottom black bar, then clamp so the block stays on-canvas and clear of the
    # content. block_half uses the worst case (MAX_LINES) so 2-line captions fit.
    block_half = (FONT_SIZE * LINE_HEIGHT_FACTOR * MAX_LINES) / 2
    bar_top = min(content_bottom, height)
    bar_height = height - bar_top

    if bar_height < FONT_SIZE * LINE_HEIGHT_FACTOR:
        # No real bottom bar detected (content runs to the edge). Sit just above
        # the bottom safe margin rather than overlapping the very edge.
        center = height - CAPTION_SAFE_MARGIN - block_half
        print("  ! little/no bottom bar found; placing caption near bottom edge.")
    else:
        bar_center = (bar_top + height) / 2
        # Allowed range for the block's center: below the content (+margin),
        # and above the bottom edge (-margin).
        lo = bar_top + CAPTION_SAFE_MARGIN + block_half
        hi = height - CAPTION_SAFE_MARGIN - block_half
        if lo > hi:
            # Block taller than the usable bar; pin to the bottom safe area and
            # warn -- a smaller FONT_SIZE would fit better.
            center = hi
            print(f"  ! caption block (~{int(block_half * 2)}px) is taller than the "
                  f"bottom bar (~{int(bar_height)}px); consider a smaller FONT_SIZE.")
        else:
            center = max(lo, min(bar_center, hi))

    return int(round(center)) + CAPTION_Y_OFFSET


# ---------------------------------------------------------------------------
# Audio extraction & transcription
# ---------------------------------------------------------------------------
def extract_audio(video: Path, out_path: Path):
    """Mono 16kHz mp3 keeps the file small (well under the API's 25MB limit)."""
    _run([
        FFMPEG, "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(out_path),
    ])


def transcribe(audio_path: Path):
    """Return a list of {start, end, text} segments via local faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "The 'faster-whisper' package is required:  pip install faster-whisper"
        )

    print(f"  loading model '{WHISPER_MODEL}' on {WHISPER_DEVICE} "
          f"({WHISPER_COMPUTE_TYPE})... (first run downloads it)")
    try:
        model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    except Exception as e:
        sys.exit(
            f"Could not load faster-whisper model '{WHISPER_MODEL}' on "
            f"{WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}: {e}\n"
            "Try WHISPER_DEVICE='cpu' and WHISPER_COMPUTE_TYPE='int8'."
        )

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=TRANSCRIBE_LANGUAGE,   # None => auto-detect
        vad_filter=True,                # skip silence; tightens segment timing
        beam_size=5,
    )
    print(f"  detected language: {info.language} "
          f"(p={info.language_probability:.2f})")

    out = []
    for seg in segments_iter:
        text = clean_segment_text(seg.text or "")
        if text:
            out.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
    return out


def clean_segment_text(text: str) -> str:
    """Tidy a raw whisper segment.

    faster-whisper splits sentences across segments, so a segment often begins
    with the leading whitespace/punctuation that joined it to the previous one
    (e.g. ", Hey..." or the low-quote comma "‚Hey..."). Strip every leading
    character that is punctuation (P*), a separator/space (Z*), a symbol (S*),
    or invisible/control (C*) -- this catches all Unicode comma variants without
    touching commas *inside* the sentence. Then collapse internal whitespace.
    """
    text = text.strip()
    i = 0
    while i < len(text) and unicodedata.category(text[i])[0] in ("P", "Z", "S", "C"):
        i += 1
    return " ".join(text[i:].split())


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    """Seconds -> ASS timestamp H:MM:SS.cc"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def wrap_caption(text: str) -> str:
    """Wrap to MAX_CHARS_PER_LINE / MAX_LINES, joined with ASS line breaks (\\N)."""
    # Final safety net: strip leading-comma artifacts again right before render,
    # in case text reached here from anywhere other than transcribe().
    text = clean_segment_text(text)
    lines = textwrap.wrap(text, width=MAX_CHARS_PER_LINE) or [text]
    if len(lines) > MAX_LINES:
        # Re-wrap a little wider so it fits within MAX_LINES where possible.
        wider = textwrap.wrap(text, width=max(MAX_CHARS_PER_LINE,
                                              len(text) // MAX_LINES + 1))
        lines = wider[:MAX_LINES]
    # Escape ASS-special characters.
    lines = [ln.replace("\\", "").replace("{", "(").replace("}", ")") for ln in lines]
    return "\\N".join(lines)


def build_ass(segments, width, height, center_y) -> str:
    primary = hex_to_ass(CAPTION_COLOR)
    outline = hex_to_ass(OUTLINE_COLOR)
    bold = -1 if FONT_BOLD else 0
    cx = width // 2

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{FONT_NAME},{FONT_SIZE},{primary},{primary},{outline},&H00000000,{bold},0,0,0,100,100,0,0,1,{OUTLINE_WIDTH},{SHADOW_DEPTH},5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # \an5 = anchor at the text block's center; \pos puts that center exactly
    # on the middle of the bottom black bar.
    lines = []
    for seg in segments:
        text = wrap_caption(seg["text"])
        start = fmt_time(seg["start"])
        end = fmt_time(seg["end"])
        body = f"{{\\an5\\pos({cx},{center_y})}}{text}"
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{body}")

    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Burn captions
# ---------------------------------------------------------------------------
def burn_captions(video: Path, ass_path: Path, output: Path):
    # Run ffmpeg from the assets dir and reference the .ass by basename, which
    # sidesteps Windows path-escaping headaches inside the filtergraph.
    _run(
        [
            FFMPEG, "-y", "-i", str(video.name),
            "-vf", f"ass={ass_path.name}",
            "-c:a", "copy",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            str(output.name),
        ],
        cwd=str(ASSETS_DIR),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("== AI Caption Generator ==")
    print(f"ffmpeg : {FFMPEG}")

    video = pick_input_video()
    print(f"input  : {video.name}")

    out_name = OUTPUT_VIDEO_NAME or f"{video.stem}_captioned.mp4"
    output = ASSETS_DIR / out_name

    width, height = get_dimensions(video)
    print(f"size   : {width} x {height}")

    center_y = compute_caption_center_y(video, width, height)
    print(f"caption center y: {center_y}")

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.mp3"
        print("extracting audio...")
        extract_audio(video, audio)

        print("transcribing locally with faster-whisper...")
        segments = transcribe(audio)
        print(f"  {len(segments)} caption segment(s)")
        if not segments:
            sys.exit("No speech detected; nothing to caption.")

    ass_path = ASSETS_DIR / f"{video.stem}.ass"
    ass_path.write_text(build_ass(segments, width, height, center_y), encoding="utf-8")
    print(f"wrote subtitles: {ass_path.name}")

    print("burning captions (this is the slow part)...")
    burn_captions(video, ass_path, output)

    print(f"\nDone -> {output}")


if __name__ == "__main__":
    main()
