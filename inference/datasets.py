# Copyright 2026 First Person
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = (
    ".txt", ".md", ".csv", ".json", ".log", ".xml", ".py", ".js", ".html",
    ".pdf", ".docx", ".xlsx", ".pptx",
)
MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".avi",
                    ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac", ".ogg", ".m4a")

MODALITIES = ("images", "videos", "audio", "documents", "spreadsheets",
              "presentations", "markdown")

# One unified media template per modality. The special token marks the anchor
# position where the adapter output is spliced into the token stream.
MEDIA_PROMPTS = {
    "image": "<image>\nDescribe this image in detail.",
    "video": "<video>\nDescribe what happens in this video.",
    "audio": "<audio>\nTranscribe and summarize this audio.",
}

# Extension routing: file suffix -> modality slot. Unlisted text-ish suffixes
# fall through to the plain-text branch of utils.tokenizeFile().
EXTENSION_MODALITY = {
    ".pdf": "documents",
    ".docx": "documents",
    ".xlsx": "spreadsheets",
    ".csv": "spreadsheets",
    ".pptx": "presentations",
    ".md": "markdown",
    ".txt": "markdown",
}


def tokenizeFileToRow(path, caption=None, includeMedia=False, samplingRate=None):
    """Open a single file with utils.tokenizeFile() and convert the result into
    a unified dataset row.

    Returns a dict or raises (corrupt / unsupported files propagate so the
    caller can drop them).
    """
    from . import utils

    kind, *payload = utils.tokenizeFile(path, samplingRate=samplingRate or utils.AUDIO_SAMPLE_RATE)

    if kind == "text":
        return {"text": payload[0], "vision": None, "audio": None}
    if kind == "image":
        if not includeMedia:
            raise ValueError("Media files disabled; set includeMedia=True to keep images.")
        return {
            "text": caption or MEDIA_PROMPTS["image"],
            "vision": payload[0],
            "audio": None,
        }
    if kind == "video":
        if not includeMedia:
            raise ValueError("Media files disabled; set includeMedia=True to keep videos.")
        return {
            "text": caption or MEDIA_PROMPTS["video"],
            "vision": payload[0],
            "audio": payload[1],
        }
    if kind == "audio":
        if not includeMedia:
            raise ValueError("Media files disabled; set includeMedia=True to keep audio.")
        return {
            "text": caption or MEDIA_PROMPTS["audio"],
            "vision": None,
            "audio": payload[0],
        }
    raise ValueError(f"Unexpected tokenizeFile result kind: {kind}")


def discoverFiles(folder, extensions=None, recursive=True):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    pattern = "**/*" if recursive else "*"
    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        yield path


def ingest_folder(folder, extensions=None, includeMedia=False, recursive=True,
                  captions=None, samplingRate=None):
    """Custom-corpus pipeline over a local folder.

    Yields shards (lists of unified rows). Every file is re-opened through
    utils.tokenizeFile(); corrupt or unsupported files are logged and skipped,
    so a bad file never aborts the whole scan.

    `captions` is an optional callable path -> caption used only for media rows.
    """
    extList = list(extensions) if extensions else (None if includeMedia else list(TEXT_EXTENSIONS))
    shard = []
    for path in discoverFiles(folder, extensions=extList, recursive=recursive):
        try:
            caption = captions(str(path)) if callable(captions) else None
            row = tokenizeFileToRow(
                path, caption=caption, includeMedia=includeMedia, samplingRate=samplingRate
            )
        except Exception as exc:
            logger.warning("Dropping %s: %s", path, exc)
            continue
        shard.append(row)
        if len(shard) >= 1000:
            yield shard
            shard = []
    if shard:
        yield shard


def rows_to_dataset(shards):
    """Assemble shards of unified rows into a datasets.Dataset (lazy import)."""
    from datasets import Dataset

    rows = [row for shard in shards for row in shard]
    if not rows:
        raise ValueError("No rows to assemble.")
    return Dataset.from_list(rows)
