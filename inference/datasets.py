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

# The single source of truth for what gets downloaded per modality. Each entry:
#   dataset     huggingface dataset id
#   config      optional config/subset name (None = default)
#   license     approved license
#   map         unified-schema column mapping:
#                 text        the caption / target text column
#                 prompt      optional instruction column prepended to text
#                 media       optional media byte column
#                 media_kind  image | audio | video
# Approved sources (verified). TriSense-2M and gated datasets are excluded.
DATASET_SOURCES = {
    "images": [
        {"dataset": "HuggingFaceM4/COCO", "license": "permissive",
         "map": {"text": "sentences_raw", "media": "image", "media_kind": "image"}},
        {"dataset": "liuhaotian/LLaVA-Instruct-150K", "license": "Apache-2.0",
         "map": {"text": "conversations", "media": "image", "media_kind": "image"}},
        {"dataset": "sbu_captions", "license": "CC-BY",
         "map": {"text": "caption", "media": "image", "media_kind": "image"}},
        {"dataset": "HuggingFaceM4/TextCaps", "license": "CC-BY-4.0",
         "map": {"text": "caption_str", "media": "image", "media_kind": "image"}},
    ],
    "videos": [
        {"dataset": "ontocord/VALID", "license": "permissive",
         "map": {"text": "caption", "media": "video", "media_kind": "video"}},
        {"dataset": "encord-team/E-MM1-100M", "license": "permissive",
         "map": {"text": "text", "media": "video", "media_kind": "video"}},
        {"dataset": "JavisVerse/MM-PreTrain", "license": "permissive",
         "map": {"text": "caption", "media": "video", "media_kind": "video"}},
        {"dataset": "ngqtrung/full-modality-video-caption", "license": "permissive",
         "map": {"text": "caption", "media": "video", "media_kind": "video"}},
    ],
    "audio": [
        {"dataset": "facebook/voxpopuli", "config": "en", "license": "CC0",
         "map": {"text": "raw_text", "media": "audio", "media_kind": "audio"}},
        {"dataset": "facebook/multilingual_librispeech", "config": "en", "license": "CC-BY-4.0",
         "map": {"text": "text", "media": "audio", "media_kind": "audio"}},
        {"dataset": "MLCommons/peoples_speech", "license": "CC-BY / CC-BY-SA",
         "map": {"text": "text", "media": "audio", "media_kind": "audio"}},
        {"dataset": "google/fleurs", "config": "en_us", "license": "CC-BY-4.0",
         "map": {"text": "transcription", "media": "audio", "media_kind": "audio"}},
        {"dataset": "PolyAI/minds14", "config": "en-AU", "license": "CC-BY-4.0",
         "map": {"text": "transcription", "media": "audio", "media_kind": "audio"}},
    ],
    "documents": [
        {"dataset": "allenai/olmOCR-mix-0225", "license": "ODC-By",
         "map": {"text": "text"}},
        {"dataset": "allenai/olmOCR-mix-1025", "config": "00_documents", "license": "ODC-By",
         "map": {"text": "text"}},
        {"dataset": "pixparse/pdfa-eng-wds", "license": "permissive",
         "map": {"text": "text"}},
        {"dataset": "Cognitive-Lab/NayanaOCR_Corpus_2025", "license": "permissive",
         "map": {"text": "text"}},
        {"dataset": "qihoo360/InduOCRBench", "license": "permissive",
         "map": {"text": "text"}},
    ],
    "spreadsheets": [
        {"dataset": "wenge-research/TableEval", "license": "permissive",
         "map": {"text": "text"}},
        {"dataset": "FinWorkBench/Finch", "license": "permissive",
         "map": {"text": "text"}},
    ],
    "presentations": [
        {"dataset": "Forceless/Zenodo10K", "license": "permissive",
         "map": {"text": "text"}},
        {"dataset": "tyrionhuu/PPTBench-Understanding", "license": "permissive",
         "map": {"text": "text"}},
        {"dataset": "NerdyVisky/RealSlide", "license": "permissive",
         "map": {"text": "text"}},
    ],
    "markdown": [
        {"dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "license": "ODC-By",
         "map": {"text": "text"}},
        {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "license": "ODC-By",
         "map": {"text": "text"}},
        {"dataset": "wikimedia/wikipedia", "config": "20231101.en", "license": "CC BY-SA",
         "map": {"text": "text"}},
        {"dataset": "open-index/open-markdown-v2", "license": "permissive",
         "map": {"text": "text"}},
        {"dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K", "license": "MIT",
         "map": {"text": "solution", "prompt": "problem"}},
    ],
}

MEDIA_SUFFIX = {"image": ".png", "audio": ".wav", "video": ".mp4"}


def get_sources(modality):
    if modality not in DATASET_SOURCES:
        raise ValueError(
            f"Unknown modality {modality!r}. Available: {list(DATASET_SOURCES)}"
        )
    return list(DATASET_SOURCES[modality])


def list_sources():
    for modality, specs in DATASET_SOURCES.items():
        print(f"[{modality}]")
        for spec in specs:
            config = f" ({spec.get('config')})" if spec.get("config") else ""
            print(f"  - {spec['dataset']}{config} [{spec['license']}]")


def _map_source_row(spec, row, extractMedia=False, samplingRate=None):
    """Map one raw source row into the unified schema."""
    mapping = spec["map"]
    prompt = row.get(mapping["prompt"]) if mapping.get("prompt") else None
    text = row.get(mapping["text"]) if mapping.get("text") else row.get("text")
    if text is None:
        text = ""

    media = row.get(mapping["media"]) if mapping.get("media") else None
    if media is not None and extractMedia:
        return _media_bytes_to_row(
            mapping["media_kind"], media,
            caption=(prompt + "\n" + text if prompt else text),
            samplingRate=samplingRate,
        )
    if prompt:
        text = f"{prompt}\n{text}"
    return {"text": text, "vision": None, "audio": None}


def _media_bytes_to_row(kind, data, caption="", samplingRate=None):
    import secrets
    import tempfile

    from . import utils

    tmp = Path(tempfile.gettempdir()) / f"{secrets.token_hex(16)}{MEDIA_SUFFIX[kind]}"
    try:
        tmp.write_bytes(data)
        return tokenizeFileToRow(
            tmp, caption=caption, includeMedia=True, samplingRate=samplingRate
        )
    finally:
        tmp.unlink(missing_ok=True)


def load_modality(modality, split="train", extractMedia=True, samplingRate=None):
    """Download a modality dataset from third-party sources only.

    Ingests every approved raw source in DATASET_SOURCES for the modality,
    mapping each row into the unified schema. Published axonura repos are
    deliberately not used.
    """
    from datasets import concatenate_datasets, load_dataset

    specs = get_sources(modality)
    built = []
    for spec in specs:
        try:
            ds = load_dataset(spec["dataset"], spec.get("config") or None, split=split)
        except Exception as exc:
            logger.warning("Skipping source %s: %s", spec["dataset"], exc)
            continue
        ds = ds.map(
            lambda row, s=spec: _map_source_row(s, row, extractMedia, samplingRate),
            remove_columns=ds.column_names,
        )
        ds = ds.filter(lambda row: bool((row.get("text") or "").strip()))
        built.append(ds)
        logger.info("Ingested %s rows from %s", len(ds), spec["dataset"])

    if not built:
        raise RuntimeError(f"No raw source available for modality: {modality}")
    return concatenate_datasets(built)


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
