"""Persistence for generated image/video artifacts and their metadata."""
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

GALLERY_DIR = Path(__file__).parent
MEDIA_DIR = GALLERY_DIR / "media"
METADATA_PATH = GALLERY_DIR / "metadata.json"

MEDIA_DIR.mkdir(exist_ok=True)


def load_items() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    return json.loads(METADATA_PATH.read_text())


def _write_items(items: list[dict]) -> None:
    METADATA_PATH.write_text(json.dumps(items, indent=2))


def _new_entry(kind: str, filename: str, entry_id: str, prompt: str, model: str, params: dict) -> dict:
    entry = {
        "id": entry_id,
        "type": kind,
        "filename": filename,
        "prompt": prompt,
        "model": model,
        "params": params,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items = load_items()
    items.append(entry)
    _write_items(items)
    return entry


def save_image(image, prompt: str, model: str, params: dict) -> dict:
    entry_id = uuid.uuid4().hex[:12]
    filename = f"{entry_id}.png"
    image.save(MEDIA_DIR / filename)
    return _new_entry("image", filename, entry_id, prompt, model, params)


def save_video(video_path, prompt: str, model: str, params: dict) -> dict:
    entry_id = uuid.uuid4().hex[:12]
    filename = f"{entry_id}.mp4"
    shutil.copy(video_path, MEDIA_DIR / filename)
    return _new_entry("video", filename, entry_id, prompt, model, params)


def delete_item(item_id: str) -> bool:
    items = load_items()
    match = next((i for i in items if i["id"] == item_id), None)
    if match is None:
        return False
    (MEDIA_DIR / match["filename"]).unlink(missing_ok=True)
    _write_items([i for i in items if i["id"] != item_id])
    return True
