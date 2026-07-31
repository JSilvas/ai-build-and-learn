"""Local viewer for generated image/video artifacts. Run with: uv run python gallery/app.py"""
from flask import Flask, jsonify, render_template, send_from_directory

import store

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/items")
def list_items():
    items = sorted(store.load_items(), key=lambda i: i["created_at"], reverse=True)
    return jsonify(items)


@app.delete("/api/items/<item_id>")
def remove_item(item_id):
    if not store.delete_item(item_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


@app.get("/media/<path:filename>")
def media(filename):
    return send_from_directory(store.MEDIA_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
