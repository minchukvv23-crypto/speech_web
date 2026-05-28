import os
import tempfile
import threading
import uuid
from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file

from app.pipeline import process_audio, set_progress_callback
from app.utils.cleanup import cleanup_paths

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = "/workspace"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

jobs = {}


def make_job():
    return {
        "status": "queued",
        "stage": "Ожидание",
        "progress": 0,
        "eta_sec": None,
        "result": None,
        "result_text": None,
        "error": None,
        "input_path": None,
        "download_name": None,
    }


def update_job(job_id: str, stage: str, progress: int, eta_sec, status: str):
    job = jobs.get(job_id)
    if not job:
        return

    job["stage"] = stage
    job["progress"] = progress
    job["eta_sec"] = eta_sec
    job["status"] = status


def run_job(job_id: str, input_path: str, enhance_mode: str):
    def progress_cb(stage, progress, eta_sec, status):
        update_job(job_id, stage, progress, eta_sec, status)

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["input_path"] = input_path

        set_progress_callback(progress_cb)

        segments, result_text = process_audio(
            input_path=input_path,
            enhance_mode=enhance_mode,
            asr_model="large",
            device="cuda",
        )

        jobs[job_id]["result"] = {"segments": segments}
        jobs[job_id]["result_text"] = result_text
        jobs[job_id]["status"] = "done"
        jobs[job_id]["stage"] = "Готово"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["eta_sec"] = 0

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        jobs[job_id]["download_name"] = f"{base_name}.txt"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["stage"] = "Ошибка"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["eta_sec"] = 0
        jobs[job_id]["error"] = str(e)

    finally:
        set_progress_callback(None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Файл не выбран"}), 400

    enhance_mode = request.form.get("enhance_mode", "none")
    if enhance_mode not in {"none", "medium", "strong"}:
        return jsonify({"error": "Некорректный режим улучшения"}), 400

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        return jsonify({"error": "Неподдерживаемый формат файла"}), 400

    job_id = uuid.uuid4().hex
    jobs[job_id] = make_job()

    fd, temp_path = tempfile.mkstemp(prefix=f"{job_id}_", suffix=suffix, dir=UPLOAD_DIR)
    os.close(fd)
    file.save(temp_path)

    worker = threading.Thread(
      target=run_job,
      args=(job_id, temp_path, enhance_mode),
      daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404

    return jsonify({
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "eta_sec": job["eta_sec"],
        "result": job["result"],
        "error": job["error"],
    })


@app.route("/api/download/<job_id>", methods=["GET"])
def download(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404

    if job["status"] != "done":
        return jsonify({"error": "Результат еще не готов"}), 400

    if not job["result_text"]:
        return jsonify({"error": "Текст результата пуст"}), 400

    content = job["result_text"].encode("utf-8")
    filename = job.get("download_name") or "result.txt"

    response = send_file(
        BytesIO(content),
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=filename,
    )

    input_path = job.get("input_path")
    cleanup_paths([input_path])
    jobs.pop(job_id, None)

    return response


@app.route("/api/delete/<job_id>", methods=["POST"])
def delete_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404

    input_path = job.get("input_path")
    cleanup_paths([input_path])
    jobs.pop(job_id, None)

    return jsonify({"message": "Все данные удалены"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
