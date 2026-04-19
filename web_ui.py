import os
import time
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from predict_mood import OUTPUT_DIR, predict_mood

app = Flask(__name__)

UPLOAD_DIR = os.path.join(OUTPUT_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'.flac', '.m4a', '.mp3', '.ogg', '.wav'}
ALLOWED_EXTENSIONS_STR = ', '.join(sorted(ALLOWED_EXTENSIONS))
FILE_INPUT_ACCEPT = ','.join(sorted(ALLOWED_EXTENSIONS))


def _is_allowed(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def _log_request(status, file_name=None, saved_path=None, result=None, error=None, route='POST /'):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {route} status={status}", flush=True)

    if file_name:
        print(f"  original_file={file_name}", flush=True)
    if saved_path:
        print(f"  saved_path={saved_path}", flush=True)
    if result:
        print(
            f"  mood={result['mood']} votes={result['votes']}/3 segments={result['segment_predictions']}",
            flush=True,
        )
        if result.get('confidence'):
            top_mood, top_score = next(iter(result['confidence'].items()))
            print(f"  top_confidence={top_mood}:{top_score}%", flush=True)
    if error:
        print(f"  error={error}", flush=True)


def _process_prediction_request(file_obj, prediction_verbose=False, route='POST /'):
    if not file_obj or not file_obj.filename:
        error = 'Please choose an audio file.'
        _log_request(status='failed', error=error, route=route)
        return None, None, error, 400

    if not _is_allowed(file_obj.filename):
        error = f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS_STR}"
        _log_request(status='failed', file_name=file_obj.filename, error=error, route=route)
        return None, None, error, 400

    safe_name = secure_filename(file_obj.filename)
    base, ext = os.path.splitext(safe_name)
    saved_name = f"{base}_{int(time.time())}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file_obj.save(saved_path)

    try:
        if prediction_verbose:
            started_at = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{started_at}] {route} prediction_started file={saved_name}", flush=True)

        result = predict_mood(saved_path, verbose=prediction_verbose)

        if prediction_verbose:
            ended_at = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ended_at}] {route} prediction_finished file={saved_name}", flush=True)

        _log_request(
            status='success',
            file_name=file_obj.filename,
            saved_path=saved_path,
            result=result,
            route=route,
        )
        return result, saved_name, None, 200
    except Exception as exc:
        error = f"Prediction failed: {exc}"
        _log_request(
            status='failed',
            file_name=file_obj.filename,
            saved_path=saved_path,
            error=error,
            route=route,
        )
        return None, saved_name, error, 500


@app.route('/', methods=['GET'])
def index():
    # Only render the template; all predictions are handled via JS fetch to /predict
    return render_template(
        'index.html',
        result=None,
        error=None,
        uploaded_name=None,
        allowed_extensions=ALLOWED_EXTENSIONS_STR,
        file_input_accept=FILE_INPUT_ACCEPT,
    )


@app.route('/predict', methods=['POST'])
def predict_api():
    result, uploaded_name, error, status_code = _process_prediction_request(
        request.files.get('audio'),
        prediction_verbose=True,
        route='POST /predict',
    )

    if error:
        return jsonify({'ok': False, 'error': error, 'uploaded_name': uploaded_name}), status_code

    return jsonify({'ok': True, 'result': result, 'uploaded_name': uploaded_name}), 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
