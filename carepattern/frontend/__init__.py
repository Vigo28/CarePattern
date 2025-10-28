import os
import json
from flask import Flask, flash, render_template, render_template_string, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from .config_loader import load_ini_config
from carepattern.core.jobs import create_job, get_job
from carepattern.core.video import start_processing

def create_app(config=None):
    app = Flask(__name__)

    if config is None:
        load_ini_config(app, "config.ini")
    else:
        app.config.from_mapping(config)

    # defaults
    app.config.setdefault('ALLOWED_EXTENSIONS', {'mp4'})
    app.config.setdefault('YOLO_POSE_MODEL', 'yolo11n-pose.pt')

    try:
        os.makedirs(app.instance_path, exist_ok=True)
        uploads_path = os.path.join(app.instance_path, 'uploads')
        os.makedirs(uploads_path, exist_ok=True)
        app.config.setdefault('UPLOAD_FOLDER', uploads_path)
    except OSError:
        pass

    app = create_routes(app)

    return app

def create_routes(app):
    @app.route('/')
    def render_root():
        uploads_dir = app.config['UPLOAD_FOLDER']
        try:
            folders = []
            for entry in sorted(os.listdir(uploads_dir)):
                entry_path = os.path.join(uploads_dir, entry)
                if os.path.isdir(entry_path):
                    files = [f for f in sorted(os.listdir(entry_path))
                             if os.path.isfile(os.path.join(entry_path, f))]
                    folder_data = {
                        'name': entry,
                        'raw': next((f"{entry}/{file}" for file in files if file == 'raw.mp4'), None),
                        'overlay': next((f"{entry}/{file}" for file in files if file == 'overlay.mp4'), None),
                        'skeleton': next((f"{entry}/{file}" for file in files if file == 'skeleton.mp4'), None),
                        'prediction': next((f"{entry}/{file}" for file in files if file == 'prediction.txt'), None),
                        'prediction_content': None,
                        'job_id': None
                    }
                    # read prediction content
                    prediction_file = os.path.join(entry_path, 'prediction.txt')
                    if os.path.exists(prediction_file):
                        with open(prediction_file, 'r') as f:
                            folder_data['prediction_content'] = f.read()
                    # read job id if present
                    job_meta = os.path.join(entry_path, 'job.json')
                    if os.path.exists(job_meta):
                        try:
                            with open(job_meta, 'r') as jf:
                                j = json.load(jf)
                                folder_data['job_id'] = j.get('job_id')
                        except Exception:
                            folder_data['job_id'] = None
                    folders.append(folder_data)
        except OSError:
            folders = []

        return render_template('root.html', folders=folders)


    @app.route('/config')
    def render_config():
        return render_template('config.html')

    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        def allowed_file(filename):
            return '.' in filename and \
                filename.rsplit('.', 1)[1].lower() in app.config.get('ALLOWED_EXTENSIONS', set())

        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                filename_no_ext = os.path.splitext(filename)[0]
                file_ext = os.path.splitext(filename)[1]
                file_folder = os.path.join(app.config['UPLOAD_FOLDER'], filename_no_ext)
                os.makedirs(file_folder, exist_ok=True) # make the subfolder

                save_path = os.path.join(file_folder, f'raw{file_ext}') # save the raw video
                file.save(save_path)

                job_id = create_job()
                output_path = os.path.join(file_folder, 'overlay.mp4')
                start_processing(save_path, output_path, job_id, model_path=app.config.get('YOLO_POSE_MODEL'))

                # persist job id next to the files so the root view can poll
                try:
                    job_meta = os.path.join(file_folder, 'job.json')
                    with open(job_meta, 'w') as jf:
                        json.dump({"job_id": job_id}, jf)
                except Exception:
                    # non-fatal: don't break uploads if writing fails
                    pass

                # If AJAX / XHR upload, return job id so frontend can poll
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
                    return jsonify({"job_id": job_id}), 202

                return redirect(url_for('yolo_progress', job_id=job_id))

        # GET request
        return render_template('uploads.html')


    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/yolo/progress/<job_id>')
    def yolo_progress(job_id):
        job = get_job(job_id)
        if not job:
            return "unknown job", 404

        # inline template unchanged...
        template = r'''
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Processing Progress</title>
          <style>
            #spinner { font-size: 1.2rem; }
            #bar { width: 100%; background: #eee; height: 18px; border-radius: 4px; margin-top: 8px; }
            #fill { height: 100%; width: 0%; background: #2b8cff; border-radius: 4px; transition: width 300ms; }
          </style>
        </head>
        <body>
          <a href="{{ url_for('render_root') }}">Home</a>
          <h3>Processing job: {{ job_id }}</h3>
          <div id="spinner">Processing... <span id="percent">0\%</span></div>
          <div id="bar"><div id="fill"></div></div>
          <div id="result" style="margin-top:12px;"></div>

          <script>
          const jobId = "{{ job_id }}";
          const statusUrl = `/yolo/status/${jobId}`;
          const resultUrl = `/yolo/result/${jobId}`;

          async function poll() {
            try {
              const r = await fetch(statusUrl);
              if (!r.ok) throw new Error('status fetch failed');
              const js = await r.json();
              const p = js.progress || 0;
              document.getElementById('fill').style.width = p + '%';
              document.getElementById('percent').textContent = p + '%';

              if (js.status === 'done') {
                document.getElementById('spinner').textContent = 'Done';
                document.getElementById('result').innerHTML = `<a href="${resultUrl}" target="_blank">Download overlay.mp4</a>`;
                clearInterval(t);
              } else if (js.status === 'error') {
                document.getElementById('spinner').textContent = 'Error';
                document.getElementById('result').textContent = js.error || 'Processing error';
                clearInterval(t);
              }
            } catch (err) {
              console.error(err);
            }
          }

          // Initial poll and regular polling
          poll();
          const t = setInterval(poll, 1500);
          </script>
        </body>
        </html>
        '''
        return render_template_string(template, job_id=job_id)

    # status endpoint used by frontend to poll job progress/status
    @app.route('/yolo/status/<job_id>')
    def yolo_status(job_id):
        job = get_job(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(job)

    # download endpoint for completed output
    @app.route('/yolo/result/<job_id>')
    def yolo_result(job_id):
        job = get_job(job_id)
        if not job:
            return "unknown job", 404
        if job.get("status") != "done":
            return "not ready", 400
        output_path = job.get("output")
        if not output_path:
            return "file missing", 404
        p = os.path.abspath(output_path)
        if not os.path.exists(p):
            return "file missing", 404
        return send_from_directory(directory=os.path.dirname(p), path=os.path.basename(p), as_attachment=True)

    if app.config.get('DEBUG', False):
        @app.route('/uploads/debug/<path:filename>')
        def upload_debug(filename):
            p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(p):
                return jsonify({"exists": False, "path": p}), 404
            size = os.path.getsize(p)
            mtime = os.path.getmtime(p)
            with open(p, 'rb') as f:
                head = f.read(64)
            return jsonify({
                "exists": True,
                "path": p,
                "size": size,
                "mtime": mtime,
                "head_hex": head.hex()
            })

    return app