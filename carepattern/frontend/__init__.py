import os
import json
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from .config_loader import load_ini_config

def create_app(config=None):
    app = Flask(__name__)

    if config is None:
        load_ini_config(app, "config.ini")
    else:
        app.config.from_mapping(config)

    try:
        os.makedirs(app.instance_path, exist_ok=True)
        uploads_path = os.path.join(app.instance_path, 'uploads')
        os.makedirs(uploads_path, exist_ok=True)
    except OSError:
        pass

    app.config.setdefault('UPLOAD_FOLDER', uploads_path)

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
                        'overlay': next((f"{entry}/{file}" for file in files if file == 'skeleton-overlay.mp4'), None),
                        'skeleton': next((f"{entry}/{file}" for file in files if file == 'skeleton.mp4'), None),
                        'prediction': next((f"{entry}/{file}" for file in files if file == 'prediction.txt'), None)
                    }
                    folders.append(folder_data)
        except OSError:
            folders = []

        try:
            app.logger.debug("folders:\\n%s", json.dumps(folders, indent=2))
        except Exception:
            app.logger.debug("folders (repr): %r", folders)

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

                # return the redirect to the uploaded file
                return redirect(url_for('uploaded_file', filename=f'{filename_no_ext}/raw{file_ext}'))

        # GET request
        return render_template('uploads.html')


    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    return app