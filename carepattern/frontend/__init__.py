import os
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
            files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
            files.sort()
        except OSError:
            files = []
        return render_template('root.html', files=files)


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
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                return redirect(url_for('uploaded_file', filename=filename))

        # GET: list uploaded files and render template with the upload form included
        uploads_dir = app.config['UPLOAD_FOLDER']
        try:
            files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
            files.sort()
        except OSError:
            files = []
        return render_template('uploads.html', files=files)


    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    return app