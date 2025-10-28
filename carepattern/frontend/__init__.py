import os
from flask import Flask, render_template
from .config_loader import load_ini_config

def create_app(config=None):
    app = Flask(__name__)

    if config is None:
        load_ini_config(app, "config.ini")
    else:
        app.config.from_mapping(config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app = create_routes(app)

    return app

def create_routes(app):
    @app.route('/')
    def render_root():
        return render_template('root.html')

    @app.route('/config')
    def render_config():
        return render_template('config.html')

    return app