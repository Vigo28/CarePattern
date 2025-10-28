from carepattern.frontend import create_app

if __name__ == "__main__":
    app = create_app()
    debug = bool(app.config.get("DEBUG", False))
    port = int(app.config.get("PORT", 5000))
    app.run(debug=debug, port=port)
