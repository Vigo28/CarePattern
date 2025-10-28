from configparser import ConfigParser
from typing import Any

def _parse_value(v: str) -> Any:
    v = v.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    # try int/float
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v

def load_ini_config(app, path="config.ini"):
    parser = ConfigParser()
    read_files = parser.read(path)
    if not read_files:
        return  # no file found

    # Prefer a dedicated section named 'flask', else DEFAULT, else first section
    if parser.has_section("flask"):
        items = parser.items("flask")
    elif parser.defaults():
        items = parser.defaults().items()
    elif parser.sections():
        items = parser.items(parser.sections()[0])
    else:
        return

    cfg = {key.upper(): _parse_value(value) for key, value in items}
    app.config.update(cfg)
