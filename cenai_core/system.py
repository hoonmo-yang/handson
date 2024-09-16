import json
import os
from pathlib import Path


def cenai_path(*args) -> Path:
    return Path(os.environ["HANDSON_DIR"]).joinpath(*args)


def get_value(key: str) -> str:
    value_json = cenai_path("cf/values.json")
    with value_json.open("rt") as fin:
        value = json.load(fin)

    return value[key]
