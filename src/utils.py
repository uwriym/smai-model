import yaml
import os
import json
from datetime import datetime

def load_config(config_path="configs/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_data(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        today = datetime.today()
        # metadata의 startDate가 오늘 이후인 일정만 필터링 (날짜 형식: YYYY-MM-DD)
        filtered_data = [
            item for item in data
            if datetime.strptime(item["metadata"].get("startDate", "1900-01-01"), "%Y-%m-%d") >= today
        ]
        return filtered_data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at {json_path}")