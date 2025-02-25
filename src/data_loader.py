# data_loader.py
import json
from datetime import datetime


def load_academic_data(json_path="data/sm_calendar_qa_dataset.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 날짜 문자열을 datetime 객체로 변환하여 전처리
    for record in data:
        metadata = record.get("metadata", {})
        if "startDate" in metadata:
            metadata["startDate"] = datetime.strptime(metadata["startDate"], "%Y-%m-%d")
        if "endDate" in metadata:
            metadata["endDate"] = datetime.strptime(metadata["endDate"], "%Y-%m-%d")

    return data


# 전처리된 데이터를 전역 변수에 저장할 수도 있음 (프로젝트 전반에서 재사용)
ACADEMIC_DATA = load_academic_data()
