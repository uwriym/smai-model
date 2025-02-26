import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from src.utils import *


def load_data(json_path):
    """JSON 데이터를 로드하고, 날짜 문자열을 datetime 객체로 변환하는 전처리 함수."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        metadata = record.get("metadata", {})
        if "startDate" in metadata:
            try:
                metadata["startDate"] = datetime.strptime(metadata["startDate"], "%Y-%m-%d")
            except Exception as e:
                print(f"날짜 변환 오류 (startDate): {e}")
        if "endDate" in metadata:
            try:
                metadata["endDate"] = datetime.strptime(metadata["endDate"], "%Y-%m-%d")
            except Exception as e:
                print(f"날짜 변환 오류 (endDate): {e}")
    return data


class AcademicCalendarRAG:
    def __init__(self, version="3", json_path=None, ollama_model=None):
        self.version = version
        config = load_config()  # config 로딩 함수 (프로젝트에 맞게 구현되어 있어야 함)
        if json_path is None:
            json_path = config.get("json_path", "data/sm_calendar_qa_dataset.json")
        if ollama_model is None:
            ollama_model = config.get("ollama_model", "exaone3.5:7.8b")
        self.json_path = json_path
        self.ollama_model = ollama_model

        # 데이터를 한 번 로드할 때 전처리(날짜 변환) 수행
        self.data = load_data(json_path)

        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.document_vectors = None
        self._prepare_vectors()

    def _prepare_vectors(self):
        contexts = [item["context"] for item in self.data]
        self.document_vectors = self.vectorizer.fit_transform(contexts)

    def _get_relevant_documents(self, query, top_k=3):
        if self.document_vectors.shape[0] == 0:
            return "관련 정보를 찾을 수 없습니다."

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_docs = [self.data[idx]["context"] for idx in top_indices if similarities[idx] > 0]
        return "\n".join(relevant_docs) if relevant_docs else "관련 정보를 찾을 수 없습니다."

    def get_answer(self, q):
        try:
            today = datetime.today()
            relevant_context = self._get_relevant_documents(q)

            prompt = f"""
            당신은 상명대학교 학생들을 위한 챗봇입니다. 답변 생성 전에, 내부적으로 아래 단계를 차례로 진행하십시오.
            1. 질문에서 핵심 키워드를 파악합니다.
            2. 제공된 학사 일정 정보({relevant_context})와 json 데이터 내의 일정 정보를 검토합니다.
            3. 각 일정의 날짜, 기간, 중요도 등을 비교 분석하여, 질문의 의도에 가장 부합하는 정보를 도출합니다.
            4. 도출된 정보를 바탕으로 최종 답변을 작성합니다.
            (내부 추론 과정은 최종 답변에 포함시키지 마십시오.)

            현재 날짜: {today.strftime('%Y년 %m월 %d일')}

            질문:
            {q}

            위 정보를 토대로, 날짜를 정확하게 명시하고 관련 학사 일정 정보를 반영한 체계적인 답변을 작성해 주세요. json 데이터에 포함된 과거 일정도 반드시 활용하여 최대한 정확한 답변을 제공해 주십시오.
            """

            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return cleaned_content.strip()

        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    rag = AcademicCalendarRAG()
    question = input("\nQ: ")
    answer = rag.get_answer(question)
    print("\nA: ", answer)
