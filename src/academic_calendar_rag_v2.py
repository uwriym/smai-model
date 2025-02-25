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
    def __init__(self, version="2", json_path=None, ollama_model=None):
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
            당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙을 엄격하게 준수하며 답변해 주세요.

            [규칙]
            1. **관련 정보 활용:** 질문과 직접 관련된 학사일정 정보({relevant_context})를 기반으로 답변합니다.
            2. **일정 선택 기준:** 과거, 현재, 미래 일정 모두 포함되어 있으며, json 데이터에 명시된 정보를 정확하게 사용해 답변합니다.
            3. **과거 일정에 관한 정확한 답변:** 질문에 과거 일정이 포함되어 있을 경우, 해당 정보를 무시하지 말고, json 데이터에 있는 과거 일정 정보를 바탕으로 구체적이고 정확하게 답변해 주세요.
            4. **질문 해석 및 재확인:** 질문이 모호하거나 여러 해석이 가능한 경우, 우선 본인이 이해한 내용을 간단히 요약하고, 필요한 경우 추가로 확인할 질문을 포함한 후 답변을 작성해 주세요.

            현재 날짜: {today.strftime('%Y년 %m월 %d일')}

            질문:
            {q}

            위 정보를 토대로, 날짜를 정확하게 명시하고 관련 학사일정 정보를 반영한 답변을 작성해 주세요. json 데이터에 있는 과거 일정도 포함되어 있으므로, 이를 바탕으로 최대한 정확한 답변을 제공해 주십시오.
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
