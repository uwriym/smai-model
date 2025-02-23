import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from src.utils import *


class AcademicCalendarRAG:
    def __init__(self, version="0", json_path=None, ollama_model=None):
        self.version = version
        config = load_config()
        if json_path is None:
            json_path = config.get("json_path", "data/sm_calendar_qa_dataset.json")
        if ollama_model is None:
            ollama_model = config.get("ollama_model", "exaone3.5:7.8b")
        self.json_path = json_path
        self.ollama_model = ollama_model
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

            prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙에 따라 답변해주세요.
1. 현재 연도 기준 이전의 일정은 제외할 것.
2. 학사일정 중 질문과 관련된 정보를 기반으로 답변할 것.
3. 현재 날짜에서 가장 가까운 일정을 출력할 것.
현재 날짜: {today.strftime('%Y년 %m월 %d일')}
아래는 질문과 관련된 학사일정 정보입니다:
{relevant_context}
질문: {q}
위 정보를 참고하여 답변해주세요. 날짜 정보는 정확하게 포함시켜주시고, 관련 정보를 찾지 못한 경우 유사한 정보를 바탕으로 대답해주세요."""

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
    question = input("질문을 입력하세요: ")
    answer = rag.get_answer(question)
    print("답변:", answer)
