import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from src.utils import *


class AcademicCalendarRAG:
    def __init__(self, version="1", json_path=None, ollama_model=None):
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

            prompt = f"""
            당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙을 엄격하게 준수하며 답변해 주세요.

            [규칙]
            1. **관련 정보 활용:** 질문과 직접 관련된 학사일정 정보({relevant_context})를 기반으로 답변합니다.
            2. **일정 선택 기준:** 과거, 현재, 미래 일정을 모두 고려하여 질문의 맥락에 가장 적절한 일정을 선택해 주세요.
            3. **질문 해석 및 재확인:** 질문이 모호하거나 여러 해석이 가능한 경우, 우선 본인이 이해한 내용을 간단히 요약해 제시하고, 필요한 경우 추가로 확인할 질문을 포함한 후 답변을 작성해 주세요.

            현재 날짜: {today.strftime('%Y년 %m월 %d일')}

            질문:
            {q}

            위 정보를 토대로, 날짜를 정확하게 명시하고 관련 일정 정보를 반영한 답변을 작성해 주세요. 만약 질문에 부합하는 구체적인 일정 정보가 부족한 경우, 유사하거나 관련 있는 정보를 바탕으로 최대한 정확하게 답변해 주십시오.
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
