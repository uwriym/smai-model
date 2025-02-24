import time
import importlib


def load_version_module(version):
    module_name = f"src.academic_calendar_rag_v{version}"
    try:
        module = importlib.import_module(module_name)
        return module.AcademicCalendarRAG
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module for version {version} not found. Expected {module_name}.py exists?")


def main():
    try:
        version = input("Enter version: ")
        AcademicCalendarRAG = load_version_module(version)
        rag_system = AcademicCalendarRAG(version=version)
        print(f"상명대학교 학사일정 RAG Loaded(version: {version})")
        print("Enter 'quit' or 'exit' to terminate")

        while True:
            question = input("\nQ: ")
            if question.lower() in ['quit', 'exit']:
                break

            start_time = time.time()
            answer = rag_system.get_answer(question)
            print("\nA:", answer)
            elapsed_time = time.time() - start_time
            print(f"\n[Elapsed Time] {elapsed_time:.2f}s")

    except Exception as e:
        print("System init failed:", e)


if __name__ == "__main__":
    main()
