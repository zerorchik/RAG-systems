import re
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq
from qdrant_client import QdrantClient, models
from qdrant_client.grpc import Qdrant
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# завантаження бібліотек
load_dotenv()

COLLECTION_NAME = 'demo_rag'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'       # маленька, швидка, 384-вимірна модель
GROQ_MODEL_NAME = 'llama-3.3-70b-versatile'     # оптимальна для невеликих проєктів
LAB_SECTION_HEADERS = ["Мета", "Хід роботи", "Результати", "Висновки"]


# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
#     """Розбиває текст на chunks фіксованого розміру з overlap.
#
#         chunk_size — розмір шматка в символах (для навчання простіше рахувати
#         символи, ніж токени).
#         overlap — скільки символів у кінці попереднього chunk повторюється
#         на початку наступного. Це потрібно, щоб речення на межі двох chunks
#         не втрачало контекст — інакше важлива інформація може "розрізатись"
#         навпіл і жоден chunk не буде релевантним для пошуку.
#         """
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     return chunks


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Розбиває текст на chunks фіксованого розміру з overlap.

        chunk_size — розмір шматка в символах (для навчання простіше рахувати
        символи, ніж токени).
        overlap — скільки символів у кінці попереднього chunk повторюється
        на початку наступного. Це потрібно, щоб речення на межі двох chunks
        не втрачало контекст — інакше важлива інформація може "розрізатись"
        навпіл і жоден chunk не буде релевантним для пошуку.
        """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    return splitter.split_text(text)


def split_by_sections(text: str, section_headers: list[str]) -> list[dict]:
    """Розбиває текст на розділи за списком відомих заголовків.

    Повертає список {"section": назва_розділу, "text": вміст_розділу}.
    Це дозволяє зберегти назву розділу в metadata кожного chunk —
    для retrieval і для показу користувачу, з якої саме секції
    взята відповідь.
    """
    # Будуємо regex, який знаходить будь-який із заголовків на початку рядка
    pattern = '|'.join(re.escape(header) for header in section_headers)
    splits = re.split(f"{pattern}", text)

    sections = []
    current_header = 'Вступ'        # текст до першого заголовка, якщо є
    current_text = ''

    for part in splits:
        if part.strip() in section_headers:
            if current_text.strip():
                sections.append({"section": current_header, "text": current_text.strip()})
            current_header = part.strip()
            current_text = ''
        else:
            current_text += part

    if current_text.strip():
        sections.append({"section": current_header, "text": current_text.strip()})

    return sections


def load_pdf_text(file_path: Path) -> str:
    """Витягує весь текст із PDF-файлу, сторінка за сторінкою."""
    reader = PdfReader(file_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def load_documents(folder: Path) -> dict[str, str]:
    """Завантажує всі PDF з папки. Повертає {ім'я_файлу: повний_текст}."""
    documents = {}
    for pdf_path in folder.glob("*.pdf"):
        documents[pdf_path.stem] = load_pdf_text(pdf_path)
    return documents


def chunk_documents(documents: dict[str, str]) -> list[dict]:
    """Розбиває кожен документ на chunks, зберігаючи джерело в metadata.

    Повертає список словників {"text": ..., "source": ...} — це і є
    "chunk з metadata", готовий для запису в Qdrant payload.
    """
    all_chunks = []
    for source_name, text in documents.items():
        for chunk in chunk_text(text):
            all_chunks.append({"text": chunk, "source": source_name})
    return all_chunks


# def chunk_documents(documents: dict[str, str]) -> list[dict]:
#     """Розбиває кожен документ на chunks по розділах, зберігаючи
#     source (файл) і section (розділ) у metadata кожного chunk.
#     """
#     all_chunks = []
#     for source_name, text in documents.items():
#         sections = split_by_sections(text, LAB_SECTION_HEADERS)
#         for section in sections:
#             # Якщо розділ завеликий — додатково ріжемо на менші chunks,
#             # але зберігаємо ту саму назву секції для кожного шматка
#             for piece in chunk_text(section['text'], chunk_size=800, overlap=100):
#                 all_chunks.append({
#                     'text': piece,
#                     'source': source_name,
#                     'section': section['section'],
#                 })
#     return all_chunks


def build_vector_store(
        client: QdrantClient,
        embedder: SentenceTransformer,
        chunks: list[dict],         # тепер список словників, а не рядків
) -> None:
    """Створює колекцію в Qdrant і заливає туди embeddings усіх chunks."""
    vector_size = embedder.get_embedding_dimension()

    # Якщо уже існує колекція, видаляємо стару і створює нову.
    # Зручно для навчання/демо, щоб не накопичувати дублікати при повторних запусках.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts).tolist()

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=chunk,          # {"text": ..., "source": ...} — уже готовий словник
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


def retrieve_relevant_chunks(
        client: QdrantClient,
        embedder: SentenceTransformer,
        query: str,
        top_k: int = 15,
) -> list[dict]:
    """Знаходить top_k найбільш релевантних chunks для запиту."""
    query_vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return [{"text": p.payload["text"], "source": p.payload["source"]} for p in results]


def generate_answear(groq_client: Groq, query: str, context_chunks: list[str]) -> str:
    """Генерує відповідь на основі retrieved context (grounding)."""
    context = '\n\n'.join(
        f'[Джерело: {chunk['source']}]\n{chunk['text']}'
        for chunk in context_chunks
    )

    # Ключовий момент промпту: явно кажемо моделі відповідати ТІЛЬКИ
    # на основі контексту — це базовий anti-hallucination прийом.
    prompt = f"""Дай відповідь на питання, використовуючи ЛИШЕ наданий контекст.
    Якщо в контексті немає відповіді, чесно скажи, що не знаєш.

    Контекст:
    {context}

    Питання: {query}
    Відповідь:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # низька temperature — менше "творчості", більше точності
    )

    return response.choices[0].message.content


# def generate_answear(groq_client: Groq, query: str, context_chunks: list[str]) -> str:
#     """Генерує відповідь на основі retrieved context (grounding)."""
#     context = '\n\n'.join(
#         f'[Джерело: {chunk['source']}, розділ: {chunk['section']}]\n{chunk['text']}'
#         for chunk in context_chunks
#     )
#
#     # Ключовий момент промпту: явно кажемо моделі відповідати ТІЛЬКИ
#     # на основі контексту — це базовий anti-hallucination прийом.
#     prompt = f"""Дай відповідь на питання, використовуючи ЛИШЕ наданий контекст.
#     Якщо в контексті немає відповіді, чесно скажи, що не знаєш.
#
#     Контекст:
#     {context}
#
#     Питання: {query}
#     Відповідь:"""
#
#     response = groq_client.chat.completions.create(
#         model=GROQ_MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2  # низька temperature — менше "творчості", більше точності
#     )
#
#     return response.choices[0].message.content


def main() -> None:
    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    qroq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 1. Ingestion: приклад декількох pdf файлів
    data = load_documents(Path('documents'))

    chunks = chunk_documents(data)

    print(f'Створено {len(chunks)} chunks')

    build_vector_store(qdrant_client, embedder, chunks)
    print(f'Vector store заповнено')

    query = input('\nЩо хочеш запитати? ')
    relevant_chunks = retrieve_relevant_chunks(qdrant_client, embedder, query)
    print(f'Знайдено {len(relevant_chunks)} релевантних chunks')

    answear = generate_answear(qroq_client, query, relevant_chunks)
    print(f'\nВідповідь: {answear}')


if __name__ == "__main__":
    main()
