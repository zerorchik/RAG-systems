import os
import uuid
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# завантаження бібліотек
load_dotenv()

COLLECTION_NAME = 'demo_rag'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'       # маленька, швидка, 384-вимірна модель
GROQ_MODEL_NAME = 'llama-3.3-70b-versatile'     # оптимальна для невеликих проєктів


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Розбиває текст на chunks фіксованого розміру з overlap.

        chunk_size — розмір шматка в символах (для навчання простіше рахувати
        символи, ніж токени).
        overlap — скільки символів у кінці попереднього chunk повторюється
        на початку наступного. Це потрібно, щоб речення на межі двох chunks
        не втрачало контекст — інакше важлива інформація може "розрізатись"
        навпіл і жоден chunk не буде релевантним для пошуку.
        """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_vector_store(
        client: QdrantClient,
        embedder: models.Embedder,
        chunks: list[str],
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

    vectors = embedder.encode(chunks).tolist()

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk},                    # payload — це метадані, які повертаються разом із результатом пошуку
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


def retrieve_relevant_chunks(
        client: QdrantClient,
        embadder: SentenceTransformer,
        query: str,
        top_k: int = 3,
) -> list[str]:
    """Знаходить top_k найбільш релевантних chunks для запиту."""
    query_vector = embadder.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [point.payload["text"] for point in results]


def generate_answear(groq_client: Groq, query: str, context_chunks: list[str]) -> str:
    """Генерує відповідь на основі retrieved context (grounding)."""
    context = '\n\n'.join(context_chunks)

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
        temperature=0.2         # низька temperature — менше "творчості", більше точності
    )

    return response.choices[0].message.content


def main() -> None:
    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    qroq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 1. Ingestion: простий приклад тексту замість файлу, щоб швидко перевірити pipeline
    sample_text = """
        Мою маму звуть Олеся. Їй 38 років.
        Вона вчилась на архітектора, але стала дизайнером.
        
        Мого тата звуть Василь, йому 36 років.
        Він столяр.
        
        Мою сестру звуть Тамара, їй 14 років.
        Вона мріє побачити єдинорога.
        """

    chunks = chunk_text(sample_text)
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
