import streamlit as st
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain.schema.output_parser import StrOutputParser

from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc


if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'page' not in st.session_state:
    st.session_state.page = 1
if "prev_genre" not in st.session_state:
    st.session_state.prev_genre = "Все жанры"
    


st.session_state.already_ran = True

with open('./data/stopwords/russian', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    
stop = set(lines) - {'не', 'ни'}

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


GENRES = [
    'Фэнтези', 'Роман', 'Фантастика', 'Молодежная проза',
    'Попаданцы', 'Эротика', 'Фанфик', 'Детективы',
    'Проза', 'Триллеры', 'Мистика/Ужасы', 'Разное',
    'Нон-фикшн', 'Мини'
]

COLLECTION_NAME = 'books-rec-project3'

model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

API_TOKEN = ''

llm = ChatMistralAI(
    model = 'mistral-small',
    temperature = .7,
    max_tokens=2000,
    api_key=API_TOKEN
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты эксперт-аналитик книгам с многолетним опытом и отличным чувством юмора! 🎯
    Твоя задача - проанализировать предоставленные данные по книге и предугадать сюжет.

    Стиль анализа:
    - Структурируй ответ с эмодзи и забавными комментариями
    - Отвечай на только русском языке, это важно
    - Generate your response in russian language

    Помни: юмор должен быть добрым и не оскорбительным. Цель - сделать анализ интересным!

    Если среди аннотаций есть что-то особенно забавное - обязательно это отметь! 😄"""),

    ("human", """📊 ДАННЫЕ ДЛЯ ЭКСПЕРТНОГО МНЕНИЯ:
{context}

🎯 ЗАПРОС НА ЭКСПЕРТИЗУ: {question}""")
])


def clean_string(text):
    text = text.lower()
    text = re.sub(r'\s{1}[а-яё]*\.\.\.$', '', text) #remove bad ending
    text = re.sub(r'<.*?>', ' ', text) # removing html-tags from text
    text = re.sub(r'http\S+|\S+@\S+', ' ', text) # removing links
    text = re.sub(r'[^а-яё\s]', ' ', text.lower()) # remove all non-letter symbols
    cleaned_text = re.sub(r'\s+', ' ', text).strip() # remove double or more spaces
    
    return cleaned_text

def lemmantize_words(text):
    lemmas = []
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    try:
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemmas.append(token.lemma)
                
    except Exception as e:
        print(f"Ошибка при обработке текста: {text[:30]}... → {e}")
        return ''
    
    return ' '.join(lemmas)

def filter_stop_words(text):
    words = []
    for word in text.split():
        if word in stop or len(word) < 2:
            continue
        words.append(word)
        
    return ' '.join(words)

def preprocess_string(string):
    if not isinstance(string, str) or len(string.strip()) == 0: # fool-protection
        return ''
    
    string = clean_string(string)
    string = lemmantize_words(string)
    string = filter_stop_words(string)
    
    return string


def render_navigation(num_pages, location):
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Назад", key=f"prev_{location}"):
            if st.session_state.page > 1:
                st.session_state.page -=1
    with col2:
        st.markdown(f"**Страница {st.session_state.page} из {num_pages}**")
    with col3:
        if st.button("Вперёд ➡️", key=f"next_{location}"):
            if st.session_state.page < num_pages:
                st.session_state.page += 1

def search_books(query, genre, top_k=5):
    client = QdrantClient(path='./db/qdrant_db')

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=hf
    )
    
    my_filter = Filter()
    
    if genre in GENRES:
        my_filter = Filter(
            must = [
                FieldCondition(
                    key='metadata.main_genre',
                    match=MatchAny(any=[genre])
                )
            ]
        )
    
    results = vector_store.similarity_search(
        preprocess_string(query),
        filter=my_filter,
        k=top_k
    )
    
    client.close()

    return results

def format_book(book):
    formatted_book = f"""
    Авторы: {book.get('author', 'Не указано')}
    Название: {book.get('title', 'Не указано')}
    Картинка: {book.get('img_url', 'Не указано')}
    Сылка: {book.get('book_url', 'Не указано')}

    Краткое Описание: {book.get('annotation', 'Не указано')}...
    """

    return formatted_book

def generate_text(doc):
    query = 'Что ты думаешь про эту книгу, ее название и ее автора?'
    formatted_context = format_book(doc.metadata)
    input_data = {
        "context": formatted_context,
        "question": query
    }
    formatted_prompt = rag_prompt.format(**input_data)
    
    # Получаем ответ
    response = llm.invoke(formatted_prompt)
    final_answer = StrOutputParser().parse(response)

    return final_answer.content

def reset_generated_texts():
    for k in list(st.session_state.keys()):
        if k.startswith("generated_"):
            del st.session_state[k]


st.title('📚 Семантический Поиск Книг')

col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input(
        'Пользовательский запрос', label_visibility='collapsed'
    )
with col2:
    if st.button('Найти'):
        reset_generated_texts() 
        st.session_state.search_triggered = True
        st.session_state.page = 1
    
genre = st.selectbox("📚 Фильтр по жанру", ['Все жанры'] + GENRES)

if genre != st.session_state.prev_genre:
    reset_generated_texts() 
    st.session_state.page = 1
    st.session_state.prev_genre = genre
    
    
items_per_page = st.slider(
    'Показывать книг', min_value=1, max_value=10, value=5, step=1
)

if st.session_state.search_triggered:
    if query != '':
        results = search_books(query, genre, items_per_page)
    else:
        results = []
    
    st.markdown(f"🔎 **Найдено результатов:** {len(results)}")
    
    for idx, doc in enumerate(results):
        with st.container():
            book_col1, book_col2 = st.columns([1,4])
            with book_col1:
                st.image(doc.metadata.get('img_url', 'https://img.freepik.com/premium-vector/broken-image-icon_268104-8936.jpg'))
            with book_col2:
                st.markdown(f'**{doc.metadata.get('author', 'No author')}**')
                st.markdown(
                    f'[**{doc.metadata.get('title','No title')}**]({doc.metadata.get('book_url','No book_url')})'
                )
                st.markdown(f'{doc.metadata.get('annotation','No annotation')}')
                st.markdown(f'📖 _Жанр: {doc.metadata.get('main_genre','No main_genre')}_')
                
                btn_key = f"generate_btn_{idx}"
                if st.button("🪄 Экспертное мнение", key=btn_key):
                    st.session_state[f"generated_{idx}"] = generate_text(doc)

                # === Show generated text if exists ===
                generated_key = f"generated_{idx}"
                if generated_key in st.session_state:
                    st.markdown(f'**Ответ Эксперта:**\n\n{st.session_state[generated_key]}')
            st.markdown('---')
            