import streamlit as st
import re

from qdrant_client import QdrantClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchAny

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




st.title('📚 Семантический Поиск Книг')

col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input(
        'Пользовательский запрос', label_visibility='collapsed'
    )
with col2:
    if st.button('Найти'):
        st.session_state.search_triggered = True
        st.session_state.page = 1
    
genre = st.selectbox("📚 Фильтр по жанру", ['Все жанры'] + GENRES)

if genre != st.session_state.prev_genre:
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
            st.markdown('---')
            