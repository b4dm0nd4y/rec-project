import streamlit as st
import pandas as pd
import numpy as np

GENRES = [
    'Фэнтези', 'Роман', 'Фантастика', 'Молодежная проза',
    'Попаданцы', 'Эротика', 'Фанфик', 'Детективы',
    'Проза', 'Триллеры', 'Мистика/Ужасы', 'Разное',
    'Нон-фикшн', 'Мини'
]

GENRES_DICT = {
    'Все жанры': 'genres.csv',
    'Фэнтези': 'phantasy.csv',
    'Роман': 'novels.csv',
    'Фантастика': 'phantastic.csv',
    'Молодежная проза': 'prose.csv',
    'Попаданцы': 'popadancy.csv',
    'Эротика': 'erotika.csv',
    'Фанфик': 'fanfiki.csv',
    'Детективы': 'detective.csv',
    'Проза': 'proza.csv',
    'Триллеры': 'triller.csv',
    'Мистика/Ужасы': 'horror.csv',
    'Разное': 'raznoye.csv',
    'Нон-фикшн': 'non-fiction.csv',
    'Мини': 'mini.csv'
}

BASE_PATH = './data/genres/'

if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'page' not in st.session_state:
    st.session_state.page = 1

@st.cache_data
def load_books(genre):
    return pd.read_csv(BASE_PATH+GENRES_DICT[genre])



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
    
    
items_per_page = st.number_input(
    'Показывать на странице', min_value=1, max_value=10, value=3, step=1
)

if st.session_state.search_triggered:
    books = load_books(genre)
    total_books = books.shape[0]
    num_pages = (total_books + items_per_page - 1) // items_per_page
    
    
    
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    st.markdown(f"🔎 **Найдено результатов:** {total_books}")
    with nav_col1:
        if st.button("⬅️ Назад", key='prev_top') and st.session_state.page > 1:
            st.session_state.page -=1
    with nav_col2:
        st.markdown(f'**Страница {st.session_state.page} из {num_pages}**')
    with nav_col3:
        if st.button("Вперёд ➡️", key='next_top') and st.session_state.page < num_pages:
            st.session_state.page +=1
    
    
    
    start_idx = (st.session_state.page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    visible_books = books.iloc[start_idx:end_idx]
    
    for index, row in visible_books.iterrows():
        with st.container():
            book_col1, book_col2 = st.columns([1,4])
            with book_col1:
                st.image(row['img_url'])
            with book_col2:
                st.markdown(f'**{row['author']}**')
                st.markdown(f'[**{row['title']}**]({row['book_url']})')
                st.markdown(f'{row['annotation']}')
                st.markdown(f'📖 _Жанр: {row['genre']}_')
            st.markdown('---')
            
            
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    st.markdown(f"🔎 **Найдено результатов:** {total_books}")
    with nav_col1:
        if st.button("⬅️ Назад", key='prev_bottom') and st.session_state.page > 1:
            st.session_state.page -=1
    with nav_col2:
        st.markdown(f'**Страница {st.session_state.page} из {num_pages}**')
    with nav_col3:
        if st.button("Вперёд ➡️", key='next_bottom') and st.session_state.page < num_pages:
            st.session_state.page +=1