import streamlit as st
import pandas as pd
import numpy as np

books = pd.read_csv('./data/books.csv')

if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'page' not in st.session_state:
    st.session_state.page = 1

st.title('Показать 10 случайных книг из базы данных')

col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input(
        'Пользовательский запрос', label_visibility='collapsed'
    )
with col2:
    if st.button('Найти'):
        st.session_state.search_triggered = True
        st.session_state.page = 1
    
items_per_page = st.number_input(
    'Показывать на странице', min_value=1, max_value=10, value=3, step=1
)

if st.session_state.search_triggered:
    total_books = books.shape[0]
    num_pages = (total_books + items_per_page - 1) // items_per_page
    
    new_page = st.number_input(
        'Страница', min_value=1, max_value=num_pages, value=1, step=1
    )
    st.session_state.page = new_page
    
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
                st.markdown(f'**{row['annotation']}**')
            st.markdown('---')