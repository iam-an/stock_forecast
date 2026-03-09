import streamlit as st
from front.pages.login_screen import login
from front.pages.main_screen import main_screen

# 初回起動時だけ初期化
if "page" not in st.session_state:
    st.session_state.page = "login"

# ページルーター
page = st.session_state.page

if page == "login":
    login()

elif page == "main_screen":
    main_screen()