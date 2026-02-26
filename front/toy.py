import streamlit as st

st.title("ログイン画面")
st.header("this is a header")

if st.button("hello"):
    st.write("holy shit")
else:
    st.write("push button")

if st.button("move main page"):
    st.switch_page("main_screen.py")