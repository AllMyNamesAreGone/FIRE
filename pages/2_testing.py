import streamlit as st

tab1, tab2, tab3 = st.tabs(["tab1", "tab2", "tab3"])
with tab1:
    "This is tab1"
with tab2:
    "This is tab2"
    with st.sidebar:
        st.write("sidebar")
with tab3:
    "This is tab3"
