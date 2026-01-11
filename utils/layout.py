import streamlit as st

def add_title(title: str, subtitle_mk: str = None):
  st.title(title)
  if subtitle_mk:
    st.markdown(subtitle_mk)

def card_container():
    return st.markdown(
        """
        <div style="
            background-color: #F9FAFB;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            height: 100%;
        ">
        """,
        unsafe_allow_html=True,
    )

def close_card():
    st.markdown("</div>", unsafe_allow_html=True)
