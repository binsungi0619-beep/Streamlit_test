"""
Corporación Favorita Dashboard
메인 진입점 - General 페이지로 자동 이동
"""

import streamlit as st

st.set_page_config(
    page_title="Corporación Favorita | Dashboard",
    page_icon="🇪🇨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 랜딩 페이지
st.markdown("""
<div style='text-align: center; padding: 100px 0;'>
    <h1 style='color: #E31837; font-size: 3rem; margin-bottom: 20px;'>
        Corporación Favorita
    </h1>
</div>
""", unsafe_allow_html=True)

# 버튼 중앙 정렬 (세로 배치)
col1, col2, col3 = st.columns([0.9, 1.3, 1])

with col2:
    if st.button("Sales Analytics Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/1_General.py")
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.link_button("¡Ho11a!", 
                   "https://www.notion.so/teamsparta/Ho11a-2b42dc3ef514805e8078ff1ea653b50c?source=copy_link", 
                   use_container_width=True)