"""
공통 CSS 스타일 (통합 버전)
"""
import streamlit as st


def apply_common_styles():
    """공통 CSS 스타일 적용"""
    st.markdown("""
    <style>
        /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* 전체 폰트 */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* 메인 배경 */
        .stApp {
            background: #F5F5F5;
        }
        
        /* 헤더 스타일 */
        h1 {
            color: #2D2D2D !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        
        h2, h3 {
            color: #2D2D2D !important;
            font-weight: 600 !important;
        }
        
        /* 메트릭 카드 스타일 */
        [data-testid="stMetric"] {
            background: white;
            padding: 20px 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid #E31837;
        }
        
        [data-testid="stMetric"] label {
            color: #666666 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #2D2D2D !important;
            font-size: 32px !important;
            font-weight: 700 !important;
        }
        
        /* 사이드바 스타일 */
        [data-testid="stSidebar"] {
            background: #2D2D2D !important;
        }
                
        /* 사이드바 라디오 버튼 (Cluster) 스타일 */
        [data-testid="stSidebar"] .stRadio > label {
            font-size: 20px !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stRadio > div > label {
            font-size: 16px !important;
        }

        /* 사이드바 "Cluster" 텍스트 크기 */
        [data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 30px !important;
            font-weight: 700 !important;
        }

        /* 사이드바 네비게이션 메뉴 글자 크기 */
        [data-testid="stSidebarNav"] a {
            font-size: 18px !important;
            padding: 12px 16px !important;
        }

        [data-testid="stSidebarNav"] span {
            font-size: 18px !important;
            font-weight: 500 !important;
        }
                
        [data-testid="stSidebar"] > div {
            background: #2D2D2D !important;
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        [data-testid="stSidebar"] code {
            color: #E31837 !important;
            background: white !important;
        }
        
        [data-testid="stSidebar"] {
            width: 280px !important;
            min-width: 280px !important;
            max-width: 280px !important;
        }

        [data-testid="stSidebar"] > div {
            width: 280px !important;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: white;
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #666666;
            padding: 12px 24px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: #E31837 !important;
            color: white !important;
        }
        
        /* Plotly 차트 컨테이너 */
        .stPlotlyChart {
            background: white;
            border-radius: 12px;
            padding: 24px 24px 32px 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        .stPlotlyChart,
        .stPlotlyChart > div,
        .stPlotlyChart > div > div,
        .stPlotlyChart > div > div > div {
            overflow: hidden !important;
        }

        .stPlotlyChart::-webkit-scrollbar,
        .stPlotlyChart > div::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        
        /* 구분선 */
        hr {
            border-color: #E0E0E0;
            margin: 2rem 0;
        }
        
        /* 데이터프레임 스타일 */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* 알림 박스 */
        .stAlert {
            border-radius: 12px;
        }
        
        /* 버튼 스타일 */
        .stButton > button {
            background: #E31837;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background: #B71430;
            color: white;
        }
        
        /* 클러스터 헤더 */
        .cluster-header {
            border-left: 6px solid #E31837;
            padding: 16px 24px;
            background: white;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: 600;
            color: #2D2D2D;
            margin: 24px 0 16px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        /* 시나리오 박스 */
        .scenario-box {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid #E31837;
            margin: 16px 0;
        }
        
        /* 스크롤바 제거 */
        ::-webkit-scrollbar {
            width: 0px !important;
            height: 0px !important;
            display: none !important;
        }
        
        * {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        
        div[data-testid="stVerticalBlock"],
        .element-container {
            overflow: visible !important;
        }
    </style>
    """, unsafe_allow_html=True)


def get_chart_layout(title=""):
    """Plotly 차트 기본 레이아웃"""
    return dict(
        title=dict(text=title, font=dict(size=16, color='#2D2D2D', family='Inter')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#666666', family='Inter'),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor='#E0E0E0', linecolor='#E0E0E0'),
        yaxis=dict(gridcolor='#E0E0E0', linecolor='#E0E0E0'),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E0E0E0',
            borderwidth=1
        )
    )


def add_sidebar_logo():
    """사이드바 로고 (상단)"""
    st.sidebar.image("assets/logo.png", use_container_width=True)


def add_sidebar_logo_bottom():
    """사이드바 로고 바닥 고정"""
    st.sidebar.markdown("""
    <style>
        [data-testid="stSidebar"] [data-testid="stImage"] {
            position: fixed !important;
            bottom: 20px !important;
            left: 20px !important;
            width: 200px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.image("assets/logo.png", width=200)


def add_page_footer():
    """페이지 하단 푸터"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999999; padding: 30px 0;'>
        <p style='margin: 0; font-size: 13px;'>
            <span style='color: #E31837; font-weight: 600;'>Corporación Favorita</span> | Sales Analytics Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)


def add_region_header(region_name: str, cluster_label: str):
    """지역 페이지 헤더"""
    st.markdown(f"""
    <div style='margin-bottom: 8px;'>
        <span style='color: #E31837; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>
            Regional Dashboard
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"# {region_name} - {cluster_label}")
