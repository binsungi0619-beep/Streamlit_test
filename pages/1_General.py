"""
Corporación Favorita - GENERAL DASHBOARD
에콰도르 소매점 판매 데이터 대시보드 (Snowflake 집계 버전)
"""

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.snowflake_conn import (
    load_general_kpi_filtered,
    load_general_transactions_kpi,
    load_daily_sales,
    load_daily_transactions,
    load_region_sales,
    load_region_transactions,
    load_store_sales,
    load_family_sales,
    load_store_promo_ratio,
    load_family_promo_uplift,
    load_stores_count,
)
from utils.styles import apply_common_styles, get_chart_layout, add_sidebar_logo_bottom, add_page_footer
from utils.config import COLORS, CHART_COLORS, REGION_COLORS

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="General | Favorita Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_common_styles()

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    add_sidebar_logo_bottom()

# ============================================================
# 메인 타이틀
# ============================================================
st.markdown(""" 
<div style='margin-bottom: 8px;'>
    <span style='color: #E31837; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>
        데이터분석 9기 최종프로젝트
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("# GENERAL DASHBOARD")

# ============================================================
# 필터 - 연도 & 분기 & 월
# ============================================================
# 초기화 처리
if st.session_state.get('reset_general_filters', False):
    st.session_state.general_year = '전체'
    st.session_state.general_quarter = '전체'
    st.session_state.general_month = '전체'
    st.session_state.reset_general_filters = False

col_y, col_q, col_m, col_reset = st.columns([2, 2, 2, 0.5])

with col_y:
    year_options = ["전체", "2013", "2014", "2015", "2016"]
    selected_year = st.selectbox("연도", year_options, index=0, key='general_year')

with col_q:
    quarter_options = ["전체", "1분기", "2분기", "3분기", "4분기"]
    selected_quarter = st.selectbox("분기", quarter_options, index=0, key='general_quarter')

with col_m:
    q_map = {
        'Q1': [1, 2, 3], 'Q2': [4, 5, 6], 'Q3': [7, 8, 9], 'Q4': [10, 11, 12],
        '1분기': [1, 2, 3], '2분기': [4, 5, 6], '3분기': [7, 8, 9], '4분기': [10, 11, 12]
    }
    
    if selected_quarter == '전체':
        month_options = ['전체'] + [f'{i}월' for i in range(1, 13)]
    else:
        month_options = ['전체'] + [f'{i}월' for i in q_map.get(selected_quarter, range(1, 13))]
    
    # 분기 바뀌면 월 초기화
    if 'general_month' in st.session_state and st.session_state.general_month not in month_options:
        st.session_state.general_month = '전체'
    
    selected_month = st.selectbox("월", month_options, index=0, key='general_month')

with col_reset:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button('필터 초기화', key='reset_general'):
        st.session_state.reset_general_filters = True
        st.rerun()

# 필터 적용 후 표시
year_display = "전체 기간" if selected_year == "전체" else f"{selected_year}년"
quarter_display = "" if selected_quarter == "전체" else f" {selected_quarter}"
month_display = "" if selected_month == "전체" else f" {selected_month}"

st.markdown(f"**{year_display}{quarter_display}{month_display}** 데이터를 분석합니다.")
st.markdown("---")

# ============================================================
# 1. KPI 빅카드
# ============================================================
st.markdown("## 핵심 지표 (KPI)")

with st.spinner('KPI 데이터 로딩 중...'):
    kpi_data = load_general_kpi_filtered(selected_year, selected_quarter, selected_month)
    trans_kpi = load_general_transactions_kpi(selected_year, selected_quarter, selected_month)
    stores_count = load_stores_count()

col1, col2, col3, col4 = st.columns(4)

total_sales = kpi_data['TOTAL_SALES'][0] if kpi_data.height > 0 else 0
col1.metric(label="총 판매량", value=f"{total_sales:,.0f}")

total_transactions = trans_kpi['TOTAL_TRANSACTIONS'][0] if trans_kpi.height > 0 else 0
col2.metric(label="총 구매 건수", value=f"{total_transactions:,.0f}건")

total_stores = stores_count['CNT'][0] if stores_count.height > 0 else 0
col3.metric(label="총 매장 수", value=f"{total_stores}개")

total_sku = kpi_data['FAMILY_COUNT'][0] if kpi_data.height > 0 else 0
col4.metric(label="총 카테고리 수", value=f"{total_sku:,}개")

st.markdown("---")

# ============================================================
# 2. 판매량 차트
# ============================================================
st.markdown("### 판매량 차트")

col_left, col_right = st.columns(2)

with col_left:
    with st.spinner('일별 판매량 로딩 중...'):
        daily_sales = load_daily_sales(selected_year, selected_quarter, selected_month).to_pandas()
    
    fig = px.line(daily_sales, x='DATE', y='TOTAL_SALES')
    fig.update_traces(line=dict(color=COLORS['primary'], width=2))
    fig.update_layout(**get_chart_layout('일별 총 판매량'))
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="", range=[daily_sales['DATE'].min(), daily_sales['DATE'].max()])
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col_right:
    with st.spinner('지역별 판매량 로딩 중...'):
        region_sales = load_region_sales(selected_year, selected_quarter, selected_month).to_pandas()
    
    fig = px.bar(region_sales, x='REGION_LABEL', y='TOTAL_SALES',
                 color='REGION_LABEL', color_discrete_map=REGION_COLORS,
                 text='TOTAL_SALES')
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(**get_chart_layout('지역별 총 판매량'))
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# ============================================================
# 3. 구매건수 차트
# ============================================================
st.markdown("### 구매건수 차트")

col_left, col_right = st.columns(2)

with col_left:
    with st.spinner('일별 거래건수 로딩 중...'):
        daily_trans = load_daily_transactions(selected_year, selected_quarter, selected_month).to_pandas()
    
    fig = px.line(daily_trans, x='DATE', y='TOTAL_TRANSACTIONS')
    fig.update_traces(line=dict(color=COLORS['primary'], width=2))
    fig.update_layout(**get_chart_layout('일별 총 구매건수'))
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="", range=[daily_trans['DATE'].min(), daily_trans['DATE'].max()])
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col_right:
    with st.spinner('지역별 거래건수 로딩 중...'):
        region_trans = load_region_transactions(selected_year, selected_quarter, selected_month).to_pandas()
    
    fig = px.bar(region_trans, x='REGION_LABEL', y='TOTAL_TRANSACTIONS',
                 color='REGION_LABEL', color_discrete_map=REGION_COLORS,
                 text='TOTAL_TRANSACTIONS')
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(**get_chart_layout('지역별 총 구매건수'))
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# ============================================================
# 4. 매장 & 카테고리 차트
# ============================================================
st.markdown("### 매장 & 카테고리 차트")

col1, col2, col3 = st.columns(3)

# 매장별 TOP5
with col1:
    with st.spinner('매장별 판매량 로딩 중...'):
        store_sales = load_store_sales(selected_year, selected_quarter, selected_month).to_pandas()
    
    top5 = store_sales.head(5).copy()
    top5['STORE_NBR'] = 'store ' + top5['STORE_NBR'].astype(str)
    
    fig = px.bar(top5, x='TOTAL_SALES', y='STORE_NBR', orientation='h',
                 color_discrete_sequence=[COLORS['primary']],
                 text='TOTAL_SALES')
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside')
    fig.update_layout(**get_chart_layout('TOP5 매장'), yaxis_title=None)
    fig.update_layout(yaxis=dict(categoryorder='total ascending', type='category'))
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# 매장별 BOTTOM5
with col2:
    bottom5 = store_sales.tail(5).copy()
    bottom5['STORE_NBR'] = 'store ' + bottom5['STORE_NBR'].astype(str)
    
    fig = px.bar(bottom5, x='TOTAL_SALES', y='STORE_NBR', orientation='h',
                 color_discrete_sequence=[COLORS['accent2']],
                 text='TOTAL_SALES')
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside')
    fig.update_layout(**get_chart_layout('BOTTOM5 매장'), yaxis_title=None)
    fig.update_layout(yaxis=dict(categoryorder='total ascending', type='category'))
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Family TOP5 + Others
with col3:
    with st.spinner('카테고리별 판매량 로딩 중...'):
        family_sales = load_family_sales(selected_year, selected_quarter, selected_month).to_pandas()
    
    top5 = family_sales.head(5).copy()
    others_sum = family_sales.iloc[5:]['TOTAL_SALES'].sum() if len(family_sales) > 5 else 0
    
    if others_sum > 0:
        others_row = pd.DataFrame({'FAMILY': ['Others'], 'TOTAL_SALES': [others_sum]})
        chart_data = pd.concat([top5, others_row], ignore_index=True)
    else:
        chart_data = top5
    
    fig = px.pie(chart_data, values='TOTAL_SALES', names='FAMILY',
                 color_discrete_sequence=CHART_COLORS + ['#DDDDDD'],
                 hole=0.4)
    fig.update_layout(**get_chart_layout('TOP5 Family'))
    fig.update_layout(showlegend=False)
    fig.update_traces(
        textposition='outside', 
        textinfo='percent+label', 
        textfont_size=11,
        marker=dict(line=dict(color='white', width=1))
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(l=20, r=55, t=45, b=50)
    )
    fig.update_traces(domain=dict(x=[0.15, 0.85], y=[0.1, 0.95]))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# ============================================================
# 5. 프로모션 분석
# ============================================================
st.markdown("### 프로모션 차트")

col_left, col_right = st.columns(2)

with col_left:
    with st.spinner('프로모션 비율 로딩 중...'):
        store_promo = load_store_promo_ratio(selected_year, selected_quarter, selected_month).to_pandas()
    
    store_promo['STORE_NBR'] = 'store ' + store_promo['STORE_NBR'].astype(str)
    
    fig = px.bar(store_promo, x='PROMO_RATIO', y='STORE_NBR', orientation='h',
                 color_discrete_sequence=[COLORS['primary']],
                 text='PROMO_RATIO')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(**get_chart_layout('매장별 프로모션 비율 TOP5 (%)'), yaxis_title=None)
    fig.update_layout(yaxis=dict(categoryorder='total ascending', type='category'))
    fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col_right:
    with st.spinner('프로모션 Uplift 로딩 중...'):
        uplift = load_family_promo_uplift(selected_year, selected_quarter, selected_month).to_pandas()
    
    if len(uplift) > 0:
        fig = px.bar(uplift, x='UPLIFT', y='FAMILY', orientation='h',
                     color_discrete_sequence=[COLORS['primary']],
                     text='UPLIFT')
        fig.update_traces(texttemplate='%{text:.2f}배', textposition='inside')
        fig.update_layout(**get_chart_layout('Family별 프로모션 Uplift TOP5'), yaxis_title=None)
        fig.update_layout(yaxis=dict(categoryorder='total ascending', type='category'))
        fig.update_layout(margin=dict(l=20, r=55, t=50, b=40)) 
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("프로모션 데이터가 부족합니다.")

# ============================================================
# 푸터
# ============================================================
add_page_footer()