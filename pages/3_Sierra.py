"""
Sierra 지역 대시보드 (Total/Base/Holiday 통합)
- 클러스터: 소형(0)/중형(1) - 2개만 있음
"""
import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date
import numpy as np

import sys
sys.path.append('..')
from utils.snowflake_conn import (
    load_clustering_results,
    load_train_data_with_cluster,
    load_predictions,
    load_train_total_sales,
    load_train_data_for_total,
    load_transactions_for_total,
    load_train_data_for_holiday,
    load_promotion_timing_predictions,
    load_sales_forecast_predictions,
    get_family_list,
    load_all_forecast_predictions,  # 이 줄 추가!
)
from utils.styles import apply_common_styles, get_chart_layout, add_sidebar_logo_bottom, add_page_footer, add_region_header
from utils.config import (
    CLUSTER_MODEL_MAP, COLORS, CHART_COLORS,
    CLUSTER_COLORS, CLUSTER_LABELS, CLUSTER_LABELS_STR,
    REGION_CLUSTER_LABELS, REGION_LABEL_TO_CLUSTER, REGION_CLUSTER_ORDER,
    HOLIDAYS, CLUSTER_BEST_MODEL, SPLIT_DATE
)

REGION = "Sierra"

st.set_page_config(page_title=f"{REGION} | Favorita Dashboard", layout="wide", initial_sidebar_state="expanded")
apply_common_styles()

# 사이드바
with st.sidebar:
    st.markdown("---")
    st.markdown("### Cluster")
    cluster_labels = REGION_CLUSTER_LABELS[REGION]
    selected_label = st.radio(f"{REGION} 클러스터", options=cluster_labels, index=0, horizontal=False, label_visibility="collapsed")
    selected_cluster = REGION_LABEL_TO_CLUSTER[REGION][selected_label]
    model_type = CLUSTER_MODEL_MAP[(REGION, selected_cluster)]
    cluster_key = f"{REGION}_{selected_cluster}"
    st.markdown("---")
    add_sidebar_logo_bottom()

add_region_header(REGION, selected_label)
tab_total, tab_base, tab_holiday = st.tabs(["Total", "Base", "Holiday"])

# TOTAL 탭
with tab_total:
    with st.spinner('데이터를 불러오는 중...'):
        clustering_df = load_clustering_results(REGION)
    
    st.markdown("## 매장 규모별 현황")
    cluster_counts = clustering_df.group_by('cluster').agg(pl.count()).to_pandas().set_index('cluster')['count'].to_dict()
    total = clustering_df.height
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 매장 수", f"{total}개")
    col2.metric("중형 매장 수", f"{cluster_counts.get(1, 0)}개")
    col3.metric("소형 매장 수", f"{cluster_counts.get(0, 0)}개")
    
    st.markdown("---")
    
    # 레이더 차트
    feature_cols = ['avg_monthly_sales', 'avg_monthly_sku', 'perishable_pct', 'avg_monthly_transactions', 'sales_cv', 'upt']
    labels = ['판매량', '고유 아이템', '신선식품', '거래건수', '판매 변동성', 'UPT']
    all_mins = {col: clustering_df[col].min() for col in feature_cols if col in clustering_df.columns}
    all_maxs = {col: clustering_df[col].max() for col in feature_cols if col in clustering_df.columns}
    
    radar_data = {}
    for cluster_num in REGION_CLUSTER_ORDER[REGION]:
        cluster_data = clustering_df.filter(pl.col('cluster') == cluster_num)
        if cluster_data.height == 0:
            continue
        values = []
        for col in feature_cols:
            if col in cluster_data.columns:
                mean_val = cluster_data[col].mean()
                min_val, max_val = all_mins[col], all_maxs[col]
                normalized = (mean_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                values.append(normalized)
            else:
                values.append(0)
        radar_data[cluster_num] = values
    
    cols = st.columns(2)
    for idx, cluster_num in enumerate(REGION_CLUSTER_ORDER[REGION]):
        with cols[idx]:
            if cluster_num in radar_data:
                label = CLUSTER_LABELS.get((REGION, cluster_num), f"클러스터 {cluster_num}")
                st.markdown(f"### {label} 매장")
                stores = clustering_df.filter(pl.col('cluster') == cluster_num).select('store_nbr').to_series().to_list()
                st.caption(f"매장: {', '.join(map(str, stores))}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=radar_data[cluster_num], theta=labels, fill='toself',
                    fillcolor=CLUSTER_COLORS[cluster_num], opacity=0.4,
                    line=dict(color=CLUSTER_COLORS[cluster_num], width=3)))
                fig.update_layout(
                                    polar=dict(
                                        domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]),
                                        radialaxis=dict(visible=True, range=[0, 1]),
                                        bgcolor='white'
                                    ),
                                    showlegend=False,
                                    height=500,
                                    margin=dict(l=10, r=50, t=0, b=40),
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 필터
    if st.session_state.get('reset_total_filters_s', False):
        st.session_state.total_year_s = '전체'
        st.session_state.total_quarter_s = '전체'
        st.session_state.total_month_s = '전체'
        st.session_state.reset_total_filters_s = False

    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 0.5])

    year_filter = f_col1.selectbox('연도', ['전체', '2013', '2014', '2015', '2016'], key='total_year_s')
    quarter_filter = f_col2.selectbox('분기', ['전체', '1분기', '2분기', '3분기', '4분기'], key='total_quarter_s')

    q_map = {'1분기': [1,2,3], '2분기': [4,5,6], '3분기': [7,8,9], '4분기': [10,11,12]}
    month_opts = ['전체'] + [f'{i}월' for i in (range(1,13) if quarter_filter == '전체' else q_map[quarter_filter])]

    # 분기 변경 시 월 초기화
    if 'total_month_s' in st.session_state and st.session_state.total_month_s not in month_opts:
        st.session_state.total_month_s = '전체'

    month_filter = f_col3.selectbox('월', month_opts, key='total_month_s')

    with f_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button('필터 초기화', key='reset_total_s'):
            st.session_state.reset_total_filters_s = True
            st.rerun()

    # 필터 적용 후 표시
    year_display = "전체 기간" if year_filter == "전체" else f"{year_filter}년"
    quarter_display = "" if quarter_filter == "전체" else f" {quarter_filter}"
    month_display = "" if month_filter == "전체" else f" {month_filter}"
    st.markdown(f"**{year_display}{quarter_display}{month_display}** 데이터를 분석합니다.")
    st.markdown("---")
    
    # 클러스터별 상세
    for cluster_num in REGION_CLUSTER_ORDER[REGION]:
        if cluster_num not in cluster_counts:
            continue
        label = CLUSTER_LABELS.get((REGION, cluster_num), f"클러스터 {cluster_num}")
        st.markdown(f'<div class="cluster-header">{label} 매장 ({cluster_counts.get(cluster_num, 0)}개)</div>', unsafe_allow_html=True)
        
        stores = clustering_df.filter(pl.col('cluster') == cluster_num).select('store_nbr').to_series().to_list()
        train = load_train_data_for_total(stores, year_filter, quarter_filter, month_filter)
        trans = load_transactions_for_total(stores, year_filter, quarter_filter, month_filter)
        
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            if train.height > 0:
                sales_daily = train.group_by('date').agg(pl.col('unit_sales').sum()).sort('date').to_pandas()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sales_daily['date'], y=sales_daily['unit_sales'], marker=dict(color='#E31837')))
                fig.update_layout(height=250, margin=dict(l=50, r=30, t=40, b=50),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                                 title=dict(text='구매건수', font=dict(size=14)))
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True)
            if trans.height > 0:
                trans_daily = trans.group_by('date').agg(pl.col('transactions').sum()).sort('date').to_pandas()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=trans_daily['date'], y=trans_daily['transactions'], marker=dict(color='#666666')))
                fig.update_layout(height=250, margin=dict(l=50, r=30, t=40, b=50),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                                 title=dict(text='구매건수', font=dict(size=14)))
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            # 도넛 차트
            if train.height > 0:
                family_sales = train.group_by('family').agg(pl.col('unit_sales').sum().alias('total')).sort('total', descending=True).to_pandas()
                top5 = family_sales.head(5)
                others = family_sales.iloc[5:]['total'].sum()
                if others > 0:
                    top5 = pd.concat([top5, pd.DataFrame({'family': ['OTHERS'], 'total': [others]})], ignore_index=True)
                
                fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                                   subplot_titles=('TOP5 카테고리', '신선/비신선'))
                
                # TOP5 카테고리 도넛
                fig.add_trace(go.Pie(labels=top5['family'], values=top5['total'], hole=0.5,
                                    marker=dict(colors=CHART_COLORS + ['#E0E0E0']),
                                    textposition='inside',  # 라벨 밖으로
                                    textinfo='label+percent',  # 라벨 + 퍼센트
                                    textfont=dict(size=10)), row=1, col=1)
                
                # 신선/비신선 도넛
                perish = train.group_by('perishable').agg(pl.col('unit_sales').sum().alias('total')).to_pandas()
                perish['label'] = perish['perishable'].map({0: '비신선', 1: '신선'})

                fig.add_trace(go.Pie(labels=perish['label'], values=perish['total'], hole=0.5,
                                    marker=dict(colors=['#666666', '#E31837']),
                                    textposition='inside',  # 라벨 밖으로
                                    textinfo='label+percent',  # 라벨 + 퍼센트
                                    textfont=dict(size=10)), row=1, col=2)
                
                fig.update_layout(height=250, margin=dict(l=0, r=40, t=40, b=60),  # 하단 여유 추가
                                 paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # 카테고리 테이블
            if train.height > 0:
                table_data = train.group_by('family').agg(pl.col('unit_sales').sum().alias('총 판매량')).sort('총 판매량', descending=True).to_pandas()
                table_data.insert(0, '순위', range(1, len(table_data) + 1))
                table_data['총 판매량'] = table_data['총 판매량'].apply(lambda x: f"{x:,.0f}")
                table_data.columns = ['순위', '카테고리', '총 판매량']
                st.dataframe(table_data, use_container_width=True, hide_index=True, height=250)

# BASE 탭
with tab_base:
    try:
        train_data = load_train_data_with_cluster(REGION, selected_cluster)
        predictions = load_predictions(REGION, selected_cluster, model_type)
        family_totals = load_train_total_sales(REGION, selected_cluster)
        base_loaded = True
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        base_loaded = False
    
    if base_loaded:
        start_date, end_date = date(2013, 1, 1), date(2017, 8, 15)
        selected_dates = st.slider("기간 선택", min_value=start_date, max_value=end_date, value=(start_date, end_date), format="YYYY-MM", key="base_slider_s")
        
        train_filtered = train_data.filter((pl.col("DATE") >= selected_dates[0]) & (pl.col("DATE") <= selected_dates[1]))
        pred_filtered = predictions.filter((pl.col("DATE") >= selected_dates[0]) & (pl.col("DATE") <= selected_dates[1]))
        
        st.markdown("---")
        st.markdown("### 총 판매량 실적·예측")
        
        train_daily = train_filtered.group_by("DATE").agg(pl.col("TOTAL_SALES").sum()).sort("DATE").to_pandas()
        pred_daily = pred_filtered.group_by("DATE").agg(pl.col("PRED_SALES").sum()).sort("DATE").to_pandas()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_daily['DATE'], y=train_daily['TOTAL_SALES'], mode='lines', name='Train', line=dict(color=COLORS['train_line'], width=2)))
        fig.add_trace(go.Scatter(x=pred_daily['DATE'], y=pred_daily['PRED_SALES'], mode='lines', name='Test', line=dict(color=COLORS['test_line'], width=2)))
        fig.add_vline(x="2016-08-16", line_width=2, line_dash="dash", line_color="#E31837")
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        st.markdown("### 상위 Family 판매량")
        top3 = family_totals.head(3).select("FAMILY").to_series().to_list()
        selected_top = st.selectbox("", options=top3, index=0, key="top_family_s")
        
        family_start = date(2015, 8, 16)
        train_top = train_data.filter((pl.col("FAMILY") == selected_top) & (pl.col("DATE") >= family_start)).group_by("DATE").agg(pl.col("TOTAL_SALES").sum()).sort("DATE").to_pandas()
        pred_top = predictions.filter(pl.col("FAMILY") == selected_top).group_by("DATE").agg(pl.col("PRED_SALES").sum()).sort("DATE").to_pandas()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_top['DATE'], y=train_top['TOTAL_SALES'], mode='lines', line=dict(color=COLORS['train_line'], width=2)))
        fig.add_trace(go.Scatter(x=pred_top['DATE'], y=pred_top['PRED_SALES'], mode='lines', line=dict(color=COLORS['test_line'], width=2)))
        fig.add_vline(x="2016-08-16", line_width=2, line_dash="dash", line_color="#E31837")
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30, t=20))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        st.markdown("### 하위 Family 판매량")
        bottom3 = family_totals.tail(3).select("FAMILY").to_series().to_list()
        selected_bottom = st.selectbox("", options=bottom3, index=0, key="bottom_family_s")
        
        train_bottom = train_data.filter((pl.col("FAMILY") == selected_bottom) & (pl.col("DATE") >= family_start)).group_by("DATE").agg(pl.col("TOTAL_SALES").sum()).sort("DATE").to_pandas()
        pred_bottom = predictions.filter(pl.col("FAMILY") == selected_bottom).group_by("DATE").agg(pl.col("PRED_SALES").sum()).sort("DATE").to_pandas()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_bottom['DATE'], y=train_bottom['TOTAL_SALES'], mode='lines', line=dict(color=COLORS['train_line'], width=2)))
        fig.add_trace(go.Scatter(x=pred_bottom['DATE'], y=pred_bottom['PRED_SALES'], mode='lines', line=dict(color=COLORS['test_line'], width=2)))
        fig.add_vline(x="2016-08-16", line_width=2, line_dash="dash", line_color="#E31837")
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30, t=20))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ============================================================
# HOLIDAY 탭
# ============================================================
with tab_holiday:
    with st.spinner('데이터를 불러오는 중...'):
        clustering_df_h = load_clustering_results(REGION)
    
    st.markdown("## 프로모션 최적 시점 분석")
    
    # 필터
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        st.markdown("#### 축제")
        holiday_opts = list(HOLIDAYS.keys())
        h_holiday = st.radio("축제", options=holiday_opts,
                            format_func=lambda x: HOLIDAYS[x]['name'],
                            key='h_holiday_sierra', label_visibility='collapsed',
                            horizontal=True)
    
    with f_col2:
        st.markdown("#### 상품 카테고리")
        families = get_family_list(REGION)
        family_opts = ['전체'] + families
        h_family = st.selectbox('카테고리', family_opts, key='h_family_sierra', label_visibility='collapsed')
    
    with f_col3:
        st.markdown("#### 연도")
        h_year = st.selectbox('연도', ['전체', '2014', '2015', '2016'], key='h_year_sierra', label_visibility='collapsed')
    
    st.markdown("---")
    
    # 데이터 로드 (전체 매장 대상)
    h_stores = clustering_df_h.select('store_nbr').to_series().to_list()
    
    with st.spinner('데이터 분석 중...'):
        train_h = load_train_data_for_holiday(h_stores, h_year)
        promotion_df = load_promotion_timing_predictions(REGION, None, h_holiday, h_family if h_family != '전체' else None)
        forecast_df = load_all_forecast_predictions(REGION)
    
    # 차트
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### 연도별 판매량 Top 5 카테고리")
        if train_h.height > 0:
            if h_year != '전체':
                filtered = train_h.filter(pl.col('date').dt.year() == int(h_year))
            else:
                filtered = train_h
            
            if filtered.height > 0:
                family_sales = filtered.group_by('family').agg(pl.col('unit_sales').sum().alias('total')).sort('total', descending=True).to_pandas()
                top5 = family_sales.head(5)
                others = family_sales.iloc[5:]['total'].sum() if len(family_sales) > 5 else 0
                if others > 0:
                    top5 = pd.concat([top5, pd.DataFrame({'family': ['OTHERS'], 'total': [others]})], ignore_index=True)
                
                fig = go.Figure()
                fig.add_trace(go.Pie(labels=top5['family'], values=top5['total'], hole=0.5,
                                    marker=dict(colors=CHART_COLORS + ['#E0E0E0']),
                                    textposition='outside', textinfo='label+percent',
                                    textfont=dict(size=12)))
                fig.update_layout(height=400, margin=dict(l=20, r=50, t=20, b=70),
                                 paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("### 축제 기간별 판매량")
        if train_h.height > 0 and h_family != '전체':
            filtered = train_h.filter(pl.col('family') == h_family)
        else:
            filtered = train_h
        
        if filtered.height > 0:
            holiday_date = HOLIDAYS[h_holiday]['date']
            filtered = filtered.with_columns([
                pl.col('date').dt.month().alias('month'),
                pl.col('date').dt.day().alias('day')
            ])
            h_month, h_day = int(holiday_date.split('-')[0]), int(holiday_date.split('-')[1])
            
            def get_timing(row):
                m, d = row['month'], row['day']
                if m == h_month and d == h_day:
                    return 'During'
                elif m == h_month and d < h_day:
                    return 'Before'
                elif m == h_month and d > h_day:
                    return 'After'
                return None
            
            pdf = filtered.to_pandas()
            pdf['timing'] = pdf.apply(get_timing, axis=1)
            pdf = pdf[pdf['timing'].notna()]
            
            if len(pdf) > 0:
                timing_sales = pdf.groupby('timing')['unit_sales'].sum().reindex(['Before', 'During', 'After']).fillna(0)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=timing_sales.index, y=timing_sales.values,
                                    marker=dict(color=['#666666', '#E31837', '#999999']),
                                    text=timing_sales.values, texttemplate='%{text:,.0f}', textposition='outside'))
                fig.update_layout(height=400, margin=dict(l=0, r=50, t=50, b=30),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("해당 축제 기간 데이터가 없습니다.")
    
    st.markdown("---")
    
    # 프로모션 최적 시점 예측
    st.markdown("## 프로모션 최적 시점 예측")
    
    st.markdown(f"""
    <div class="scenario-box">
        <h4 style='margin-bottom: 12px; color: #E31837;'>비즈니스 질문</h4>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
            "<strong>Sierra 지역 전체 매장</strong>에서 
            <strong>{HOLIDAYS[h_holiday]['name']}</strong>에 
            <strong>{h_family if h_family != '전체' else '전체 카테고리'}</strong> 
            프로모션을 언제 하면 가장 효과적인가?"
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if promotion_df.height > 0:
        timing_stats = promotion_df.group_by('timing').agg(pl.col('uplift_pred').mean().alias('avg_uplift')).to_pandas()
        timing_dict = dict(zip(timing_stats['timing'], timing_stats['avg_uplift']))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            val = timing_dict.get('before', 0)
            st.metric("Before", f"{'+' if val >= 0 else ''}{val:.1%}")
        with col2:
            val = timing_dict.get('during', 0)
            st.metric("During", f"{'+' if val >= 0 else ''}{val:.1%}")
        with col3:
            val = timing_dict.get('after', 0)
            st.metric("After", f"{'+' if val >= 0 else ''}{val:.1%}")
        
        st.markdown("---")
        
        # 프로모션 효과 비교
        st.markdown("## 프로모션 효과 비교")
        
        if forecast_df.height > 0:
            result = create_promotion_effect_charts(forecast_df, promotion_df, h_holiday, h_family)
            
            if result:
                fig_effect, uplift_info = result
                st.plotly_chart(fig_effect, use_container_width=True)
                
                st.markdown("---")
                
                # 프로모션 전략 제안
                st.markdown("## 프로모션 전략 제안")
                
                best_timing, best_uplift, strategy, show_balloons = get_best_timing_and_strategy(uplift_info)
                
                if strategy:
                    sign = '+' if strategy['best_uplift'] >= 0 else ''
                    increase_sign = '+' if strategy['best_increase'] >= 0 else ''
                    safety_sign = '+' if strategy['safety_stock_pct'] >= 0 else ''
                    
                    action_items_html = ''.join([
                        f'<div style="display: flex; align-items: center; padding: 8px 0; color: #333333;">'
                        f'<span style="color: #28a745; margin-right: 8px; font-weight: bold;">✓</span> {action}</div>'
                        for action in strategy['actions']
                    ])
                    
                    strategy_html = f"""
                    <div style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 16px 0;">
                        <div style="display: flex; gap: 16px; margin-bottom: 24px;">
                            <div style="flex: 1; background: linear-gradient(135deg, #E31837 0%, #B71430 100%); color: white; border-radius: 12px; padding: 24px; text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px;">{strategy['best_timing']}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8;">추천 시점</div>
                            </div>
                            <div style="flex: 1; background: #F8F9FA; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E9ECEF;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #E31837;">{sign}{strategy['best_uplift']:.1%}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8; color: #666;">예상 Uplift</div>
                            </div>
                            <div style="flex: 1; background: #F8F9FA; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E9ECEF;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #2D2D2D;">{strategy['best_with_promo']:,.0f}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8; color: #666;">예상 판매량</div>
                            </div>
                            <div style="flex: 1; background: #F8F9FA; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E9ECEF;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #27AE60;">{increase_sign}{strategy['best_increase']:,.0f}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8; color: #666;">예상 증가량</div>
                            </div>
                        </div>
                        <div style="height: 1px; background: #E9ECEF; margin: 24px 0;"></div>
                        <h3 style="color: #2D2D2D; margin-bottom: 12px; font-size: 1.25rem;">{strategy['title']}</h3>
                        <p style="color: #666666; font-size: 1rem; line-height: 1.6; margin-bottom: 20px;">{strategy['detail']}</p>
                        <div style="background: #F8F9FA; padding: 16px 20px; border-radius: 8px;">
                            <strong style="color: #2D2D2D;">액션 가이드</strong>
                            <div style="margin-top: 12px;">{action_items_html}</div>
                        </div>
                    </div>
                    """
                    st.markdown(strategy_html, unsafe_allow_html=True)
                    
                    # 안전재고 전략
                    st.markdown("## 안전재고 전략")
                    
                    safety_guide_items = [
                        f"기본 재고: {strategy['best_with_promo']:,.0f}개 확보",
                        f"안전재고: {safety_sign}{strategy['safety_stock_qty']:,.0f}개 추가 확보",
                        f"총 권장 재고: {strategy['total_recommended_qty']:,.0f}개",
                        "계산 기준: 95% 신뢰수준 (Z=1.96)"
                    ]
                    safety_guide_html = ''.join([
                        f'<div style="display: flex; align-items: center; padding: 8px 0; color: #333333;">'
                        f'<span style="color: #28a745; margin-right: 8px; font-weight: bold;">✓</span> {item}</div>'
                        for item in safety_guide_items
                    ])
                    
                    safety_html = f"""
                    <div style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 16px 0;">
                        <div style="display: flex; gap: 16px; margin-bottom: 24px;">
                            <div style="flex: 1; background: linear-gradient(135deg, #E67E22 0%, #D35400 100%); color: white; border-radius: 12px; padding: 24px; text-align: center;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px;">{strategy['best_timing']}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8;">재고 확보 시점</div>
                            </div>
                            <div style="flex: 1; background: #F8F9FA; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E9ECEF;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #E67E22;">{safety_sign}{strategy['safety_stock_pct']:.1f}%</div>
                                <div style="font-size: 0.85rem; opacity: 0.8; color: #666;">권장 비율</div>
                            </div>
                            <div style="flex: 1; background: #F8F9FA; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E9ECEF;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #2D2D2D;">{safety_sign}{strategy['safety_stock_qty']:,.0f}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8; color: #666;">권장 수량</div>
                            </div>
                            <div style="flex: 1; background: #F8F9FA; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E9ECEF;">
                                <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: {strategy['safety_volatility_color']};">{strategy['safety_volatility']}</div>
                                <div style="font-size: 0.85rem; opacity: 0.8; color: #666;">수요 변동성</div>
                            </div>
                        </div>
                        <div style="height: 1px; background: #E9ECEF; margin: 24px 0;"></div>
                        <h3 style="color: #2D2D2D; margin-bottom: 12px; font-size: 1.25rem;">{strategy['safety_msg']}</h3>
                        <p style="color: #666666; font-size: 1rem; line-height: 1.6; margin-bottom: 20px;">{strategy['safety_detail']}</p>
                        <div style="background: #F8F9FA; padding: 16px 20px; border-radius: 8px;">
                            <strong style="color: #2D2D2D;">재고 가이드</strong>
                            <div style="margin-top: 12px;">{safety_guide_html}</div>
                        </div>
                    </div>
                    """
                    st.markdown(safety_html, unsafe_allow_html=True)
            else:
                st.info("프로모션 효과 분석을 위한 데이터가 충분하지 않습니다.")
        else:
            st.info("예측 데이터가 없습니다.")
    else:
        st.warning("프로모션 예측 데이터가 없습니다.")

add_page_footer()
