"""
Costa 지역 대시보드 (Total/Base/Holiday 통합)
- 클러스터: 중형(0)/소형(1)/대형(2) - Costa는 번호와 라벨이 다름!
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
    load_daily_sales_for_total,
    load_family_sales_for_total,
    load_transactions_for_total,
    load_train_data_for_holiday,
    load_promotion_timing_predictions,
    load_sales_forecast_predictions,
    get_family_list,
    load_all_forecast_predictions,
)
from utils.styles import apply_common_styles, get_chart_layout, add_sidebar_logo_bottom, add_page_footer, add_region_header
from utils.config import (
    CLUSTER_MODEL_MAP, COLORS, CHART_COLORS,
    CLUSTER_COLORS, CLUSTER_LABELS, CLUSTER_LABELS_STR,
    REGION_CLUSTER_LABELS, REGION_LABEL_TO_CLUSTER, REGION_CLUSTER_ORDER,
    HOLIDAYS, CLUSTER_BEST_MODEL, SPLIT_DATE
)
from utils.charts import create_promotion_effect_charts, get_best_timing_and_strategy

REGION = "Costa"

st.set_page_config(page_title=f"{REGION} | Favorita Dashboard", layout="wide", initial_sidebar_state="expanded")
apply_common_styles()

# 사이드바
with st.sidebar:
    st.markdown("---")
    st.markdown("### Cluster")
    cluster_labels = REGION_CLUSTER_LABELS[REGION]
    selected_label = st.radio(f"{REGION} 클러스터", options=cluster_labels, index=2, horizontal=False, label_visibility="collapsed")
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
    cluster_counts = clustering_df.group_by('cluster').agg(pl.len()).to_pandas().set_index('cluster')['len'].to_dict()
    total = clustering_df.height
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 매장 수", f"{total}개")
    col2.metric("대형 매장 수", f"{cluster_counts.get(2, 0)}개")
    col3.metric("중형 매장 수", f"{cluster_counts.get(0, 0)}개")
    col4.metric("소형 매장 수", f"{cluster_counts.get(1, 0)}개")
    
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
    
    cols = st.columns(3)
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
    if st.session_state.get('reset_total_filters_c', False):
        st.session_state.total_year_c = '전체'
        st.session_state.total_quarter_c = '전체'
        st.session_state.total_month_c = '전체'
        st.session_state.reset_total_filters_c = False

    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 0.5])

    year_filter = f_col1.selectbox('연도', ['전체', '2013', '2014', '2015', '2016'], key='total_year_c')
    quarter_filter = f_col2.selectbox('분기', ['전체', '1분기', '2분기', '3분기', '4분기'], key='total_quarter_c')

    q_map = {'1분기': [1,2,3], '2분기': [4,5,6], '3분기': [7,8,9], '4분기': [10,11,12]}
    month_opts = ['전체'] + [f'{i}월' for i in (range(1,13) if quarter_filter == '전체' else q_map[quarter_filter])]

    # 분기 변경 시 월 초기화
    if 'total_month_c' in st.session_state and st.session_state.total_month_c not in month_opts:
        st.session_state.total_month_c = '전체'

    month_filter = f_col3.selectbox('월', month_opts, key='total_month_c')

    with f_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button('필터 초기화', key='reset_total_c'):
            st.session_state.reset_total_filters_c = True
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
        
        # 최적화된 집계 함수 사용
        daily_sales = load_daily_sales_for_total(stores, year_filter, quarter_filter, month_filter)
        family_sales = load_family_sales_for_total(stores, year_filter, quarter_filter, month_filter)
        trans = load_transactions_for_total(stores, year_filter, quarter_filter, month_filter)
        
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            if daily_sales.height > 0:
                sales_pd = daily_sales.to_pandas()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sales_pd['date'], y=sales_pd['unit_sales'], marker=dict(color='#E31837')))
                fig.update_layout(height=250, margin=dict(l=50, r=30, t=40, b=50),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                                 title=dict(text='판매량', font=dict(size=14)))
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True)
            if trans.height > 0:
                trans_pd = trans.to_pandas()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=trans_pd['date'], y=trans_pd['transactions'], marker=dict(color='#666666')))
                fig.update_layout(height=250, margin=dict(l=50, r=30, t=40, b=50),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                                 title=dict(text='구매건수', font=dict(size=14)))
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            # 도넛 차트
            if family_sales.height > 0:
                family_pd = family_sales.to_pandas()
                
                # TOP5 카테고리
                family_agg = family_pd.groupby('family')['unit_sales'].sum().reset_index()
                family_agg = family_agg.sort_values('unit_sales', ascending=False)
                top5 = family_agg.head(5)
                others = family_agg.iloc[5:]['unit_sales'].sum()
                if others > 0:
                    top5 = pd.concat([top5, pd.DataFrame({'family': ['OTHERS'], 'unit_sales': [others]})], ignore_index=True)
                
                # 신선/비신선
                perish_agg = family_pd.groupby('perishable')['unit_sales'].sum().reset_index()
                perish_agg['label'] = perish_agg['perishable'].map({0: '비신선', 1: '신선'})
                
                fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                                   subplot_titles=('TOP5 카테고리', '신선/비신선'))
                
                # TOP5 카테고리 도넛
                fig.add_trace(go.Pie(labels=top5['family'], values=top5['unit_sales'], hole=0.5,
                                    marker=dict(colors=CHART_COLORS + ['#E0E0E0']),
                                    textposition='inside',
                                    textinfo='label+percent',
                                    textfont=dict(size=10)), row=1, col=1)
                
                # 신선/비신선 도넛
                fig.add_trace(go.Pie(labels=perish_agg['label'], values=perish_agg['unit_sales'], hole=0.5,
                                    marker=dict(colors=['#666666', '#E31837']),
                                    textposition='inside',
                                    textinfo='label+percent',
                                    textfont=dict(size=10)), row=1, col=2)
                
                fig.update_layout(height=250, margin=dict(l=0, r=40, t=20, b=40),
                                 paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # 카테고리 테이블
            if family_sales.height > 0:
                table_data = family_pd.groupby('family')['unit_sales'].sum().reset_index()
                table_data = table_data.sort_values('unit_sales', ascending=False)
                table_data.insert(0, '순위', range(1, len(table_data) + 1))
                table_data['unit_sales'] = table_data['unit_sales'].apply(lambda x: f"{x:,.0f}")
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
        selected_dates = st.slider("기간 선택", min_value=start_date, max_value=end_date, value=(start_date, end_date), format="YYYY-MM", key="base_slider_c")
        
        train_filtered = train_data.filter((pl.col("DATE") >= selected_dates[0]) & (pl.col("DATE") <= selected_dates[1]))
        pred_filtered = predictions.filter((pl.col("DATE") >= selected_dates[0]) & (pl.col("DATE") <= selected_dates[1]))
        
        st.markdown("---")
        st.markdown("### 총 판매량 실적·예측")
        
        train_daily = train_filtered.group_by("DATE").agg(pl.col("TOTAL_SALES").sum()).sort("DATE")
        pred_daily = pred_filtered.group_by("DATE").agg(pl.col("PRED_SALES").sum()).sort("DATE")
        
        fig = go.Figure()
        if train_daily.height > 0:
            fig.add_trace(go.Scatter(x=train_daily["DATE"].to_list(), y=train_daily["TOTAL_SALES"].to_list(),
                                    mode='lines', name='실제', line=dict(color='#E31837', width=2)))
        if pred_daily.height > 0:
            fig.add_trace(go.Scatter(x=pred_daily["DATE"].to_list(), y=pred_daily["PRED_SALES"].to_list(),
                                    mode='lines', name='예측', line=dict(color='#1E88E5', width=2, dash='dot')))
        
        fig.update_layout(height=400, margin=dict(l=50, r=30, t=40, b=50),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Family별 판매량")
        
        top_families = family_totals.head(5)['FAMILY'].to_list()
        
        cols = st.columns(min(5, len(top_families)))
        for idx, fam in enumerate(top_families):
            with cols[idx]:
                fam_train = train_filtered.filter(pl.col("FAMILY") == fam)
                fam_pred = pred_filtered.filter(pl.col("FAMILY") == fam)
                
                fig = go.Figure()
                if fam_train.height > 0:
                    daily = fam_train.group_by("DATE").agg(pl.col("TOTAL_SALES").sum()).sort("DATE")
                    fig.add_trace(go.Scatter(x=daily["DATE"].to_list(), y=daily["TOTAL_SALES"].to_list(),
                                            mode='lines', name='실제', line=dict(color='#E31837')))
                if fam_pred.height > 0:
                    daily = fam_pred.group_by("DATE").agg(pl.col("PRED_SALES").sum()).sort("DATE")
                    fig.add_trace(go.Scatter(x=daily["DATE"].to_list(), y=daily["PRED_SALES"].to_list(),
                                            mode='lines', name='예측', line=dict(color='#1E88E5', dash='dot')))
                
                fig.update_layout(height=200, margin=dict(l=30, r=10, t=30, b=30),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                                 title=dict(text=fam, font=dict(size=11)), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# HOLIDAY 탭
with tab_holiday:
    st.markdown("## 휴일별 판매 분석")
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        holiday_opts = list(HOLIDAYS.keys())
        h_holiday = st.selectbox("휴일 선택", holiday_opts, key='holiday_select_c')
    with h_col2:
        family_list = get_family_list(REGION)
        h_family = st.selectbox("카테고리 선택", ['전체'] + family_list, key='family_select_c')
    
    # 프로모션 예측 데이터 로드
    promotion_df = load_promotion_timing_predictions(REGION, holiday_filter=h_holiday, family_filter=h_family if h_family != '전체' else None)
    forecast_df = load_all_forecast_predictions(REGION)
    
    stores = clustering_df.select('store_nbr').to_series().to_list()
    
    # 실제 판매 데이터
    st.markdown("## 휴일 전후 실제 판매량")
    
    holiday_info = HOLIDAYS.get(h_holiday, {})
    holiday_month = holiday_info.get('month')
    holiday_day = holiday_info.get('day')
    
    if holiday_month and holiday_day:
        with st.spinner("데이터 로딩 중..."):
            train_holiday = load_train_data_for_holiday(stores, year_filter='전체')
        
        if train_holiday.height > 0:
            pdf = train_holiday.to_pandas()
            pdf['month'] = pd.to_datetime(pdf['date']).dt.month
            pdf['day'] = pd.to_datetime(pdf['date']).dt.day
            
            # 해당 월 필터
            pdf = pdf[pdf['month'] == holiday_month]
            
            # 타이밍 분류
            pdf['timing'] = pdf['day'].apply(
                lambda d: 'During' if d == holiday_day else ('Before' if d < holiday_day else 'After')
            )
            
            # Family 필터
            if h_family != '전체':
                pdf = pdf[pdf['family'] == h_family]
            
            pdf = pdf[pdf['timing'].notna()]
            
            if len(pdf) > 0:
                timing_sales = pdf.groupby('timing')['unit_sales'].sum().reindex(['Before', 'During', 'After']).fillna(0)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=timing_sales.index, y=timing_sales.values,
                                    marker=dict(color=['#666666', '#E31837', '#999999']),
                                    text=timing_sales.values, texttemplate='%{text:,.0f}', textposition='outside'))
                fig.update_layout(height=400, margin=dict(l=0, r=50, t=20, b=30),
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
            "<strong>Costa 지역 전체 매장</strong>에서 
            <strong>{HOLIDAYS[h_holiday]['name']}</strong>에 
            <strong>{h_family if h_family != '전체' else '전체 카테고리'}</strong> 
            프로모션을 언제 하면 가장 효과적인가?"
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if promotion_df.height > 0:
        timing_stats = promotion_df.group_by('timing').agg(pl.col('predicted_uplift').mean().alias('avg_uplift')).to_pandas()
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
