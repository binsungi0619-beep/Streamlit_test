"""
차트 및 전략 생성 헬퍼 함수
- 모든 지역 페이지에서 공통 사용
"""
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.config import HOLIDAYS


def create_promotion_effect_charts(forecast_df, promotion_df, holiday_name, family_filter):
    """프로모션 효과 비교 차트 (3개 서브플롯)"""
    
    if forecast_df.height == 0 or promotion_df.height == 0:
        return None
    
    holiday_date = HOLIDAYS[holiday_name]['date']
    h_month, h_day = int(holiday_date.split('-')[0]), int(holiday_date.split('-')[1])
    
    # 컬럼명 확인
    date_col = 'DATE' if 'DATE' in forecast_df.columns else 'date'
    family_col = 'FAMILY' if 'FAMILY' in forecast_df.columns else 'family'

    # pred_col 체크 (소문자 우선)
    available_cols = forecast_df.columns
    if 'pred_sales' in available_cols:
        pred_col = 'pred_sales'
    elif 'PRED_SALES' in available_cols:
        pred_col = 'PRED_SALES'
    elif 'y_pred_avg' in available_cols:
        pred_col = 'y_pred_avg'
    else:
        raise ValueError(f"예측값 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {available_cols}")
    
    forecast_df = forecast_df.with_columns([
        pl.col(date_col).dt.month().alias('month'),
        pl.col(date_col).dt.day().alias('day')
    ])
    
    # Polars에서 직접 timing 계산
    forecast_df = forecast_df.with_columns([
        pl.when((pl.col('month') == h_month) & (pl.col('day') == h_day)).then(pl.lit('during'))
          .when((pl.col('month') == h_month) & (pl.col('day') < h_day)).then(pl.lit('before'))
          .when((pl.col('month') == h_month) & (pl.col('day') > h_day)).then(pl.lit('after'))
          .otherwise(pl.lit(None)).alias('timing')
    ])
    
    forecast_df = forecast_df.filter(pl.col('timing').is_not_null())
    
    if family_filter and family_filter != '전체':
        if family_col in forecast_df.columns:
            forecast_df = forecast_df.filter(pl.col(family_col) == family_filter)
    
    if forecast_df.height == 0:
        return None
    
    # Polars에서 집계 (이미 위에서 정의한 pred_col 사용)
    base_pred = forecast_df.group_by('timing').agg(pl.col(pred_col).sum().alias('total')).to_pandas()
    base_pred_dict = dict(zip(base_pred['timing'], base_pred['total']))
    
    before_pred = base_pred_dict.get('before', 0)
    during_pred = base_pred_dict.get('during', 0)
    after_pred = base_pred_dict.get('after', 0)
    
    promotion_stats = (
        promotion_df
        .group_by('timing')
        .agg(pl.col('uplift_pred').mean().alias('avg_uplift'))
        .to_pandas()
    )
    
    uplift_dict = dict(zip(promotion_stats['timing'], promotion_stats['avg_uplift']))
    
    before_uplift = uplift_dict.get('before', 0)
    during_uplift = uplift_dict.get('during', 0)
    after_uplift = uplift_dict.get('after', 0)
    
    before_with_promo = before_pred * (1 + before_uplift)
    during_with_promo = during_pred * (1 + during_uplift)
    after_with_promo = after_pred * (1 + after_uplift)
    
    # 안전재고 계산
    std_dev = forecast_df.select(pl.col(pred_col).std()).item()
    mean_val = forecast_df.select(pl.col(pred_col).mean()).item()
    safety_stock_pct = (1.96 * std_dev / mean_val * 100) if mean_val > 0 else 0
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Before (축제 전)', 'During (축제 당일)', 'After (축제 후)'),
        horizontal_spacing=0.1
    )
    
    fig.add_trace(go.Bar(
        x=['기본 예측', '프로모션 적용'],
        y=[before_pred, before_with_promo],
        marker=dict(color=['#CCCCCC', '#E31837']),  
        text=[f'{before_pred:,.0f}', f'{before_with_promo:,.0f}'],
        textposition='outside',
        textfont=dict(size=10),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=['기본 예측', '프로모션 적용'],
        y=[during_pred, during_with_promo],
        marker=dict(color=['#CCCCCC', '#B71430']),
        text=[f'{during_pred:,.0f}', f'{during_with_promo:,.0f}'],
        textposition='outside',
        textfont=dict(size=10),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=['기본 예측', '프로모션 적용'],
        y=[after_pred, after_with_promo],
        marker=dict(color=['#CCCCCC', '#2D2D2D']),
        text=[f'{after_pred:,.0f}', f'{after_with_promo:,.0f}'],
        textposition='outside',
        textfont=dict(size=10),
        showlegend=False
    ), row=1, col=3)
    
    max_val = max(before_with_promo, during_with_promo, after_with_promo, 
                  before_pred, during_pred, after_pred) * 1.3
    
    fig.update_yaxes(range=[0, max_val], showgrid=True, gridcolor='#F0F0F0')
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white'
    )
    
    uplift_info = {
        'before': {'pred': before_pred, 'uplift': before_uplift, 'with_promo': before_with_promo},
        'during': {'pred': during_pred, 'uplift': during_uplift, 'with_promo': during_with_promo},
        'after': {'pred': after_pred, 'uplift': after_uplift, 'with_promo': after_with_promo},
        'safety_stock_pct': safety_stock_pct
    }
    
    return fig, uplift_info


def get_best_timing_and_strategy(uplift_info):
    """최적 시점 및 전략 메시지 생성"""
    
    if not uplift_info:
        return None, None, None, False
    
    timings = ['before', 'during', 'after']
    uplifts = [uplift_info[t]['uplift'] for t in timings]
    preds = [uplift_info[t]['pred'] for t in timings]
    with_promos = [uplift_info[t]['with_promo'] for t in timings]
    
    best_timing_idx = uplifts.index(max(uplifts))
    best_timing = timings[best_timing_idx]
    best_uplift = uplifts[best_timing_idx]
    best_pred = preds[best_timing_idx]
    best_with_promo = with_promos[best_timing_idx]
    best_increase = best_with_promo - best_pred
    
    safety_stock_pct = uplift_info.get('safety_stock_pct', 0)
    safety_stock_qty = best_with_promo * (safety_stock_pct / 100) if safety_stock_pct > 0 else 0
    total_recommended_qty = best_with_promo + safety_stock_qty
    
    timing_korean = {'before': 'Before', 'during': 'During', 'after': 'After'}
    timing_desc = {'before': '축제 전', 'during': '축제 당일', 'after': '축제 후'}
    
    if safety_stock_pct <= 15:
        safety_volatility = "낮음"
        safety_volatility_color = "#28a745"
        safety_msg = f"{timing_korean[best_timing]}({timing_desc[best_timing]}) 시점까지 기본 재고 대비 {safety_stock_pct:.1f}%의 안전재고를 확보하세요."
        safety_detail = "수요가 안정적으로 예측됩니다. 최소한의 안전재고만 확보해도 재고 부족의 위험이 낮습니다."
    elif safety_stock_pct <= 30:
        safety_volatility = "보통"
        safety_volatility_color = "#FF9800"
        safety_msg = f"{timing_korean[best_timing]}({timing_desc[best_timing]}) 시점까지 기본 재고 대비 {safety_stock_pct:.1f}%의 안전재고 확보를 권장합니다."
        safety_detail = "큰 수요 변동은 없을 것으로 보이지만, 프로모션 기간 중 예상보다 판매가 늘 수 있으므로 품절 방지를 막기 위한 여유 재고를 준비하세요."
    elif safety_stock_pct <= 50:
        safety_volatility = "높음"
        safety_volatility_color = "#E31837"
        safety_msg = f"{timing_korean[best_timing]}({timing_desc[best_timing]})에 수요 급증이 예상됩니다. 기본 재고 대비 {safety_stock_pct:.1f}%의 충분한 안전재고를 확보하세요."
        safety_detail = "수요 변동성이 높아 예측 대비 실제 판매량 차이가 클 수 있습니다. 충분한 안전재고로 품절 리스크를 최소화하세요."
    else:
        safety_volatility = "매우 높음"
        safety_volatility_color = "#E31837"
        safety_msg = f"{timing_korean[best_timing]}({timing_desc[best_timing]}) 시점에 수요 변동이 매우 큽니다. {safety_stock_pct:.1f}%의 안전재고를 준비하세요."
        safety_detail = "수요 예측 불확실성이 매우 높습니다. 대량의 안전재고를 확보하여 품절에 대비하세요."
    
    sorted_preds = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)
    pred_rank = [i for i, (idx, _) in enumerate(sorted_preds) if idx == best_timing_idx][0]
    
    show_balloons = (pred_rank == 0)
    
    if pred_rank == 0:
        strategy_title = "최적의 타이밍입니다! 이익 극대화를 노리세요."
        strategy_detail = "이 시점은 기본 수요가 가장 높으면서 프로모션 효과도 최대입니다. 적극적인 재고 확보와 프로모션 진행을 권장합니다."
        strategy_actions = ["충분한 재고 확보", "적극적인 프로모션 진행"]
    elif pred_rank == 1:
        strategy_title = "숨은 기회입니다! 프로모션으로 판매량을 끌어올리세요."
        strategy_detail = "기본 수요는 중간 수준이지만, 프로모션 반응도가 높습니다. 프로모션을 통해 추가 매출을 확보할 수 있는 기회입니다."
        strategy_actions = ["타겟 프로모션 진행", "할인율 조정으로 마진 확보"]
    else:
        strategy_title = "효율적인 프로모션 시점입니다! 적은 투자로 높은 효과를 기대하세요."
        strategy_detail = "기본 판매량은 낮지만 Uplift 효과가 가장 큽니다. 프로모션 비용 대비 효율이 높아 ROI 극대화가 가능합니다."
        strategy_actions = ["테스트 프로모션 진행", "효율적인 예산 분배", "신규 고객 유입 기회로 활용"]
    
    strategy_message = {
        'title': strategy_title,
        'detail': strategy_detail,
        'actions': strategy_actions,
        'best_timing': timing_korean[best_timing],
        'best_uplift': best_uplift,
        'best_pred': best_pred,
        'best_with_promo': best_with_promo,
        'best_increase': best_increase,
        'safety_stock_pct': safety_stock_pct,
        'safety_stock_qty': safety_stock_qty,
        'safety_volatility': safety_volatility,
        'safety_volatility_color': safety_volatility_color,
        'safety_msg': safety_msg,
        'safety_detail': safety_detail,
        'total_recommended_qty': total_recommended_qty
    }
    
    return best_timing, best_uplift, strategy_message, show_balloons