"""
Snowflake 연결 유틸리티 (통합 버전)
- General, Total, Base, Holiday 모든 페이지에서 사용
"""
import streamlit as st
import polars as pl
from snowflake.connector import connect
import os

# ============================================================
# Snowflake 연결
# ============================================================
@st.cache_resource
def get_snowflake_connection():
    """Snowflake 연결 (캐싱)"""
    return connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
    )


def execute_query(query: str) -> pl.DataFrame:
    """쿼리 실행 후 Polars DataFrame 반환"""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    cursor.close()
    return pl.DataFrame(data, schema=columns, orient="row")


def execute_query_lowercase(query: str) -> pl.DataFrame:
    """쿼리 실행 후 Polars DataFrame 반환 (컬럼명 소문자)"""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0].lower() for desc in cursor.description]
    data = cursor.fetchall()
    cursor.close()
    return pl.DataFrame(data, schema=columns, orient="row")


# ============================================================
# GENERAL 대시보드용 데이터 로드 (집계된 데이터만)
# ============================================================
@st.cache_data(ttl=3600)
def load_general_kpi_filtered(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """필터링된 KPI 데이터"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12],
         '1분기': [1,2,3], '2분기': [4,5,6], '3분기': [7,8,9], '4분기': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        SUM(t.UNIT_SALES) as TOTAL_SALES,
        COUNT(DISTINCT t.STORE_NBR) as STORE_COUNT,
        COUNT(DISTINCT i.FAMILY) as FAMILY_COUNT
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_general_transactions_kpi(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """거래건수 KPI"""
    conditions = ["DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"YEAR(DATE) = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"MONTH(DATE) = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"MONTH(DATE) IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT SUM(TRANSACTIONS) as TOTAL_TRANSACTIONS
    FROM TRANSACTIONS_PREPROCESSED
    WHERE {where_clause}
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_daily_sales(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """일별 판매량 (차트용)"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        t.DATE,
        SUM(t.UNIT_SALES) as TOTAL_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    WHERE {where_clause}
    GROUP BY t.DATE
    ORDER BY t.DATE
    """
    df = execute_query(query)
    return df.with_columns(pl.col("DATE").cast(pl.Date))


@st.cache_data(ttl=3600)
def load_daily_transactions(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """일별 거래건수 (차트용)"""
    conditions = ["DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"YEAR(DATE) = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"MONTH(DATE) = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"MONTH(DATE) IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        DATE,
        SUM(TRANSACTIONS) as TOTAL_TRANSACTIONS
    FROM TRANSACTIONS_PREPROCESSED
    WHERE {where_clause}
    GROUP BY DATE
    ORDER BY DATE
    """
    df = execute_query(query)
    return df.with_columns(pl.col("DATE").cast(pl.Date))


@st.cache_data(ttl=3600)
def load_region_sales(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """지역별 판매량"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        c.REGION_LABEL,
        SUM(t.UNIT_SALES) as TOTAL_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN CLUSTERING_RESULTS_STORE_KMEANS c ON t.STORE_NBR = c.STORE_NBR
    WHERE {where_clause}
    GROUP BY c.REGION_LABEL
    ORDER BY TOTAL_SALES DESC
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_region_transactions(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """지역별 거래건수"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"MONTH(t.DATE) = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"MONTH(t.DATE) IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        c.REGION_LABEL,
        SUM(t.TRANSACTIONS) as TOTAL_TRANSACTIONS
    FROM TRANSACTIONS_PREPROCESSED t
    JOIN CLUSTERING_RESULTS_STORE_KMEANS c ON t.STORE_NBR = c.STORE_NBR
    WHERE {where_clause}
    GROUP BY c.REGION_LABEL
    ORDER BY TOTAL_TRANSACTIONS DESC
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_store_sales(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """매장별 판매량"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        t.STORE_NBR,
        SUM(t.UNIT_SALES) as TOTAL_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    WHERE {where_clause}
    GROUP BY t.STORE_NBR
    ORDER BY TOTAL_SALES DESC
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_family_sales(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """Family별 판매량"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        i.FAMILY,
        SUM(t.UNIT_SALES) as TOTAL_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    GROUP BY i.FAMILY
    ORDER BY TOTAL_SALES DESC
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_store_promo_ratio(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """매장별 프로모션 비율"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT 
        t.STORE_NBR,
        SUM(CASE WHEN t.ONPROMOTION = 1 THEN 1 ELSE 0 END) as PROMO_COUNT,
        COUNT(*) as TOTAL_COUNT,
        (SUM(CASE WHEN t.ONPROMOTION = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as PROMO_RATIO
    FROM TRAIN_PREPROCESSED_VER2 t
    WHERE {where_clause}
    GROUP BY t.STORE_NBR
    ORDER BY PROMO_RATIO DESC
    LIMIT 5
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_family_promo_uplift(year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """Family별 프로모션 Uplift"""
    conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        conditions.append(f"t.YEAR = {year_filter}")
    
    if month_filter != '전체':
        month_num = int(month_filter.replace('월', ''))
        conditions.append(f"t.MONTH = {month_num}")
    elif quarter_filter != '전체':
        q_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = q_map.get(quarter_filter, [])
        if months:
            conditions.append(f"t.MONTH IN ({','.join(map(str, months))})")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    WITH promo_sales AS (
        SELECT 
            i.FAMILY,
            AVG(t.UNIT_SALES) as PROMO_AVG
        FROM TRAIN_PREPROCESSED_VER2 t
        JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
        WHERE {where_clause} AND t.ONPROMOTION = 1
        GROUP BY i.FAMILY
    ),
    no_promo_sales AS (
        SELECT 
            i.FAMILY,
            AVG(t.UNIT_SALES) as NO_PROMO_AVG
        FROM TRAIN_PREPROCESSED_VER2 t
        JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
        WHERE {where_clause} AND t.ONPROMOTION = 0
        GROUP BY i.FAMILY
    )
    SELECT 
        p.FAMILY,
        p.PROMO_AVG,
        n.NO_PROMO_AVG,
        (p.PROMO_AVG / NULLIF(n.NO_PROMO_AVG, 0)) as UPLIFT
    FROM promo_sales p
    JOIN no_promo_sales n ON p.FAMILY = n.FAMILY
    WHERE n.NO_PROMO_AVG > 0
    ORDER BY UPLIFT DESC
    LIMIT 5
    """
    return execute_query(query)


@st.cache_data(ttl=3600)
def load_stores_count():
    """총 매장 수"""
    query = "SELECT COUNT(*) as CNT FROM STORES_PREPROCESSED"
    return execute_query(query)


# ============================================================
# 클러스터링 결과 로드
# ============================================================
@st.cache_data(ttl=3600)
def load_clustering_results(region_label: str = None) -> pl.DataFrame:
    """클러스터링 결과 로드"""
    if region_label:
        query = f"""
        SELECT 
            STORE_NBR,
            REGION_LABEL,
            CLUSTER,
            AVG_MONTHLY_SALES,
            AVG_MONTHLY_SKU,
            PERISHABLE_PCT,
            AVG_MONTHLY_TRANSACTIONS,
            SALES_CV,
            UPT
        FROM CLUSTERING_RESULTS_STORE_KMEANS
        WHERE REGION_LABEL = '{region_label}'
        ORDER BY CLUSTER DESC, STORE_NBR
        """
    else:
        query = """
        SELECT 
            STORE_NBR,
            REGION_LABEL,
            CLUSTER,
            AVG_MONTHLY_SALES,
            AVG_MONTHLY_SKU,
            PERISHABLE_PCT,
            AVG_MONTHLY_TRANSACTIONS,
            SALES_CV,
            UPT
        FROM CLUSTERING_RESULTS_STORE_KMEANS
        """
    return execute_query_lowercase(query)


# ============================================================
# BASE 탭용 데이터 로드
# ============================================================
@st.cache_data(ttl=3600)
def load_train_data_with_cluster(region_label: str, cluster: int) -> pl.DataFrame:
    """Train 데이터 로드 (클러스터 조인 + Family 조인)"""
    query = f"""
    SELECT 
        t.DATE,
        i.FAMILY,
        SUM(t.UNIT_SALES) as TOTAL_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN CLUSTERING_RESULTS_STORE_KMEANS c ON t.STORE_NBR = c.STORE_NBR
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE c.REGION_LABEL = '{region_label}'
      AND c.CLUSTER = {cluster}
      AND t.DATE <= '2016-08-15'
    GROUP BY t.DATE, i.FAMILY
    ORDER BY t.DATE, i.FAMILY
    """
    df = execute_query(query)
    return df.with_columns(pl.col("DATE").cast(pl.Date))


@st.cache_data(ttl=3600)
def load_predictions(region_label: str, cluster: int, model_type: str) -> pl.DataFrame:
    """예측 데이터 로드"""
    table_name = f"SALES_FORECAST_PREDICTIONS_DETAIL_{region_label.upper()}_{cluster}"
    
    query = f"""
    SELECT 
        DATE,
        FAMILY,
        AVG(Y_PRED) as PRED_SALES
    FROM {table_name}
    WHERE MODEL_TYPE = '{model_type}'
    GROUP BY DATE, FAMILY
    ORDER BY DATE, FAMILY
    """
    df = execute_query(query)
    return df.with_columns(pl.col("DATE").cast(pl.Date))


@st.cache_data(ttl=3600)
def load_train_total_sales(region_label: str, cluster: int) -> pl.DataFrame:
    """Train 기간 Family별 총 판매량 (TOP/BOTTOM 계산용)"""
    query = f"""
    SELECT 
        i.FAMILY,
        SUM(t.UNIT_SALES) as TOTAL_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN CLUSTERING_RESULTS_STORE_KMEANS c ON t.STORE_NBR = c.STORE_NBR
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE c.REGION_LABEL = '{region_label}'
      AND c.CLUSTER = {cluster}
      AND t.DATE <= '2016-08-15'
    GROUP BY i.FAMILY
    ORDER BY TOTAL_SALES DESC
    """
    return execute_query(query)


# ============================================================
# TOTAL 탭용 데이터 로드 (최적화 버전)
# ============================================================
@st.cache_data(ttl=3600)
def load_train_data_for_total(store_list, year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """Total 탭용 Train 데이터 로드 (Snowflake에서 미리 집계)"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        date_conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    if month_filter != '전체':
        m_num = month_filter.replace('월', '')
        date_conditions.append(f"MONTH(t.DATE) = {m_num}")
    elif quarter_filter != '전체':
        q_num = quarter_filter.replace('분기', '')
        date_conditions.append(f"QUARTER(t.DATE) = {q_num}")
    
    where_clause = f"t.STORE_NBR IN ({store_str}) AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        t.DATE as date,
        i.FAMILY as family,
        i.PERISHABLE as perishable,
        SUM(t.UNIT_SALES) as unit_sales
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    GROUP BY t.DATE, i.FAMILY, i.PERISHABLE
    ORDER BY t.DATE
    """
    
    df = execute_query_lowercase(query)
    if 'date' in df.columns:
        df = df.with_columns(pl.col('date').cast(pl.Date))
    return df


@st.cache_data(ttl=3600)
def load_daily_sales_for_total(store_list, year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """Total 탭용 일별 판매량 (서버에서 집계)"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        date_conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    if month_filter != '전체':
        m_num = month_filter.replace('월', '')
        date_conditions.append(f"MONTH(t.DATE) = {m_num}")
    elif quarter_filter != '전체':
        q_num = quarter_filter.replace('분기', '')
        date_conditions.append(f"QUARTER(t.DATE) = {q_num}")
    
    where_clause = f"t.STORE_NBR IN ({store_str}) AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        t.DATE,
        SUM(t.UNIT_SALES) as UNIT_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    WHERE {where_clause}
    GROUP BY t.DATE
    ORDER BY t.DATE
    """
    
    df = execute_query_lowercase(query)
    if 'date' in df.columns:
        df = df.with_columns(pl.col('date').cast(pl.Date))
    return df


@st.cache_data(ttl=3600)
def load_family_sales_for_total(store_list, year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """Total 탭용 카테고리별 판매량 (서버에서 집계)"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        date_conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    if month_filter != '전체':
        m_num = month_filter.replace('월', '')
        date_conditions.append(f"MONTH(t.DATE) = {m_num}")
    elif quarter_filter != '전체':
        q_num = quarter_filter.replace('분기', '')
        date_conditions.append(f"QUARTER(t.DATE) = {q_num}")
    
    where_clause = f"t.STORE_NBR IN ({store_str}) AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        i.FAMILY,
        i.PERISHABLE,
        SUM(t.UNIT_SALES) as UNIT_SALES
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    GROUP BY i.FAMILY, i.PERISHABLE
    ORDER BY UNIT_SALES DESC
    """
    
    return execute_query_lowercase(query)


@st.cache_data(ttl=3600)
def load_transactions_for_total(store_list, year_filter='전체', quarter_filter='전체', month_filter='전체'):
    """Total 탭용 Transactions 데이터 로드 (서버에서 집계)"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        date_conditions.append(f"YEAR(DATE) = {year_filter}")
    
    if month_filter != '전체':
        m_num = month_filter.replace('월', '')
        date_conditions.append(f"MONTH(DATE) = {m_num}")
    elif quarter_filter != '전체':
        q_num = quarter_filter.replace('분기', '')
        date_conditions.append(f"QUARTER(DATE) = {q_num}")
    
    where_clause = f"STORE_NBR IN ({store_str}) AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        DATE,
        SUM(TRANSACTIONS) as TRANSACTIONS
    FROM TRANSACTIONS_PREPROCESSED
    WHERE {where_clause}
    GROUP BY DATE
    ORDER BY DATE
    """
    
    df = execute_query_lowercase(query)
    if 'date' in df.columns:
        df = df.with_columns(pl.col('date').cast(pl.Date))
    return df


# ============================================================
# HOLIDAY 탭용 데이터 로드
# ============================================================
@st.cache_data(ttl=3600)
def load_train_data_for_holiday(store_list, year_filter='전체'):
    """Holiday 탭용 Train 데이터 로드"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["t.DATE <= '2016-08-15'"]
    
    if year_filter != '전체':
        date_conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    where_clause = f"t.STORE_NBR IN ({store_str})"
    if date_conditions:
        where_clause += " AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        t.DATE,
        t.STORE_NBR,
        t.UNIT_SALES,
        i.FAMILY
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    """
    
    df = execute_query_lowercase(query)
    if 'date' in df.columns:
        df = df.with_columns(pl.col('date').cast(pl.Date))
    return df


@st.cache_data(ttl=3600)
def load_promotion_timing_predictions(region_label: str, cluster_filter=None, holiday_filter=None, family_filter=None):
    """프로모션 타이밍 예측 데이터 로드"""
    conditions = [f"FINAL_CLUSTER LIKE '{region_label}%'"]
    
    if cluster_filter and cluster_filter != '전체':
        conditions.append(f"FINAL_CLUSTER = '{cluster_filter}'")
    
    if holiday_filter and holiday_filter != '전체':
        conditions.append(f"HOLIDAY_NAME = '{holiday_filter}'")
    
    if family_filter and family_filter != '전체':
        conditions.append(f"FAMILY = '{family_filter}'")
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
    SELECT *
    FROM PROMOTION_TIMING_PREDICTIONS
    WHERE {where_clause}
    """
    
    return execute_query_lowercase(query)


@st.cache_data(ttl=3600)
def load_sales_forecast_predictions(cluster_name: str, best_model: str):
    """판매 예측 데이터 로드 (클러스터별)"""
    table_name = f"SALES_FORECAST_PREDICTIONS_DETAIL_{cluster_name.upper()}"
    
    query = f"""
    SELECT 
        DATE,
        FAMILY,
        FOLD,
        Y_PRED
    FROM {table_name}
    WHERE MODEL_TYPE = '{best_model}'
    """
    
    try:
        df = execute_query_lowercase(query)
        if 'date' in df.columns:
            df = df.with_columns(pl.col('date').cast(pl.Date))
        
        df = df.group_by(['date', 'family']).agg(
            pl.col('y_pred').mean().alias('y_pred_avg')
        )
        return df
    except Exception as e:
        return pl.DataFrame()


@st.cache_data(ttl=3600)
def get_family_list(region_label: str):
    """Family 목록 조회"""
    query = f"""
    SELECT DISTINCT FAMILY
    FROM PROMOTION_TIMING_PREDICTIONS
    WHERE FINAL_CLUSTER LIKE '{region_label}%'
    ORDER BY FAMILY
    """
    
    df = execute_query(query)
    return df['FAMILY'].to_list()


# ============================================================
# HOLIDAY 탭 최적화 함수
# ============================================================
@st.cache_data(ttl=3600)
def load_train_data_for_holiday_aggregated(store_list, year_filter='전체'):
    """Holiday 탭용 Train 데이터 - 서버에서 집계"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["t.DATE <= '2016-08-15'"]
    if year_filter != '전체':
        date_conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    where_clause = f"t.STORE_NBR IN ({store_str}) AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        i.FAMILY,
        SUM(t.UNIT_SALES) as total_sales
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    GROUP BY i.FAMILY
    ORDER BY total_sales DESC
    """
    return execute_query_lowercase(query)


@st.cache_data(ttl=3600)
def load_train_timing_sales(store_list, year_filter='전체', holiday_month=None, holiday_day=None):
    """Holiday 탭용 - Before/During/After 판매량을 서버에서 집계"""
    store_str = ','.join([str(s) for s in store_list])
    
    date_conditions = ["t.DATE <= '2016-08-15'"]
    if year_filter != '전체':
        date_conditions.append(f"YEAR(t.DATE) = {year_filter}")
    
    # 해당 월 데이터만 필터
    if holiday_month:
        date_conditions.append(f"MONTH(t.DATE) = {holiday_month}")
    
    where_clause = f"t.STORE_NBR IN ({store_str}) AND " + " AND ".join(date_conditions)
    
    query = f"""
    SELECT 
        CASE 
            WHEN DAY(t.DATE) = {holiday_day} THEN 'During'
            WHEN DAY(t.DATE) < {holiday_day} THEN 'Before'
            WHEN DAY(t.DATE) > {holiday_day} THEN 'After'
        END as timing,
        i.FAMILY,
        SUM(t.UNIT_SALES) as total_sales
    FROM TRAIN_PREPROCESSED_VER2 t
    JOIN ITEMS_PREPROCESSED i ON t.ITEM_NBR = i.ITEM_NBR
    WHERE {where_clause}
    GROUP BY timing, i.FAMILY
    """
    return execute_query_lowercase(query)


@st.cache_data(ttl=3600)
def load_all_forecast_predictions(region_label: str):
    """전체 클러스터의 예측 데이터를 한 번에 로드"""
    from utils.config import REGION_CLUSTER_ORDER, CLUSTER_BEST_MODEL
    
    all_dfs = []
    for cluster_num in REGION_CLUSTER_ORDER.get(region_label, []):
        cluster_key = f"{region_label}_{cluster_num}"
        best_model = CLUSTER_BEST_MODEL.get(cluster_key.lower(), 'XGBoost')
        table_name = f"SALES_FORECAST_PREDICTIONS_DETAIL_{cluster_key.upper()}"
        
        query = f"""
        SELECT 
            DATE,
            FAMILY,
            AVG(Y_PRED) as PRED_SALES
        FROM {table_name}
        WHERE MODEL_TYPE = '{best_model}'
        GROUP BY DATE, FAMILY
        """
        
        try:
            df = execute_query_lowercase(query)
            if df.height > 0:
                if 'date' in df.columns:
                    df = df.with_columns(pl.col('date').cast(pl.Date))
                all_dfs.append(df)
        except:
            pass
    
    if all_dfs:
        return pl.concat(all_dfs)
    return pl.DataFrame()
