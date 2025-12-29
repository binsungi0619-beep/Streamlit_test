"""
설정 파일 - 클러스터별 모델 매핑, 컬러, 날짜 설정 등
"""

# ============================================================
# 클러스터별 선택된 모델 매핑
# ============================================================
CLUSTER_MODEL_MAP = {
    ("Quito", 0): "XGBoost",
    ("Quito", 1): "RandomForest",
    ("Quito", 2): "RandomForest",
    ("Sierra", 0): "RandomForest",
    ("Sierra", 1): "XGBoost",
    ("Costa", 0): "XGBoost",
    ("Costa", 1): "XGBoost",
    ("Costa", 2): "CatBoost",
}

# 클러스터 키 문자열 버전 (Holiday용)
CLUSTER_BEST_MODEL = {
    'quito_0': 'XGBoost',
    'quito_1': 'RandomForest',
    'quito_2': 'RandomForest',
    'sierra_0': 'RandomForest',
    'sierra_1': 'XGBoost',
    'costa_0': 'XGBoost',
    'costa_1': 'XGBoost',
    'costa_2': 'CatBoost',
}

# ============================================================
# 클러스터 라벨 매핑 (한글)
# ============================================================
CLUSTER_LABELS = {
    ("Quito", 0): "소형",
    ("Quito", 1): "중형",
    ("Quito", 2): "대형",
    ("Sierra", 0): "소형",
    ("Sierra", 1): "중형",
    ("Costa", 0): "중형",
    ("Costa", 1): "소형",
    ("Costa", 2): "대형",
}

# 문자열 키 버전 (Holiday용)
CLUSTER_LABELS_STR = {
    'Quito_0': '소형',
    'Quito_1': '중형',
    'Quito_2': '대형',
    'Sierra_0': '소형',
    'Sierra_1': '중형',
    'Costa_0': '중형',
    'Costa_1': '소형',
    'Costa_2': '대형',
}

# 라벨 → 클러스터 번호 역매핑
LABEL_TO_CLUSTER = {
    "Quito": {"소형": 0, "중형": 1, "대형": 2},
    "Sierra": {"소형": 0, "중형": 1},
    "Costa": {"중형": 0, "소형": 1, "대형": 2},
}

# ============================================================
# 지역별 클러스터 목록
# ============================================================
REGION_CLUSTERS = {
    "Quito": [0, 1, 2],
    "Sierra": [0, 1],
    "Costa": [0, 1, 2],
}

# 지역별 라벨 목록 (대형 → 소형 순서)
REGION_CLUSTER_LABELS = {
    "Quito": ["대형매장", "중형매장", "소형매장"],
    "Sierra": ["중형매장", "소형매장"],
    "Costa": ["대형매장", "중형매장", "소형매장"],
}

# 지역별 라벨 → 클러스터 번호 매핑
REGION_LABEL_TO_CLUSTER = {
    "Quito": {"소형매장": 0, "중형매장": 1, "대형매장": 2},
    "Sierra": {"소형매장": 0, "중형매장": 1},
    "Costa": {"중형매장": 0, "소형매장": 1, "대형매장": 2},
}

# 지역별 클러스터 표시 순서 (대형 → 소형)
REGION_CLUSTER_ORDER = {
    "Quito": [2, 1, 0],      # 대형(2) → 중형(1) → 소형(0)
    "Sierra": [1, 0],         # 중형(1) → 소형(0)
    "Costa": [2, 0, 1],       # 대형(2) → 중형(0) → 소형(1)
}

# ============================================================
# 클러스터 컬러
# ============================================================
CLUSTER_COLORS = {
    0: '#E31837',  # Red - 빨강
    1: '#4A90E2',  # Blue - 파랑
    2: '#2D2D2D',  # Dark Gray - 진한 회색
}

CLUSTER_COLORS_STR = {
    'Quito_0': '#FF6B6B',
    'Quito_1': '#4ECDC4',
    'Quito_2': '#45B7D1',
    'Sierra_0': '#FF6B6B',
    'Sierra_1': '#4ECDC4',
    'Costa_0': '#4ECDC4',
    'Costa_1': '#FF6B6B',
    'Costa_2': '#45B7D1',
}

# ============================================================
# 브랜드 컬러
# ============================================================
COLORS = {
    'primary': '#E31837',
    'primary_dark': '#B71430',
    'secondary': '#2D2D2D',
    'text': '#333333',
    'light_gray': '#F5F5F5',
    'white': '#FFFFFF',
    'accent1': '#E31837',
    'accent2': '#666666',
    'accent3': '#999999',
    'train_line': '#999999',
    'test_line': '#E31837',
}

# 차트 컬러 팔레트
CHART_COLORS = ['#E31837', '#2D2D2D', '#666666', '#999999', '#CCCCCC']
REGION_COLORS = {'Quito': '#E31837', 'Sierra': '#2D2D2D', 'Costa': '#666666'}

# ============================================================
# 축제 정보
# ============================================================
HOLIDAYS = {
    'Navidad': {'date': '12-25', 'name': '크리스마스', 'is_closed': True},
    'Dia de Difuntos': {'date': '11-02', 'name': '망자의 날', 'is_closed': False},
    'Independencia de Cuenca': {'date': '11-03', 'name': '쿠엥카 독립기념일', 'is_closed': False},
}

# ============================================================
# 날짜 설정
# ============================================================
SPLIT_DATE = '2016-08-15'
TRAIN_END_DATE = '2016-08-15'
TEST_START_DATE = '2016-08-16'
TEST_END_DATE = '2017-08-15'
