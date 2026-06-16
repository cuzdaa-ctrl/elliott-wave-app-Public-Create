import sys, subprocess
try:
    import pkg_resources
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "setuptools"], check=True)
    import pkg_resources

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date
import sqlite3
import io
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="엘리엇 파동 분석기 + 매매일지", layout="wide", page_icon="📈")

# ─── SQLite DB ───────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "journal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            market_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            investment REAL NOT NULL,
            reason TEXT NOT NULL,
            memo TEXT,
            stop_loss REAL NOT NULL,
            target_price REAL,
            status TEXT DEFAULT '보유중',
            exit_price REAL,
            exit_reason TEXT,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            stop_respected INTEGER,
            review_memo TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_conn():
    return sqlite3.connect(DB_PATH)

def load_trades(status=None):
    conn = get_conn()
    if status:
        df = pd.read_sql("SELECT * FROM trades WHERE status=? ORDER BY trade_date DESC, id DESC",
                         conn, params=(status,))
    else:
        df = pd.read_sql("SELECT * FROM trades ORDER BY trade_date DESC, id DESC", conn)
    conn.close()
    return df

def save_trade(data: dict):
    conn = get_conn()
    cols = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    conn.execute(f"INSERT INTO trades ({cols}) VALUES ({placeholders})", list(data.values()))
    conn.commit()
    conn.close()

def close_trade(trade_id, exit_price, exit_reason, stop_respected, review_memo=""):
    conn = get_conn()
    row = conn.execute("SELECT entry_price, quantity FROM trades WHERE id=?", (trade_id,)).fetchone()
    if row:
        entry_price, quantity = row
        realized_pnl = (exit_price - entry_price) * quantity
        realized_pnl_pct = (exit_price - entry_price) / entry_price * 100
        conn.execute("""
            UPDATE trades SET status='종료', exit_price=?, exit_reason=?,
            realized_pnl=?, realized_pnl_pct=?, stop_respected=?, review_memo=?
            WHERE id=?
        """, (exit_price, exit_reason, realized_pnl, realized_pnl_pct,
              stop_respected, review_memo, trade_id))
        conn.commit()
    conn.close()

def update_review(trade_id, review_memo):
    conn = get_conn()
    conn.execute("UPDATE trades SET review_memo=? WHERE id=?", (review_memo, trade_id))
    conn.commit()
    conn.close()

# ─── 데이터 로드 ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_stock_list():
    try:
        krx = fdr.StockListing('KRX')
        krx = krx[['Code', 'Name']].dropna()
        krx['Code'] = krx['Code'].astype(str).str.zfill(6)
        return krx
    except Exception:
        return pd.DataFrame(columns=['Code', 'Name'])

@st.cache_data(ttl=86400)
def load_crypto_list():
    return {
        'BTC/KRW': '비트코인', 'ETH/KRW': '이더리움', 'XRP/KRW': '리플',
        'ADA/KRW': '에이다', 'SOL/KRW': '솔라나', 'DOGE/KRW': '도지코인',
        'DOT/KRW': '폴카닷', 'AVAX/KRW': '아발란체', 'MATIC/KRW': '폴리곤',
        'LINK/KRW': '체인링크',
    }

def find_stock_code(query, stock_df):
    query = query.strip()
    if query.isdigit() and len(query) == 6:
        row = stock_df[stock_df['Code'] == query]
        return (query, row.iloc[0]['Name']) if not row.empty else (query, query)
    exact = stock_df[stock_df['Name'] == query]
    if not exact.empty:
        return exact.iloc[0]['Code'], exact.iloc[0]['Name']
    partial = stock_df[stock_df['Name'].str.contains(query, na=False)]
    if not partial.empty:
        return partial.iloc[0]['Code'], partial.iloc[0]['Name']
    return query, query

@st.cache_data(ttl=600)
def load_price_data(code, start, end):
    # 한국 주식(6자리 숫자)은 pykrx 사용 — Streamlit Cloud에서도 안정적
    if code.isdigit() and len(code) == 6:
        # 한국 주식 — pykrx (KRX 직접 접근)
        from pykrx import stock as krx
        s = start.replace('-', '')
        e = end.replace('-', '')
        df = krx.get_market_ohlcv_by_date(s, e, code)
        df = df.rename(columns={'시가': 'Open', '고가': 'High', '저가': 'Low',
                                 '종가': 'Close', '거래량': 'Volume'})
        df.index = pd.to_datetime(df.index)
    elif '/' not in code:
        # 미국 주식 — FinanceDataReader (Yahoo Finance 경유)
        df = fdr.DataReader(code.upper(), start, end)
        df.index = pd.to_datetime(df.index)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})
    else:
        # 코인 — FinanceDataReader
        df = fdr.DataReader(code, start, end)
        df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=['Close'])
    return df

# ─── 기술적 지표 계산 ─────────────────────────────────────────────────────────
def calc_bb(close, period=20, std_dev=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

def calc_ichimoku(df, t=9, k=26, s=52):
    tenkan = (df['High'].rolling(t).max() + df['Low'].rolling(t).min()) / 2
    kijun  = (df['High'].rolling(k).max() + df['Low'].rolling(k).min()) / 2
    spanA  = ((tenkan + kijun) / 2).shift(k)
    spanB  = ((df['High'].rolling(s).max() + df['Low'].rolling(s).min()) / 2).shift(k)
    chikou = df['Close'].shift(-k)
    return tenkan, kijun, spanA, spanB, chikou

def calc_ma(close, period, kind='EMA'):
    if kind == 'EMA':
        return close.ewm(span=period, adjust=False).mean()
    return close.rolling(period).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ─── ZigZag ──────────────────────────────────────────────────────────────────
def zigzag(prices, threshold_pct):
    arr = prices.values
    pivots, last_pivot_idx, last_pivot_val, direction = [], 0, arr[0], None
    for i in range(1, len(arr)):
        change = (arr[i] - last_pivot_val) / last_pivot_val * 100
        if direction is None:
            if change >= threshold_pct:
                pivots.append({'idx': last_pivot_idx, 'price': last_pivot_val, 'type': 'L'})
                direction = 'up'; last_pivot_idx = i; last_pivot_val = arr[i]
            elif change <= -threshold_pct:
                pivots.append({'idx': last_pivot_idx, 'price': last_pivot_val, 'type': 'H'})
                direction = 'down'; last_pivot_idx = i; last_pivot_val = arr[i]
        elif direction == 'up':
            if arr[i] > last_pivot_val:
                last_pivot_idx = i; last_pivot_val = arr[i]
            elif (last_pivot_val - arr[i]) / last_pivot_val * 100 >= threshold_pct:
                pivots.append({'idx': last_pivot_idx, 'price': last_pivot_val, 'type': 'H'})
                direction = 'down'; last_pivot_idx = i; last_pivot_val = arr[i]
        else:
            if arr[i] < last_pivot_val:
                last_pivot_idx = i; last_pivot_val = arr[i]
            elif (arr[i] - last_pivot_val) / last_pivot_val * 100 >= threshold_pct:
                pivots.append({'idx': last_pivot_idx, 'price': last_pivot_val, 'type': 'L'})
                direction = 'up'; last_pivot_idx = i; last_pivot_val = arr[i]
    if not pivots or pivots[-1]['idx'] != last_pivot_idx:
        pivots.append({'idx': last_pivot_idx, 'price': last_pivot_val,
                       'type': 'H' if direction == 'up' else 'L'})
    return pivots

# ─── 엘리엇 파동 ─────────────────────────────────────────────────────────────
def count_elliott_waves(pivots):
    if len(pivots) < 3:
        return None
    lows = [p for p in pivots if p['type'] == 'L']
    if not lows:
        return None
    pt0 = min(lows, key=lambda x: x['price'])
    after = [p for p in pivots if p['idx'] > pt0['idx']]
    if len(after) < 2:
        return None
    wave_pts = {0: pt0}
    wave_num = 1
    for p in after:
        if wave_num > 4:
            break
        if p['type'] == ('H' if wave_num % 2 == 1 else 'L'):
            wave_pts[wave_num] = p
            wave_num += 1
    if 3 not in wave_pts:
        return None

    validations = []
    if 2 in wave_pts:
        validations.append(("(1) 2파 > 0점", wave_pts[2]['price'] > wave_pts[0]['price']))
    else:
        validations.append(("(1) 2파 > 0점", None))

    w1 = abs(wave_pts[1]['price'] - wave_pts[0]['price']) if 1 in wave_pts else None
    w3 = abs(wave_pts[3]['price'] - wave_pts[2]['price']) if (2 in wave_pts and 3 in wave_pts) else None
    w5 = abs(wave_pts.get(5, {}).get('price', np.nan) - wave_pts.get(4, {}).get('price', np.nan)) if 4 in wave_pts else None
    if w3 is not None and w1 is not None:
        lengths = [l for l in [w1, w3, w5] if l is not None and not np.isnan(l)]
        validations.append(("(2) 3파 ≠ 최단파", w3 != min(lengths) if lengths else True))
    else:
        validations.append(("(2) 3파 ≠ 최단파", None))

    if 4 in wave_pts and 1 in wave_pts:
        validations.append(("(3) 4파 > 1파 고점", wave_pts[4]['price'] > wave_pts[1]['price']))
    else:
        validations.append(("(3) 4파 > 1파 고점", None))

    wave1_len = abs(wave_pts[1]['price'] - wave_pts[0]['price']) if 1 in wave_pts else None
    fib_targets = {}
    if wave1_len and 2 in wave_pts:
        b = wave_pts[2]['price']
        fib_targets['3파_1.618'] = b + wave1_len * 1.618
        fib_targets['3파_2.618'] = b + wave1_len * 2.618
    if wave1_len and 4 in wave_pts:
        b = wave_pts[4]['price']
        fib_targets['5파_0.618'] = b + wave1_len * 0.618
        fib_targets['5파_1.000'] = b + wave1_len * 1.000
        fib_targets['5파_1.618'] = b + wave1_len * 1.618

    return {'wave_points': wave_pts, 'validations': validations,
            'fib_targets': fib_targets, 'wave1_len': wave1_len}

def compute_trendline(wave_pts, df):
    if 2 not in wave_pts or 4 not in wave_pts:
        return None
    p2, p4 = wave_pts[2], wave_pts[4]
    slope = (p4['price'] - p2['price']) / max(p4['idx'] - p2['idx'], 1)
    best_pt = p2
    for i in range(p2['idx'], p4['idx'] + 1):
        line_val = p2['price'] + slope * (i - p2['idx'])
        if df['Low'].iloc[i] < line_val and df['Low'].iloc[i] < best_pt['price']:
            best_pt = {'idx': i, 'price': df['Low'].iloc[i], 'type': 'L'}
    if best_pt is not p2:
        slope = (p4['price'] - best_pt['price']) / max(p4['idx'] - best_pt['idx'], 1)
        anchor = best_pt
    else:
        anchor = p2
    return {'anchor': anchor, 'p4': p4, 'slope': slope}

def trendline_price_at(tl, x_idx):
    return tl['anchor']['price'] + tl['slope'] * (x_idx - tl['anchor']['idx'])

# ─── 코인게코 ID 매핑 ─────────────────────────────────────────────────────────
COINGECKO_IDS = {
    'BTC/KRW': 'bitcoin',    'ETH/KRW': 'ethereum',  'XRP/KRW': 'ripple',
    'ADA/KRW': 'cardano',    'SOL/KRW': 'solana',    'DOGE/KRW': 'dogecoin',
    'DOT/KRW': 'polkadot',   'AVAX/KRW': 'avalanche-2',
    'MATIC/KRW': 'matic-network', 'LINK/KRW': 'chainlink',
}

# DART 공시 유형 한글
DART_REPORT_LABELS = {
    '분기보고서': ('📊', '#44aaff'), '반기보고서': ('📊', '#44aaff'),
    '사업보고서': ('📊', '#44aaff'), '영업(잠정)실적': ('📊', '#44ffaa'),
    '유상증자': ('⚠️', '#ff8800'),  '무상증자': ('🎁', '#88ff44'),
    '자기주식': ('💰', '#ffcc44'),   '배당': ('💸', '#ffcc44'),
    '합병': ('🔀', '#ff8800'),       '분할': ('🔀', '#ff8800'),
    '소송': ('⚖️', '#ff4444'),      '제재': ('🚫', '#ff4444'),
    '횡령': ('🚨', '#ff2222'),       '부도': ('💥', '#ff2222'),
}

# ─── 인사이트 데이터 함수 ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def dart_get_corp_code(stock_code: str, api_key: str):
    try:
        r = requests.get("https://opendart.fss.or.kr/api/company.json",
                         params={'crtfc_key': api_key, 'stock_code': stock_code},
                         timeout=5)
        d = r.json()
        if d.get('status') == '000':
            return d.get('corp_code')
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800)
def dart_get_disclosures(corp_code: str, api_key: str, days: int = 60):
    end   = datetime.today().strftime('%Y%m%d')
    start = (datetime.today() - timedelta(days=days)).strftime('%Y%m%d')
    try:
        r = requests.get("https://opendart.fss.or.kr/api/list.json",
                         params={'crtfc_key': api_key, 'corp_code': corp_code,
                                 'bgn_de': start, 'end_de': end, 'page_count': 30},
                         timeout=6)
        d = r.json()
        if d.get('status') == '000':
            return d.get('list', [])
    except Exception:
        pass
    return []

@st.cache_data(ttl=1800)
def naver_get_earnings(stock_code: str):
    """네이버 금융 분기 실적 (최근 4개 분기)"""
    url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.encoding = 'euc-kr'
        from html.parser import HTMLParser
        import re
        # 실적 요약 테이블: id="_svp_consolidated_q" 또는 비연결
        text = r.text
        # 매출액, 영업이익, 순이익 블록 간단 파싱
        tables = re.findall(r'<table[^>]*id="[^"]*svp_[^"]*"[^>]*>.*?</table>', text, re.DOTALL)
        if not tables:
            return None
        rows_data = []
        for cell in re.findall(r'<td[^>]*>(.*?)</td>', tables[0], re.DOTALL):
            clean = re.sub('<[^>]+>', '', cell).strip().replace('\xa0', '').replace(',', '')
            rows_data.append(clean)
        # 구조: 날짜행 + 매출/영업이익/순이익 각 4분기
        return rows_data[:20] if rows_data else None
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_fear_greed():
    """암호화폐 공포·탐욕 지수 (alternative.me 무료)"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=5)
        data = r.json().get('data', [])
        return data  # [{'value': '72', 'value_classification': 'Greed', 'timestamp': ...}, ...]
    except Exception:
        return []

@st.cache_data(ttl=600)
def fetch_coingecko(coin_id: str):
    """코인게코 시장 데이터 (무료, 키 불필요)"""
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            params={'localization': 'false', 'tickers': 'false',
                    'community_data': 'false', 'developer_data': 'false'},
            timeout=8,
            headers={'Accept': 'application/json'},
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ─── 펀더멘털 데이터 (Naver 스크래핑 + yfinance) ─────────────────────────────

@st.cache_data(ttl=3600)
def fetch_fundamental(code: str, market: str = "한국 주식") -> dict:
    """
    한국 주식: PER/PBR (Naver 스크래핑) + 기관보유율 (yfinance)
    미국 주식: trailingPE / shortRatio / 기관보유율 (yfinance)
    """
    import re, yfinance as yf
    result = {}

    if market == "한국 주식" and code.isdigit() and len(code) == 6:
        h = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://finance.naver.com/',
            'Accept-Language': 'ko-KR,ko;q=0.9',
        }
        try:
            r = requests.get(
                f'https://finance.naver.com/item/main.naver?code={code}',
                headers=h, timeout=8
            )
            r.encoding = 'euc-kr'
            text = r.text
            per_m = re.search(r'PER[^0-9]*([0-9]+\.[0-9]+)', text)
            pbr_m = re.search(r'PBR[^0-9]*([0-9]+\.[0-9]+)', text)
            result['per'] = float(per_m.group(1)) if per_m else None
            result['pbr'] = float(pbr_m.group(1)) if pbr_m else None
        except Exception:
            result['per'] = None
            result['pbr'] = None

        try:
            t = yf.Ticker(f'{code}.KS')
            info = t.info
            val = info.get('heldPercentInstitutions')
            result['institutional_hold'] = float(val) if val is not None else None
        except Exception:
            result['institutional_hold'] = None

    elif market == "미국 주식":
        try:
            t = yf.Ticker(code.upper())
            info = t.info
            result['per'] = info.get('trailingPE') or info.get('forwardPE')
            result['pbr'] = info.get('priceToBook')
            result['institutional_hold'] = info.get('heldPercentInstitutions')
            result['short_ratio'] = info.get('shortRatio')  # days to cover
            result['dividend_yield'] = info.get('dividendYield')
        except Exception:
            result['per'] = None
            result['pbr'] = None
            result['institutional_hold'] = None
            result['short_ratio'] = None
            result['dividend_yield'] = None

    return result

# ─── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")

    market = st.radio("시장 선택", ["한국 주식", "미국 주식", "암호화폐"], horizontal=True)
    if market == "한국 주식":
        query = st.text_input("종목명 또는 코드", value="삼성전자",
                              placeholder="예: 삼성전자 또는 005930")
    elif market == "미국 주식":
        query = st.text_input("티커 심볼", value="AAPL",
                              placeholder="예: AAPL, TSLA, NVDA, MSFT").upper().strip()
    else:
        crypto_list = load_crypto_list()
        crypto_options = [f"{v} ({k})" for k, v in crypto_list.items()]
        selected_crypto = st.selectbox("코인 선택", crypto_options)
        query = list(crypto_list.keys())[crypto_options.index(selected_crypto)]

    period_map = {"3개월": 90, "6개월": 180, "1년": 365, "2년": 730}
    period_label = st.selectbox("조회 기간", list(period_map.keys()), index=2)
    days = period_map[period_label]
    threshold = st.slider("ZigZag 민감도 (%)", 2, 20, 7, 1)
    analyze_btn = st.button("🔍 분석 시작", use_container_width=True, type="primary")

    st.markdown("---")
    st.subheader("📐 보조지표")

    show_bb       = st.checkbox("볼린저 밴드 (BB 20)", value=True)
    show_ichimoku = st.checkbox("이치모쿠 구름", value=True)
    show_rsi      = st.checkbox("RSI (14)", value=True)
    show_macd     = st.checkbox("MACD (12/26/9)", value=False)

    st.markdown("**이동평균선**")
    ma_kind = st.radio("종류", ["EMA", "SMA"], horizontal=True)
    ma_options = [5, 20, 60, 120, 200]
    ma_selected = st.multiselect("기간 선택", ma_options, default=[20, 60, 120])

    MA_COLORS = {5: '#FFFF44', 20: '#FF8800', 60: '#FF44FF',
                 120: '#44FFFF', 200: '#FF4444'}

    st.markdown("---")
    st.subheader("🔑 API 키 설정")
    dart_api_key = st.text_input(
        "DART API 키 (한국 주식 공시)",
        type="password",
        placeholder="dart.fss.or.kr 에서 무료 발급",
        help="금융감독원 전자공시시스템(DART) API 키. 무료 발급: dart.fss.or.kr → OpenAPI → 인증키 신청",
    )
    st.caption("코인 인사이트는 키 없이 자동 조회됩니다.")

# ════════════════════════════════════════════════════════════════════════════
# 메인 탭
# ════════════════════════════════════════════════════════════════════════════
st.title("📈 엘리엇 파동 분석기 + 매매일지")
tab_wave, tab_signal, tab_journal = st.tabs(["📊 엘리엇 파동 분석", "🎯 종합 매매 판단", "📓 매매일지"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: 엘리엇 파동 분석
# ════════════════════════════════════════════════════════════════════════════
with tab_wave:
    if analyze_btn:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days)

        # 이치모쿠/BB가 충분한 데이터를 필요로 하므로 여유분 추가
        fetch_start = start_date - timedelta(days=100)

        with st.spinner("데이터 로드 중..."):
            try:
                if market == "한국 주식":
                    stock_df = load_stock_list()
                    code, name = find_stock_code(query, stock_df)
                    if not (code.isdigit() and len(code) == 6):
                        # pykrx로 직접 검색 시도
                        from pykrx import stock as krx
                        today_str = end_date.strftime('%Y%m%d')
                        all_codes = krx.get_market_ticker_list(today_str, market="ALL")
                        matched = [(c, krx.get_market_ticker_name(c)) for c in all_codes
                                   if query in krx.get_market_ticker_name(c)]
                        if matched:
                            code, name = matched[0]
                        else:
                            st.error(f"'{query}' 종목을 찾을 수 없습니다.\n\n"
                                     "6자리 종목코드를 직접 입력해보세요. (예: 059090)")
                            st.stop()
                    df_full = load_price_data(code, fetch_start.strftime('%Y-%m-%d'),
                                             end_date.strftime('%Y-%m-%d'))
                    title = f"{name} ({code})"
                elif market == "미국 주식":
                    code = query.upper().strip()
                    df_full = load_price_data(code, fetch_start.strftime('%Y-%m-%d'),
                                             end_date.strftime('%Y-%m-%d'))
                    title = code
                else:
                    code = query
                    df_full = load_price_data(code, fetch_start.strftime('%Y-%m-%d'),
                                             end_date.strftime('%Y-%m-%d'))
                    title = selected_crypto

                if df_full.empty:
                    st.error("데이터를 불러올 수 없습니다.")
                    st.stop()
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")
                st.stop()

        # 지표 계산은 전체 데이터로, 차트 표시는 원래 기간으로
        close = df_full['Close']

        # 볼린저 밴드
        bb_upper, bb_mid, bb_lower = calc_bb(close)

        # 이치모쿠
        tenkan, kijun, spanA, spanB, chikou = calc_ichimoku(df_full)

        # 이동평균선
        mas = {p: calc_ma(close, p, ma_kind) for p in ma_selected}

        # RSI / MACD
        rsi   = calc_rsi(close)
        macd_line, macd_signal, macd_hist = calc_macd(close)

        # 표시 구간으로 자르기
        df = df_full[df_full.index >= pd.Timestamp(start_date.strftime('%Y-%m-%d'))].copy()
        idx = df_full.index >= pd.Timestamp(start_date.strftime('%Y-%m-%d'))
        bb_upper, bb_mid, bb_lower = bb_upper[idx], bb_mid[idx], bb_lower[idx]
        tenkan, kijun = tenkan[idx], kijun[idx]
        spanA, spanB  = spanA[idx], spanB[idx]
        chikou        = chikou[idx]
        rsi           = rsi[idx]
        macd_line, macd_signal, macd_hist = macd_line[idx], macd_signal[idx], macd_hist[idx]
        mas           = {p: s[idx] for p, s in mas.items()}

        # ── 서브플롯 구성 ──────────────────────────────────────────────────
        n_sub = 1  # volume 항상
        if show_rsi:  n_sub += 1
        if show_macd: n_sub += 1

        row_heights = [0.60] if n_sub == 1 else ([0.55, 0.20, 0.25] if n_sub == 3 else [0.60, 0.18, 0.22])
        # volume은 항상 마지막 앞에, RSI/MACD 순서
        sub_rows = ['volume']
        if show_rsi:  sub_rows.append('rsi')
        if show_macd: sub_rows.append('macd')

        total_rows = 1 + len(sub_rows)
        rh_main   = 0.60
        rh_volume = 0.12
        rh_rsi    = 0.14 if show_rsi  else 0
        rh_macd   = 0.14 if show_macd else 0
        extra = rh_volume + rh_rsi + rh_macd
        rh_main = 1.0 - extra
        r_heights = [rh_main, rh_volume] + ([rh_rsi] if show_rsi else []) + ([rh_macd] if show_macd else [])

        vol_row  = 2
        rsi_row  = 3 if show_rsi else None
        macd_row = (3 if not show_rsi else 4) if show_macd else None

        fig = make_subplots(
            rows=total_rows, cols=1,
            row_heights=r_heights,
            shared_xaxes=True,
            vertical_spacing=0.02,
        )

        # ── 캔들차트 ──────────────────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#FF3B3B', increasing_fillcolor='#FF3B3B',
            decreasing_line_color='#3B8BFF', decreasing_fillcolor='#3B8BFF',
            name='캔들', showlegend=False,
        ), row=1, col=1)

        # ── 볼린저 밴드 ──────────────────────────────────────────────────
        if show_bb:
            fig.add_trace(go.Scatter(
                x=df.index, y=bb_upper, mode='lines',
                line=dict(color='rgba(255,200,0,0.6)', width=1),
                name='BB상단', showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=bb_mid, mode='lines',
                line=dict(color='rgba(255,200,0,0.4)', width=1, dash='dot'),
                name='BB중심', showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=bb_lower, mode='lines',
                line=dict(color='rgba(255,200,0,0.6)', width=1),
                fill='tonexty',
                fillcolor='rgba(255,200,0,0.04)',
                name='BB하단', showlegend=True,
            ), row=1, col=1)

        # ── 이치모쿠 ─────────────────────────────────────────────────────
        if show_ichimoku:
            fig.add_trace(go.Scatter(
                x=df.index, y=tenkan, mode='lines',
                line=dict(color='#FF6B6B', width=1),
                name='전환선(9)', showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=kijun, mode='lines',
                line=dict(color='#4DABF7', width=1),
                name='기준선(26)', showlegend=True,
            ), row=1, col=1)
            # 구름 (A > B → 상승구름 녹색 / A < B → 하락구름 빨강)
            fig.add_trace(go.Scatter(
                x=df.index, y=spanA, mode='lines',
                line=dict(color='rgba(0,200,100,0)', width=0),
                name='선행A', showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=spanB, mode='lines',
                line=dict(color='rgba(0,200,100,0)', width=0),
                fill='tonexty',
                fillcolor='rgba(0,200,80,0.15)',
                name='구름(상승)', showlegend=True,
            ), row=1, col=1)
            # 하락 구름 (spanB > spanA 구간)
            spanA_arr = spanA.values
            spanB_arr = spanB.values
            bear_spanA = np.where(spanB_arr >= spanA_arr, spanA_arr, np.nan)
            bear_spanB = np.where(spanB_arr >= spanA_arr, spanB_arr, np.nan)
            fig.add_trace(go.Scatter(
                x=df.index, y=bear_spanA, mode='lines',
                line=dict(color='rgba(255,80,80,0)', width=0),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=bear_spanB, mode='lines',
                line=dict(color='rgba(255,80,80,0)', width=0),
                fill='tonexty',
                fillcolor='rgba(255,60,60,0.15)',
                name='구름(하락)', showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=chikou, mode='lines',
                line=dict(color='rgba(180,180,180,0.5)', width=1, dash='dot'),
                name='후행스팬', showlegend=True,
            ), row=1, col=1)

        # ── 이동평균선 ────────────────────────────────────────────────────
        for period, ma_series in mas.items():
            fig.add_trace(go.Scatter(
                x=df.index, y=ma_series,
                mode='lines',
                line=dict(color=MA_COLORS.get(period, '#AAAAAA'), width=1.2),
                name=f'{ma_kind}{period}',
            ), row=1, col=1)

        # ── ZigZag + 엘리엇 파동 ─────────────────────────────────────────
        current_price = df['Close'].iloc[-1]
        pivots = zigzag(df['Close'], threshold)
        result = count_elliott_waves(pivots)

        zz_x = [df.index[p['idx']] for p in pivots]
        zz_y = [p['price'] for p in pivots]
        fig.add_trace(go.Scatter(
            x=zz_x, y=zz_y, mode='lines',
            line=dict(color='rgba(150,150,150,0.4)', width=1, dash='dot'),
            name='ZigZag', showlegend=False,
        ), row=1, col=1)

        if result:
            wave_pts = result['wave_points']
            wv_colors = {0: 'white', 1: '#FFD700', 2: '#FFA500', 3: '#FF4444', 4: '#FF8C00'}
            wv_labels = {0: '0', 1: '①', 2: '②', 3: '③', 4: '④'}
            for wn, pt in wave_pts.items():
                dt = df.index[pt['idx']]
                price = pt['price']
                offset = price * 0.02
                fig.add_trace(go.Scatter(
                    x=[dt], y=[price + (offset if pt['type'] == 'H' else -offset)],
                    mode='markers+text',
                    marker=dict(size=12, color=wv_colors.get(wn, 'white'),
                                symbol='circle', line=dict(width=1, color='black')),
                    text=[wv_labels.get(wn, str(wn))],
                    textposition='top center' if pt['type'] == 'H' else 'bottom center',
                    textfont=dict(size=14, color=wv_colors.get(wn, 'white')),
                    name=f"{wn}파", showlegend=False,
                ), row=1, col=1)

            sorted_waves = sorted(wave_pts.items())
            fig.add_trace(go.Scatter(
                x=[df.index[p['idx']] for _, p in sorted_waves],
                y=[p['price'] for _, p in sorted_waves],
                mode='lines',
                line=dict(color='rgba(255,215,0,0.5)', width=2),
                name='파동연결', showlegend=False,
            ), row=1, col=1)

            zero_price = wave_pts[0]['price']
            fig.add_hline(y=zero_price, line_dash='dot', line_color='#6699FF', line_width=1.5,
                          annotation_text=f"무효라인 {zero_price:,.0f}",
                          annotation_position="bottom right", row=1, col=1)
            if current_price < zero_price:
                st.error(f"⚠️ 무효라인 이탈! 현재가 {current_price:,.0f} < 무효라인 {zero_price:,.0f}")

            fib_colors = {'3파_1.618': 'gold', '3파_2.618': 'orange',
                          '5파_0.618': '#FFFF66', '5파_1.000': '#FFD700', '5파_1.618': '#FFA500'}
            for label, price in result['fib_targets'].items():
                pct_diff = (price - current_price) / current_price * 100
                sign = "+" if pct_diff >= 0 else ""
                fig.add_hline(y=price, line_dash='dash',
                              line_color=fib_colors.get(label, 'gold'), line_width=1,
                              annotation_text=f"{label}: {price:,.0f} ({sign}{pct_diff:.1f}%)",
                              annotation_position="right", row=1, col=1)

            tl = compute_trendline(wave_pts, df)
            n_bars = len(df)
            if tl:
                tl_x = [df.index[xi] for xi in range(tl['anchor']['idx'], n_bars)]
                tl_y = [trendline_price_at(tl, xi) for xi in range(tl['anchor']['idx'], n_bars)]
                fig.add_trace(go.Scatter(x=tl_x, y=tl_y, mode='lines',
                                         line=dict(color='#00FF88', width=2),
                                         name='퍼즈각도선(2-4)'), row=1, col=1)

            low_waves  = [pt for wn, pt in wave_pts.items() if pt['type'] == 'L' and wn != 0]
            high_waves = [pt for wn, pt in wave_pts.items() if pt['type'] == 'H']
            if len(low_waves) >= 2 and len(high_waves) >= 1:
                lp1, lp2 = sorted(low_waves, key=lambda x: x['idx'])[:2]
                slope_ch = (lp2['price'] - lp1['price']) / max(lp2['idx'] - lp1['idx'], 1)
                end_xi   = min(n_bars - 1, lp2['idx'] + int(n_bars * 0.1))
                ch_x     = [df.index[lp1['idx']], df.index[end_xi]]
                ch_y_low = [lp1['price'], lp1['price'] + slope_ch * (end_xi - lp1['idx'])]
                offset_ch = high_waves[0]['price'] - (lp1['price'] + slope_ch * (high_waves[0]['idx'] - lp1['idx']))
                ch_y_high = [y + offset_ch for y in ch_y_low]
                for ch_y, ch_name in [(ch_y_low, '채널하단'), (ch_y_high, '채널상단')]:
                    fig.add_trace(go.Scatter(x=ch_x, y=ch_y, mode='lines',
                                             line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dash'),
                                             name=ch_name, showlegend=False), row=1, col=1)

        # ── 거래량 ────────────────────────────────────────────────────────
        vol_colors = ['#FF3B3B' if c >= o else '#3B8BFF'
                      for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                             marker_color=vol_colors, name='거래량',
                             opacity=0.6, showlegend=False), row=vol_row, col=1)

        # ── RSI ───────────────────────────────────────────────────────────
        if show_rsi and rsi_row:
            fig.add_trace(go.Scatter(
                x=df.index, y=rsi, mode='lines',
                line=dict(color='#BB86FC', width=1.5),
                name='RSI(14)', showlegend=False,
            ), row=rsi_row, col=1)
            for lvl, clr, lbl in [(70, 'rgba(255,80,80,0.4)', '과매수'), (30, 'rgba(80,200,255,0.4)', '과매도')]:
                fig.add_hline(y=lvl, line_dash='dot', line_color=clr, line_width=1,
                              annotation_text=lbl, annotation_position="right", row=rsi_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=rsi_row, col=1)

        # ── MACD ─────────────────────────────────────────────────────────
        if show_macd and macd_row:
            hist_colors = ['#FF3B3B' if v >= 0 else '#3B8BFF' for v in macd_hist]
            fig.add_trace(go.Bar(x=df.index, y=macd_hist,
                                 marker_color=hist_colors, name='MACD 히스토그램',
                                 opacity=0.7, showlegend=False), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode='lines',
                                     line=dict(color='#00BFFF', width=1.5),
                                     name='MACD', showlegend=False), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd_signal, mode='lines',
                                     line=dict(color='#FF8C00', width=1.5),
                                     name='시그널', showlegend=False), row=macd_row, col=1)
            fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=macd_row, col=1)
            fig.update_yaxes(title_text="MACD", row=macd_row, col=1)

        # ── 레이아웃 ──────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=18)),
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=820,
            legend=dict(orientation='h', yanchor='bottom', y=1.01,
                        xanchor='right', x=1, font=dict(size=11)),
            margin=dict(r=130, t=60),
            hovermode='x unified',
        )
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=vol_row, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ── 분석 결과 패널 ────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            price_unit = "원" if market == "한국 주식" else ("$" if market == "미국 주식" else "")
            price_fmt = f"${current_price:,.2f}" if market == "미국 주식" else f"{current_price:,.0f}{price_unit}"
            st.metric("현재가", price_fmt)
            rsi_now = rsi.iloc[-1]
            rsi_status = "과매수 🔴" if rsi_now >= 70 else ("과매도 🟢" if rsi_now <= 30 else "중립 ⚪")
            st.metric("RSI(14)", f"{rsi_now:.1f}", rsi_status)
            if pivots:
                st.caption(f"ZigZag 피벗 {len(pivots)}개 (민감도 {threshold}%)")

        if result:
            wave_pts = result['wave_points']
            with col2:
                st.subheader("📋 파동 검증")
                for rule, ok in result['validations']:
                    if ok is None:  st.write(f"⚪ {rule} — 데이터 부족")
                    elif ok:        st.write(f"✅ {rule} — PASS")
                    else:           st.write(f"❌ {rule} — FAIL")
            with col3:
                st.subheader("🎯 피보나치 목표가")
                for label, price in result['fib_targets'].items():
                    pct = (price - current_price) / current_price * 100
                    sign = "+" if pct >= 0 else ""
                    st.write(f"**{label}**: {price:,.0f} ({sign}{pct:.1f}%)")
            tl = compute_trendline(wave_pts, df)
            if tl:
                tl_now = trendline_price_at(tl, len(df) - 1)
                status = "🟢 HOLD" if current_price > tl_now else "🔴 BREAK"
                st.info(f"**퍼즈각도선**: {tl_now:,.0f}  |  {status}")
            st.subheader("📊 파동 포인트")
            rows = []
            for wn in sorted(wave_pts.keys()):
                pt = wave_pts[wn]
                rows.append({'파동': wn, '날짜': df.index[pt['idx']].strftime('%Y-%m-%d'),
                             '가격': f"{pt['price']:,.0f}",
                             '타입': '고점' if pt['type'] == 'H' else '저점'})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("파동 구조를 탐지하지 못했습니다. 조회 기간을 늘리거나 민감도를 낮춰보세요.")

        # ── 펀더멘털 체크 ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔎 펀더멘털 체크")

        if market == "암호화폐":
            st.info("펀더멘털 체크는 주식만 지원합니다.")
        else:
            with st.spinner("펀더멘털 데이터 조회 중..."):
                fd = fetch_fundamental(code, market)

            def fund_chip(label, value_str, status_color, comment):
                bg  = {'green': '#002211', 'yellow': '#1f1800', 'red': '#2d0000', 'gray': '#1a1a1a'}[status_color]
                bdr = {'green': '#00cc44', 'yellow': '#ffcc00', 'red': '#ff3333', 'gray': '#555555'}[status_color]
                return (f"<div style='background:{bg};border-left:4px solid {bdr};"
                        f"border-radius:8px;padding:14px 16px;'>"
                        f"<div style='color:#aaa;font-size:0.8em;margin-bottom:2px;'>{label}</div>"
                        f"<div style='color:{bdr};font-size:1.6em;font-weight:bold;'>{value_str}</div>"
                        f"<div style='color:#ccc;font-size:0.82em;margin-top:4px;'>{comment}</div>"
                        f"</div>")

            fc1, fc2, fc3 = st.columns(3)
            green_count = 0
            red_count   = 0

            # PER (한국/미국 공통)
            per = fd.get('per')
            per_label = "PER (주가수익비율)"
            if per is None:
                with fc1: st.markdown(fund_chip(per_label, "조회 불가", "gray", "데이터 없음"), unsafe_allow_html=True)
            elif per <= 15:
                green_count += 1
                with fc1: st.markdown(fund_chip(per_label, f"{per:.1f}배", "green", "✅ 저평가 구간 (≤15)"), unsafe_allow_html=True)
            elif per <= 25:
                with fc1: st.markdown(fund_chip(per_label, f"{per:.1f}배", "yellow", "⚠️ 적정 수준 (15~25)"), unsafe_allow_html=True)
            else:
                red_count += 1
                with fc1: st.markdown(fund_chip(per_label, f"{per:.1f}배", "red", "🚫 고평가 구간 (>25)"), unsafe_allow_html=True)

            # PBR (한국/미국 공통)
            pbr = fd.get('pbr')
            pbr_label = "PBR (주가순자산비율)"
            if pbr is None:
                with fc2: st.markdown(fund_chip(pbr_label, "조회 불가", "gray", "데이터 없음"), unsafe_allow_html=True)
            elif pbr <= 1.0:
                green_count += 1
                with fc2: st.markdown(fund_chip(pbr_label, f"{pbr:.2f}배", "green", "✅ 자산가치 이하 (≤1.0)"), unsafe_allow_html=True)
            elif pbr <= 2.0:
                with fc2: st.markdown(fund_chip(pbr_label, f"{pbr:.2f}배", "yellow", "⚠️ 적정 수준 (1~2)"), unsafe_allow_html=True)
            else:
                red_count += 1
                with fc2: st.markdown(fund_chip(pbr_label, f"{pbr:.2f}배", "red", "🚫 고평가 (>2.0)"), unsafe_allow_html=True)

            # 세 번째 카드: 미국은 공매도 커버일수, 한국은 기관보유율
            if market == "미국 주식":
                sr = fd.get('short_ratio')
                if sr is None:
                    with fc3: st.markdown(fund_chip("공매도 커버일수 (Short Ratio)", "조회 불가", "gray", "데이터 없음"), unsafe_allow_html=True)
                elif sr <= 3:
                    green_count += 1
                    with fc3: st.markdown(fund_chip("공매도 커버일수 (Short Ratio)", f"{sr:.1f}일", "green", "✅ 공매도 압력 낮음 (≤3일)"), unsafe_allow_html=True)
                elif sr <= 7:
                    with fc3: st.markdown(fund_chip("공매도 커버일수 (Short Ratio)", f"{sr:.1f}일", "yellow", "⚠️ 보통 수준 (3~7일)"), unsafe_allow_html=True)
                else:
                    red_count += 1
                    with fc3: st.markdown(fund_chip("공매도 커버일수 (Short Ratio)", f"{sr:.1f}일", "red", "🚫 공매도 압력 높음 (>7일)"), unsafe_allow_html=True)
            else:
                ih = fd.get('institutional_hold')
                if ih is None:
                    with fc3: st.markdown(fund_chip("기관 보유율", "조회 불가", "gray", "데이터 없음"), unsafe_allow_html=True)
                elif ih >= 0.30:
                    green_count += 1
                    with fc3: st.markdown(fund_chip("기관 보유율", f"{ih*100:.1f}%", "green", "✅ 기관 비중 높음 (≥30%)"), unsafe_allow_html=True)
                elif ih >= 0.15:
                    with fc3: st.markdown(fund_chip("기관 보유율", f"{ih*100:.1f}%", "yellow", "⚠️ 기관 비중 보통 (15~30%)"), unsafe_allow_html=True)
                else:
                    red_count += 1
                    with fc3: st.markdown(fund_chip("기관 보유율", f"{ih*100:.1f}%", "red", "🚫 기관 비중 낮음 (<15%)"), unsafe_allow_html=True)

            # 종합 판정
            st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
            if green_count >= 2:
                st.success("✅ 파동 진입 근거 보강됨 — 펀더멘털이 기술적 신호를 뒷받침합니다.")
            elif red_count >= 2:
                st.error("🚫 펀더멘털 역풍 — 파동 신호가 나와도 신중하게 접근하세요.")
            else:
                st.warning("⚠️ 혼조 — 추가 확인 후 판단하세요.")
            st.markdown("</div>", unsafe_allow_html=True)

            if market == "미국 주식":
                st.caption("PER·PBR·공매도: yfinance (Yahoo Finance 기준) | 중장기 참고용")
            else:
                st.caption("PER·PBR: Naver Finance 실시간 스크래핑 | 기관보유율: yfinance (분기 공시 기준) | 중장기 참고용")

    else:
        st.info("👈 사이드바에서 설정 후 **분석 시작**을 클릭하세요.")
        st.markdown("""
        ### 추가된 보조지표
        - **볼린저 밴드**: 20일 SMA ± 2σ
        - **이치모쿠 구름**: 전환선·기준선·구름·후행스팬
        - **이동평균선**: EMA/SMA 5·20·60·120·200 (조합 선택)
        - **RSI(14)**: 과매수(70) / 과매도(30) 기준선
        - **MACD(12/26/9)**: 히스토그램 + MACD선 + 시그널선
        """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: 종합 매매 판단
# ════════════════════════════════════════════════════════════════════════════

def get_signal_data(df):
    """전체 지표 계산 및 신호 딕셔너리 반환"""
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']
    cur    = close.iloc[-1]

    # 이동평균 (EMA 기준)
    ma_periods = [5, 20, 60, 120, 200]
    ma_val = {}
    for p in ma_periods:
        if len(close) >= p:
            ma_val[p] = close.ewm(span=p, adjust=False).mean().iloc[-1]

    # 이평선 배열 점수 (정배열일수록 높음)
    align_score = 0
    sorted_p = sorted(ma_val.keys())
    for i in range(len(sorted_p) - 1):
        if ma_val[sorted_p[i]] > ma_val[sorted_p[i+1]]:
            align_score += 1   # 단기 > 장기 = 정배열
        else:
            align_score -= 1
    max_pairs = max(len(sorted_p) - 1, 1)

    # 현재가 vs 각 MA 위치
    above_ma = {p: cur > v for p, v in ma_val.items()}

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    rsi_now    = rsi_series.iloc[-1]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    macd_now    = macd_hist.iloc[-1]
    macd_prev   = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
    if macd_prev < 0 and macd_now >= 0:
        macd_cross = 'golden'
    elif macd_prev > 0 and macd_now <= 0:
        macd_cross = 'dead'
    else:
        macd_cross = 'none'
    macd_above_zero = macd_line.iloc[-1] > 0

    # 볼린저밴드 %B
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).iloc[-1]
    bb_pct   = ((cur - bb_lower.iloc[-1]) / bb_range * 100) if bb_range > 0 else 50

    # 거래량 (최근 5일 평균 / 20일 평균)
    vol_ratio = (volume.rolling(5).mean().iloc[-1] /
                 volume.rolling(20).mean().iloc[-1]) if volume.rolling(20).mean().iloc[-1] > 0 else 1

    # 수익률
    def ret(n):
        return (cur / close.iloc[-n-1] - 1) * 100 if len(close) > n else 0
    ret_5, ret_20, ret_60 = ret(5), ret(20), ret(60)

    # 지지/저항 (ZigZag 피벗에서 추출)
    pivots_raw = zigzag(close, 5)
    recent_highs = sorted([p['price'] for p in pivots_raw if p['type'] == 'H' and p['price'] > cur], )
    recent_lows  = sorted([p['price'] for p in pivots_raw if p['type'] == 'L' and p['price'] < cur],
                          reverse=True)
    support1    = recent_lows[0]  if recent_lows  else cur * 0.95
    support2    = recent_lows[1]  if len(recent_lows) > 1 else support1 * 0.97
    resistance1 = recent_highs[0] if recent_highs else cur * 1.05
    resistance2 = recent_highs[1] if len(recent_highs) > 1 else resistance1 * 1.03

    return dict(
        cur=cur, ma_val=ma_val, align_score=align_score, max_pairs=max_pairs,
        above_ma=above_ma, rsi_now=rsi_now, macd_now=macd_now, macd_cross=macd_cross,
        macd_above_zero=macd_above_zero, bb_pct=bb_pct, vol_ratio=vol_ratio,
        ret_5=ret_5, ret_20=ret_20, ret_60=ret_60,
        support1=support1, support2=support2,
        resistance1=resistance1, resistance2=resistance2,
    )


def calc_verdict(s):
    """신호 점수 → 종합 판단"""
    score = 50

    # 이평선 배열 (최대 ±20)
    score += (s['align_score'] / s['max_pairs']) * 20

    # 현재가 위치 (120MA, 200MA)
    if s['above_ma'].get(120, False): score += 7
    else: score -= 7
    if s['above_ma'].get(200, False): score += 5
    else: score -= 5
    if s['above_ma'].get(60, False):  score += 4
    else: score -= 4

    # RSI
    r = s['rsi_now']
    if   r > 80:           score -= 15
    elif r > 70:           score -= 8
    elif 60 < r <= 70:     score += 3
    elif 45 <= r <= 60:    score += 8
    elif 35 <= r < 45:     score += 4
    elif 25 <= r < 35:     score -= 3   # 과매도 직전, 반등 가능이지만 위험
    else:                  score -= 15  # 극단적 과매도

    # MACD
    if s['macd_above_zero']:   score += 6
    else:                      score -= 6
    if s['macd_cross'] == 'golden': score += 10
    elif s['macd_cross'] == 'dead': score -= 10

    # 볼린저밴드 %B
    bp = s['bb_pct']
    if   bp > 95: score -= 12
    elif bp > 80: score -= 4
    elif bp < 10: score -= 8   # 극단적 하락 = 위험 신호
    elif bp < 25: score += 4   # 하단 근처 = 단기 반등 가능
    elif 40 <= bp <= 65: score += 4

    # 거래량 추세 (최근 거래 증가 = 관심 증가)
    if s['vol_ratio'] > 1.5:  score += 4
    elif s['vol_ratio'] < 0.7: score -= 2

    score = max(0, min(100, round(score)))

    if score >= 75:
        return dict(score=score, label="강한 매수 신호", action="매수 / 추가매수",
                    color="#00cc44", bg="#002211", emoji="🟢",
                    summary="추세가 강하게 살아있습니다. 진입 또는 추가매수를 고려해볼 수 있습니다.")
    elif score >= 60:
        return dict(score=score, label="매수 고려", action="매수 / 보유",
                    color="#88cc00", bg="#111f00", emoji="🟢",
                    summary="상승 추세 유지 중. 조정 구간에서 분할 매수 전략이 유효합니다.")
    elif score >= 47:
        return dict(score=score, label="관망", action="관망 / 보유",
                    color="#ffcc00", bg="#1f1800", emoji="🟡",
                    summary="방향성이 불확실합니다. 추가 신호를 확인한 후 판단하세요.")
    elif score >= 33:
        return dict(score=score, label="주의 — 비중 축소 고려", action="일부 익절 / 관망",
                    color="#ff8800", bg="#1f0e00", emoji="🟠",
                    summary="추세가 약화되고 있습니다. 보유 중이라면 비중 축소를 고려하세요.")
    else:
        return dict(score=score, label="손절 / 매도 신호", action="손절 / 매도",
                    color="#ff2222", bg="#2d0000", emoji="🔴",
                    summary="추세 전환 가능성이 높습니다. 핵심 지지선 이탈 시 즉시 청산을 고려하세요.")


with tab_signal:
    if analyze_btn:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days)
        fetch_start = start_date - timedelta(days=150)

        with st.spinner("분석 중..."):
            try:
                if market == "한국 주식":
                    stock_df = load_stock_list()
                    code, name = find_stock_code(query, stock_df)
                    if not (code.isdigit() and len(code) == 6):
                        from pykrx import stock as krx
                        today_str = end_date.strftime('%Y%m%d')
                        all_codes = krx.get_market_ticker_list(today_str, market="ALL")
                        matched = [(c, krx.get_market_ticker_name(c)) for c in all_codes
                                   if query in krx.get_market_ticker_name(c)]
                        if matched:
                            code, name = matched[0]
                        else:
                            st.error(f"'{query}' 종목을 찾을 수 없습니다.\n\n"
                                     "6자리 종목코드를 직접 입력해보세요.")
                            st.stop()
                    df_s = load_price_data(code, fetch_start.strftime('%Y-%m-%d'),
                                           end_date.strftime('%Y-%m-%d'))
                    sig_title = f"{name} ({code})"
                elif market == "미국 주식":
                    code = query.upper().strip()
                    df_s = load_price_data(code, fetch_start.strftime('%Y-%m-%d'),
                                           end_date.strftime('%Y-%m-%d'))
                    sig_title = code
                else:
                    code  = query
                    df_s  = load_price_data(code, fetch_start.strftime('%Y-%m-%d'),
                                            end_date.strftime('%Y-%m-%d'))
                    sig_title = selected_crypto
                if df_s.empty:
                    st.error("데이터를 불러올 수 없습니다.")
                    st.stop()
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")
                st.stop()

        s = get_signal_data(df_s)
        v = calc_verdict(s)
        cur      = s['cur']
        unit_str = "원" if market == "한국 주식" else ("$" if market == "미국 주식" else "")

        # ── 종합 판정 박스 ─────────────────────────────────────────────────
        st.markdown(f"## {sig_title} 종합 매매 판단")
        st.markdown(
            f"<div style='background:{v['bg']};border:2px solid {v['color']};border-radius:12px;"
            f"padding:20px 24px;margin-bottom:16px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div>"
            f"<div style='color:{v['color']};font-size:2em;font-weight:bold;'>"
            f"{v['emoji']} {v['label']}</div>"
            f"<div style='color:#cccccc;font-size:1.1em;margin-top:6px;'>{v['summary']}</div>"
            f"</div>"
            f"<div style='text-align:right;'>"
            f"<div style='color:#aaaaaa;font-size:0.9em;'>종합 점수</div>"
            f"<div style='color:{v['color']};font-size:3em;font-weight:bold;'>{v['score']}</div>"
            f"<div style='color:#aaaaaa;font-size:0.85em;'>/ 100</div>"
            f"</div></div>"
            f"<div style='margin-top:14px;background:rgba(255,255,255,0.08);border-radius:8px;height:12px;'>"
            f"<div style='background:{v['color']};width:{v['score']}%;height:100%;border-radius:8px;'></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── 핵심 가격선 ────────────────────────────────────────────────────
        st.markdown("---")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("현재가", f"{cur:,.0f}{unit_str}")
        pc2.metric("1차 지지선 (손절 기준)",
                   f"{s['support1']:,.0f}{unit_str}",
                   f"{(s['support1']-cur)/cur*100:.1f}%",
                   delta_color="off")
        pc3.metric("2차 지지선",
                   f"{s['support2']:,.0f}{unit_str}",
                   f"{(s['support2']-cur)/cur*100:.1f}%",
                   delta_color="off")
        pc4.metric("1차 저항선 (목표)",
                   f"{s['resistance1']:,.0f}{unit_str}",
                   f"+{(s['resistance1']-cur)/cur*100:.1f}%")

        # ── 지표별 신호 카드 ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📡 지표별 신호")

        def signal_card(label, value_str, status, comment, color):
            bg = {"🟢": "#002211", "🟡": "#1f1800", "🔴": "#2d0000"}.get(status, "#1a1a1a")
            return (f"<div style='background:{bg};border-left:4px solid {color};"
                    f"border-radius:8px;padding:12px 14px;height:100%;'>"
                    f"<div style='color:#aaaaaa;font-size:0.8em;'>{label}</div>"
                    f"<div style='color:{color};font-size:1.4em;font-weight:bold;'>{value_str}</div>"
                    f"<div style='color:#cccccc;font-size:0.85em;margin-top:4px;'>{status} {comment}</div>"
                    f"</div>")

        # 이평선 배열
        total_p = len(s['ma_val'])
        above_cnt = sum(1 for p in [20,60,120,200] if s['above_ma'].get(p, False))
        align_ratio = s['align_score'] / s['max_pairs']
        if align_ratio >= 0.6:
            align_clr = "#00cc44"; align_st = "🟢"; align_txt = "정배열 — 상승 추세"
        elif align_ratio <= -0.6:
            align_clr = "#ff3333"; align_st = "🔴"; align_txt = "역배열 — 하락 추세"
        else:
            align_clr = "#ffcc00"; align_st = "🟡"; align_txt = "혼조 — 추세 전환 중"

        # RSI 해석
        r = s['rsi_now']
        if r > 70:   rsi_clr="#ff3333"; rsi_st="🔴"; rsi_txt="과매수 — 단기 조정 주의"
        elif r < 30: rsi_clr="#ff8800"; rsi_st="🔴"; rsi_txt="과매도 — 반등 가능하나 위험"
        elif r >= 50: rsi_clr="#00cc44"; rsi_st="🟢"; rsi_txt="강세 구간"
        elif r >= 40: rsi_clr="#88cc00"; rsi_st="🟡"; rsi_txt="중립 — 방향 확인 필요"
        else:        rsi_clr="#ffcc00"; rsi_st="🟡"; rsi_txt="약세 — 반등 여부 주시"

        # MACD 해석
        if s['macd_cross'] == 'golden':
            macd_clr="#00cc44"; macd_st="🟢"; macd_txt="골든크로스 발생!"
        elif s['macd_cross'] == 'dead':
            macd_clr="#ff3333"; macd_st="🔴"; macd_txt="데드크로스 발생!"
        elif s['macd_above_zero']:
            macd_clr="#88cc00"; macd_st="🟢"; macd_txt="0선 위 — 상승 모멘텀"
        else:
            macd_clr="#ff8800"; macd_st="🟠"; macd_txt="0선 아래 — 하락 모멘텀"

        # BB %B 해석
        bp = s['bb_pct']
        if bp > 90:   bb_clr="#ff3333"; bb_st="🔴"; bb_txt="상단 돌파 — 과열 주의"
        elif bp > 70: bb_clr="#ffcc00"; bb_st="🟡"; bb_txt="상단 근접 — 과매수 구간"
        elif bp < 15: bb_clr="#ff8800"; bb_st="🔴"; bb_txt="하단 이탈 — 강한 하락 중"
        elif bp < 30: bb_clr="#88cc00"; bb_st="🟡"; bb_txt="하단 근접 — 반등 가능"
        else:         bb_clr="#00cc44"; bb_st="🟢"; bb_txt="밴드 중간 — 안정적"

        # 거래량
        vr = s['vol_ratio']
        if vr > 1.5:   vol_clr="#00cc44"; vol_st="🟢"; vol_txt="거래량 급증 — 강한 관심"
        elif vr > 1.1: vol_clr="#88cc00"; vol_st="🟢"; vol_txt="평균 이상"
        elif vr > 0.8: vol_clr="#ffcc00"; vol_st="🟡"; vol_txt="평균 수준"
        else:          vol_clr="#aaaaaa"; vol_st="🟡"; vol_txt="거래 감소"

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1: st.markdown(signal_card("이평선 배열", align_txt.split("—")[0].strip(),
                                          align_st, align_txt.split("—")[1].strip(), align_clr),
                              unsafe_allow_html=True)
        with r2: st.markdown(signal_card("RSI (14)", f"{r:.1f}",
                                          rsi_st, rsi_txt, rsi_clr), unsafe_allow_html=True)
        with r3: st.markdown(signal_card("MACD", "골든크로스" if s['macd_cross']=='golden'
                                          else ("데드크로스" if s['macd_cross']=='dead'
                                          else ("0선 위" if s['macd_above_zero'] else "0선 아래")),
                                          macd_st, macd_txt, macd_clr), unsafe_allow_html=True)
        with r4: st.markdown(signal_card("BB %B", f"{bp:.0f}%",
                                          bb_st, bb_txt, bb_clr), unsafe_allow_html=True)
        with r5: st.markdown(signal_card("거래량", f"평균 대비 {vr:.1f}x",
                                          vol_st, vol_txt, vol_clr), unsafe_allow_html=True)

        # ── 이평선 상세 ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📏 이동평균선 상세")
        ma_rows = []
        for p in [5, 20, 60, 120, 200]:
            if p in s['ma_val']:
                v_ma  = s['ma_val'][p]
                diff  = (cur - v_ma) / v_ma * 100
                pos   = "위 ✅" if cur > v_ma else "아래 ❌"
                ma_rows.append({
                    "EMA": f"EMA {p}",
                    "현재값": f"{v_ma:,.0f}{unit_str}",
                    "현재가 위치": pos,
                    "괴리율": f"{'+' if diff>=0 else ''}{diff:.2f}%",
                })
        st.dataframe(pd.DataFrame(ma_rows), use_container_width=True, hide_index=True)

        # ── 수익률 현황 ────────────────────────────────────────────────────
        st.subheader("📈 기간별 수익률")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("5일", f"{'+' if s['ret_5']>=0 else ''}{s['ret_5']:.2f}%")
        rc2.metric("20일", f"{'+' if s['ret_20']>=0 else ''}{s['ret_20']:.2f}%")
        rc3.metric("60일", f"{'+' if s['ret_60']>=0 else ''}{s['ret_60']:.2f}%")

        # ── 물타기 / 손절 가이드 ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 상황별 가이드")

        # 물타기 조건 판단
        add_ok = (s['above_ma'].get(120, False) and
                  s['align_score'] >= 0 and
                  s['rsi_now'] < 55 and
                  s['rsi_now'] > 25)

        # 손절 조건 판단
        cut_now = (not s['above_ma'].get(120, False) and
                   s['align_score'] < 0 and
                   (s['macd_cross'] == 'dead' or not s['macd_above_zero']))

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            if add_ok:
                st.markdown(
                    "<div style='background:#002211;border:1px solid #00cc44;border-radius:10px;padding:16px;'>"
                    "<div style='color:#00cc44;font-size:1.1em;font-weight:bold;'>✅ 물타기 가능 구간</div>"
                    "<div style='color:#cccccc;font-size:0.9em;margin-top:8px;'>"
                    "장기 추세 살아있고 RSI 조정권.<br>"
                    "단, 1차 지지선 위에서만 분할 매수.</div></div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='background:#1f1100;border:1px solid #ff8800;border-radius:10px;padding:16px;'>"
                    "<div style='color:#ff8800;font-size:1.1em;font-weight:bold;'>⚠️ 물타기 위험 구간</div>"
                    "<div style='color:#cccccc;font-size:0.9em;margin-top:8px;'>"
                    "장기 이평선 하회 또는 추세 약화.<br>"
                    "추가 매수보다 관망을 권장.</div></div>",
                    unsafe_allow_html=True)

        with gc2:
            stop_pct = (cur - s['support1']) / cur * 100
            st.markdown(
                f"<div style='background:#1a1a2e;border:1px solid #4499ff;border-radius:10px;padding:16px;'>"
                f"<div style='color:#4499ff;font-size:1.1em;font-weight:bold;'>📌 손절 기준선</div>"
                f"<div style='color:#cccccc;font-size:0.9em;margin-top:8px;'>"
                f"1차 지지선: <b style='color:white;'>{s['support1']:,.0f}{unit_str}</b><br>"
                f"현재가 대비 <b style='color:#ff8888;'>{stop_pct:.1f}%</b> 하락 시 이탈.<br>"
                f"이탈 확인되면 손절 실행.</div></div>",
                unsafe_allow_html=True)

        with gc3:
            if cut_now:
                st.markdown(
                    "<div style='background:#2d0000;border:1px solid #ff2222;border-radius:10px;padding:16px;'>"
                    "<div style='color:#ff2222;font-size:1.1em;font-weight:bold;'>🚨 손절 적극 고려</div>"
                    "<div style='color:#cccccc;font-size:0.9em;margin-top:8px;'>"
                    "이평선 역배열 + MACD 음전환.<br>"
                    "추세 전환 가능성 높음.<br>보유 포지션 점검 필요.</div></div>",
                    unsafe_allow_html=True)
            else:
                tgt_pct = (s['resistance1'] - cur) / cur * 100
                st.markdown(
                    f"<div style='background:#002211;border:1px solid #44ff88;border-radius:10px;padding:16px;'>"
                    f"<div style='color:#44ff88;font-size:1.1em;font-weight:bold;'>🎯 1차 목표가</div>"
                    f"<div style='color:#cccccc;font-size:0.9em;margin-top:8px;'>"
                    f"저항선: <b style='color:white;'>{s['resistance1']:,.0f}{unit_str}</b><br>"
                    f"현재가 대비 <b style='color:#88ff88;'>+{tgt_pct:.1f}%</b> 상승 목표.<br>"
                    f"도달 시 일부 익절 고려.</div></div>",
                    unsafe_allow_html=True)

        # ── 미니 가격 차트 (최근 90일) ────────────────────────────────────
        st.markdown("---")
        st.subheader("📉 최근 90일 차트 + 지지/저항")
        df_chart = df_s[df_s.index >= df_s.index[-1] - pd.Timedelta(days=90)]

        fig_sig = go.Figure()
        cclr = ['#FF3B3B' if c >= o else '#3B8BFF'
                for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig_sig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'],  close=df_chart['Close'],
            increasing_line_color='#FF3B3B', increasing_fillcolor='#FF3B3B',
            decreasing_line_color='#3B8BFF', decreasing_fillcolor='#3B8BFF',
            name='캔들', showlegend=False,
        ))
        # EMA 20, 60, 120
        for p, clr in [(20,'#FF8800'), (60,'#FF44FF'), (120,'#44FFFF')]:
            if len(df_s) >= p:
                ma_s = df_s['Close'].ewm(span=p, adjust=False).mean()
                ma_chart = ma_s[ma_s.index >= df_chart.index[0]]
                fig_sig.add_trace(go.Scatter(x=ma_chart.index, y=ma_chart,
                                             mode='lines', line=dict(color=clr, width=1.2),
                                             name=f'EMA{p}'))
        # 지지/저항선
        for price, label, clr in [
            (s['support1'],    f"지지1 {s['support1']:,.0f}", '#4499ff'),
            (s['support2'],    f"지지2 {s['support2']:,.0f}", '#2266cc'),
            (s['resistance1'], f"저항1 {s['resistance1']:,.0f}", '#ff6644'),
            (s['resistance2'], f"저항2 {s['resistance2']:,.0f}", '#cc3322'),
        ]:
            fig_sig.add_hline(y=price, line_dash='dash', line_color=clr, line_width=1.2,
                              annotation_text=label, annotation_position="right")
        fig_sig.update_layout(
            template='plotly_dark', height=420,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', y=1.05),
            margin=dict(r=120, t=30),
            hovermode='x unified',
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # ── 인사이트 & 주요 일정 ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("📰 인사이트 & 주요 일정")

        if market == "한국 주식":
            # ── DART 공시 ──────────────────────────────────────────────────
            with st.expander("📋 최근 공시 (60일)", expanded=True):
                if not dart_api_key:
                    st.info("사이드바에 DART API 키를 입력하면 공시 정보를 볼 수 있습니다.\n\n"
                            "**무료 발급:** dart.fss.or.kr → OpenAPI → 인증키 신청 (즉시 발급)")
                else:
                    with st.spinner("DART 공시 조회 중..."):
                        corp_code = dart_get_corp_code(code, dart_api_key)
                    if not corp_code:
                        st.warning("종목의 DART 기업코드를 찾을 수 없습니다.")
                    else:
                        disclosures = dart_get_disclosures(corp_code, dart_api_key, days=60)
                        if not disclosures:
                            st.info("최근 60일간 공시가 없습니다.")
                        else:
                            for d in disclosures[:15]:
                                rpt  = d.get('report_nm', '')
                                dt   = d.get('rcept_dt', '')
                                dt_f = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}" if len(dt) == 8 else dt
                                # 중요 키워드 매칭
                                icon, clr = '📄', '#aaaaaa'
                                for keyword, (ki, kc) in DART_REPORT_LABELS.items():
                                    if keyword in rpt:
                                        icon, clr = ki, kc
                                        break
                                st.markdown(
                                    f"<div style='padding:6px 10px;margin:3px 0;"
                                    f"border-left:3px solid {clr};border-radius:4px;"
                                    f"background:rgba(255,255,255,0.03);'>"
                                    f"<span style='color:#888;font-size:0.8em;'>{dt_f}</span>&nbsp;&nbsp;"
                                    f"{icon} <span style='color:{clr};'>{rpt}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

            # ── 네이버 분기 실적 ───────────────────────────────────────────
            with st.expander("📊 분기 실적 (네이버 금융)", expanded=False):
                with st.spinner("실적 데이터 조회 중..."):
                    earnings = naver_get_earnings(code)
                if earnings:
                    st.caption("※ 네이버 금융 기준 (연결 기준, 억원)")
                    st.write(" / ".join(str(x) for x in earnings[:8]))
                else:
                    st.info("실적 데이터를 가져오지 못했습니다. 네이버 금융에서 직접 확인해주세요.")
                st.markdown(f"[🔗 네이버 금융 바로가기](https://finance.naver.com/item/main.naver?code={code})")

        elif market == "미국 주식":
            # ── 미국 주식 — yfinance 뉴스 + 기업 정보 ──────────────────────
            with st.expander("🏢 기업 정보 & 주요 지표", expanded=True):
                try:
                    import yfinance as yf
                    t = yf.Ticker(code.upper())
                    info = t.info
                    long_name   = info.get('longName', code)
                    sector      = info.get('sector', '—')
                    industry    = info.get('industry', '—')
                    country     = info.get('country', '—')
                    market_cap  = info.get('marketCap')
                    fwd_pe      = info.get('forwardPE')
                    div_yield   = info.get('dividendYield')
                    week52_high = info.get('fiftyTwoWeekHigh')
                    week52_low  = info.get('fiftyTwoWeekLow')
                    cur_price   = info.get('currentPrice') or info.get('regularMarketPrice')

                    st.markdown(f"**{long_name}** ({code.upper()})")
                    st.caption(f"{sector} > {industry} | {country}")

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    if market_cap:
                        mc1.metric("시가총액", f"${market_cap/1e9:.1f}B")
                    if fwd_pe:
                        mc2.metric("Forward PER", f"{fwd_pe:.1f}배")
                    if div_yield:
                        mc3.metric("배당수익률", f"{div_yield*100:.2f}%")
                    if week52_high and week52_low and cur_price:
                        pos = (cur_price - week52_low) / (week52_high - week52_low) * 100
                        mc4.metric("52주 위치", f"{pos:.0f}%",
                                   help=f"52주 저점 ${week52_low:.2f} ~ 고점 ${week52_high:.2f}")
                except Exception as e:
                    st.info(f"기업 정보를 불러오지 못했습니다: {e}")

            with st.expander("📰 최근 뉴스 (Yahoo Finance)", expanded=True):
                try:
                    import yfinance as yf
                    t = yf.Ticker(code.upper())
                    news_list = t.news or []
                    if not news_list:
                        st.info("최근 뉴스가 없습니다.")
                    else:
                        for n in news_list[:8]:
                            title_n  = n.get('title', '')
                            pub_time = n.get('providerPublishTime', 0)
                            pub_str  = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d') if pub_time else ''
                            pub_name = n.get('publisher', '')
                            url_n    = n.get('link', '#')
                            st.markdown(
                                f"<div style='padding:6px 10px;margin:3px 0;"
                                f"border-left:3px solid #4488ff;border-radius:4px;"
                                f"background:rgba(255,255,255,0.03);'>"
                                f"<span style='color:#888;font-size:0.8em;'>{pub_str} · {pub_name}</span><br>"
                                f"<a href='{url_n}' target='_blank' style='color:#aaccff;text-decoration:none;'>"
                                f"{title_n}</a></div>",
                                unsafe_allow_html=True,
                            )
                except Exception:
                    st.info("뉴스를 불러오지 못했습니다.")

        else:
            # ── 코인 — 공포탐욕 + CoinGecko ──────────────────────────────
            coin_id = COINGECKO_IDS.get(code)

            # 공포·탐욕 지수
            with st.expander("😨 암호화폐 공포·탐욕 지수", expanded=True):
                fg_data = fetch_fear_greed()
                if fg_data:
                    cur_fg   = fg_data[0]
                    val      = int(cur_fg['value'])
                    cls_name = cur_fg['value_classification']
                    cls_kor  = {'Extreme Fear': '극단적 공포', 'Fear': '공포',
                                'Neutral': '중립', 'Greed': '탐욕',
                                'Extreme Greed': '극단적 탐욕'}.get(cls_name, cls_name)
                    if val <= 25:   fg_clr = '#ff2222'; fg_bg = '#2d0000'
                    elif val <= 45: fg_clr = '#ff8800'; fg_bg = '#1f1100'
                    elif val <= 55: fg_clr = '#ffcc00'; fg_bg = '#1f1800'
                    elif val <= 75: fg_clr = '#88cc00'; fg_bg = '#111f00'
                    else:           fg_clr = '#00cc44'; fg_bg = '#002211'

                    fc1, fc2 = st.columns([1, 2])
                    with fc1:
                        st.markdown(
                            f"<div style='background:{fg_bg};border:2px solid {fg_clr};"
                            f"border-radius:12px;padding:16px;text-align:center;'>"
                            f"<div style='color:{fg_clr};font-size:3em;font-weight:bold;'>{val}</div>"
                            f"<div style='color:{fg_clr};font-size:1em;'>{cls_kor}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with fc2:
                        st.markdown("**최근 7일 추이**")
                        for item in fg_data[:7]:
                            v   = int(item['value'])
                            c   = item['value_classification']
                            ck  = {'Extreme Fear':'극단 공포','Fear':'공포','Neutral':'중립',
                                   'Greed':'탐욕','Extreme Greed':'극단 탐욕'}.get(c, c)
                            bar = '█' * (v // 10) + '░' * (10 - v // 10)
                            clr = '#ff2222' if v<=25 else ('#ff8800' if v<=45 else
                                  ('#ffcc00' if v<=55 else ('#88cc00' if v<=75 else '#00cc44')))
                            ts  = item.get('timestamp','')
                            try:
                                dt_label = datetime.fromtimestamp(int(ts)).strftime('%m/%d')
                            except Exception:
                                dt_label = ''
                            st.markdown(
                                f"<span style='color:#666;font-size:0.8em;'>{dt_label}</span> "
                                f"<span style='color:{clr};font-family:monospace;'>{bar}</span> "
                                f"<span style='color:{clr};'>{v} {ck}</span>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("공포·탐욕 지수를 불러오지 못했습니다.")

            # CoinGecko 시장 데이터
            with st.expander("🌐 코인 시장 데이터 (CoinGecko)", expanded=True):
                if not coin_id:
                    st.info("이 코인의 CoinGecko 데이터가 없습니다.")
                else:
                    with st.spinner("CoinGecko 조회 중..."):
                        cg = fetch_coingecko(coin_id)
                    if not cg:
                        st.warning("CoinGecko API 응답 없음 (요청 한도 초과 시 잠시 후 재시도)")
                    else:
                        md = cg.get('market_data', {})
                        def cg_val(key, currency='usd'):
                            v = md.get(key, {})
                            return v.get(currency) if isinstance(v, dict) else v

                        cg1, cg2, cg3, cg4 = st.columns(4)
                        cg1.metric("시가총액 순위", f"#{cg.get('market_cap_rank','?')}")
                        mc = cg_val('market_cap', 'krw')
                        cg2.metric("시가총액", f"{mc/1e12:.2f}조원" if mc and mc > 1e12
                                   else (f"{mc/1e8:.0f}억원" if mc else "-"))
                        vol = cg_val('total_volume', 'krw')
                        cg3.metric("24h 거래량", f"{vol/1e8:.0f}억원" if vol else "-")
                        sup = md.get('circulating_supply')
                        cg4.metric("유통 공급량", f"{sup:,.0f}" if sup else "-")

                        st.markdown("**기간별 수익률**")
                        pr1, pr2, pr3, pr4, pr5 = st.columns(5)
                        for col, label, key in [
                            (pr1, "24h",  'price_change_percentage_24h'),
                            (pr2, "7일",  'price_change_percentage_7d'),
                            (pr3, "14일", 'price_change_percentage_14d'),
                            (pr4, "30일", 'price_change_percentage_30d'),
                            (pr5, "1년",  'price_change_percentage_1y'),
                        ]:
                            pct = md.get(key)
                            if pct is not None:
                                col.metric(label, f"{'+' if pct>=0 else ''}{pct:.1f}%")

                        # ATH / ATL
                        st.markdown("**역대 고점 / 저점**")
                        ah1, ah2 = st.columns(2)
                        ath     = cg_val('ath', 'krw')
                        ath_pct = md.get('ath_change_percentage', {}).get('krw')
                        atl     = cg_val('atl', 'krw')
                        atl_pct = md.get('atl_change_percentage', {}).get('krw')
                        ah1.metric("역대 최고가 (ATH)",
                                   f"{ath:,.0f}원" if ath else "-",
                                   f"{ath_pct:.1f}% from ATH" if ath_pct else None,
                                   delta_color="off")
                        ah2.metric("역대 최저가 (ATL)",
                                   f"{atl:,.0f}원" if atl else "-",
                                   f"+{atl_pct:.0f}% from ATL" if atl_pct else None)

    else:
        st.info("👈 사이드바에서 종목을 선택하고 **분석 시작**을 클릭하세요.")
        st.markdown("""
        ### 종합 매매 판단이란?
        종목을 분석해서 **지금 사야 할지 / 기다려야 할지 / 손절해야 할지**를 자동으로 판단해드립니다.

        **판단 기준**
        - 이동평균선 정배열/역배열
        - RSI 과매수/과매도
        - MACD 골든크로스/데드크로스
        - 볼린저밴드 위치
        - 거래량 변화
        - 주요 지지/저항선

        **결과 항목**
        - 종합 점수 (0~100) + 매수/관망/손절 판정
        - 물타기 가능 여부
        - 손절 기준선
        - 1차 목표가
        """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: 매매일지
# ════════════════════════════════════════════════════════════════════════════
with tab_journal:
    st.header("📓 매매일지")

    all_trades = load_trades()
    if not all_trades.empty:
        csv_buf = io.StringIO()
        all_trades.to_csv(csv_buf, index=False, encoding='utf-8-sig')
        st.download_button("📥 전체 거래 CSV 내보내기",
                           csv_buf.getvalue().encode('utf-8-sig'),
                           "매매일지.csv", "text/csv")

    j_tab1, j_tab2, j_tab3, j_tab4 = st.tabs(["➕ 거래 입력", "📋 보유 포지션", "📜 종료 거래", "📊 통계"])

    # ── 거래 입력 ────────────────────────────────────────────────────────────
    with j_tab1:
        st.subheader("새 거래 기록")
        c1, c2, c3 = st.columns(3)
        with c1:
            trade_date  = st.date_input("날짜", value=date.today(), key="j_date")
        with c2:
            symbol      = st.text_input("종목명", placeholder="예: 삼성전자, BTC", key="j_sym")
        with c3:
            market_type = st.radio("구분", ["주식", "코인"], horizontal=True, key="j_mkt")

        c4, c5, c6 = st.columns(3)
        with c4:
            entry_price = st.number_input("진입가", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="j_ep")
        with c5:
            quantity    = st.number_input("수량", min_value=0.0, value=0.0,
                                          step=0.0001 if market_type == "코인" else 1.0,
                                          format="%.4f", key="j_qty")
        with c6:
            investment  = entry_price * quantity
            unit        = "원" if market_type == "주식" else ""
            st.metric("투입금액 (자동계산)", f"{investment:,.0f}{unit}")

        c7, c8 = st.columns([1, 2])
        with c7:
            reason = st.selectbox("진입 근거",
                                  ["3파 초입", "2파 조정 매수", "5파 진행", "이평선 지지", "박스권 돌파", "기타"],
                                  key="j_rsn")
        with c8:
            memo = st.text_input("메모 (한 줄)", placeholder="추가 메모...", key="j_memo")

        c9, c10, c11 = st.columns(3)
        with c9:
            stop_loss = st.number_input("손절가 ★ 필수", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="j_sl")
        with c10:
            if entry_price > 0 and stop_loss > 0 and stop_loss < entry_price and quantity > 0:
                sl_loss = (entry_price - stop_loss) * quantity
                sl_pct  = (entry_price - stop_loss) / entry_price * 100
                st.markdown(
                    f"<div style='background:#3d0000;padding:12px;border-radius:8px;border-left:4px solid #ff4444;'>"
                    f"<div style='color:#ff6666;font-size:0.85em;'>손절 시 손실</div>"
                    f"<div style='color:#ff2222;font-size:1.5em;font-weight:bold;'>-{sl_loss:,.0f}{unit}</div>"
                    f"<div style='color:#ff4444;font-size:1.1em;font-weight:bold;'>-{sl_pct:.2f}%</div></div>",
                    unsafe_allow_html=True)
            elif stop_loss == 0:
                st.markdown(
                    "<div style='background:#1a0a00;padding:10px;border-radius:8px;border-left:4px solid #ff8800;color:#ff8800;'>⚠️ 손절가 미입력</div>",
                    unsafe_allow_html=True)
        with c11:
            target_price = st.number_input("목표가", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="j_tp")

        if entry_price > 0 and stop_loss > 0 and stop_loss < entry_price and target_price > entry_price:
            rr = (target_price - entry_price) / (entry_price - stop_loss)
            rr_color = "#00cc44" if rr >= 2.0 else ("#88cc00" if rr >= 1.5 else "#ffaa00")
            rr_bg    = "#002211" if rr >= 2.0 else ("#111f00" if rr >= 1.5 else "#1f1100")
            warning  = " ⚠️ 손익비 부족 — 진입 재고려" if rr < 1.5 else ""
            st.markdown(
                f"<div style='background:{rr_bg};padding:12px;border-radius:8px;border-left:4px solid {rr_color};margin-top:8px;'>"
                f"<span style='color:{rr_color};font-size:1.4em;font-weight:bold;'>손익비 {rr:.2f}</span>"
                f"<span style='color:{rr_color};'>{warning}</span></div>",
                unsafe_allow_html=True)

        st.markdown("---")
        if st.button("💾 거래 저장", type="primary", key="j_save"):
            errors = []
            if not symbol.strip():         errors.append("종목명을 입력해주세요.")
            if entry_price <= 0:           errors.append("진입가를 입력해주세요.")
            if quantity <= 0:              errors.append("수량을 입력해주세요.")
            if stop_loss <= 0:             errors.append("손절가는 필수 입력입니다.")
            elif stop_loss >= entry_price: errors.append("손절가는 진입가보다 낮아야 합니다.")
            if errors:
                for e in errors: st.error(e)
            else:
                save_trade({
                    'trade_date': trade_date.strftime('%Y-%m-%d'),
                    'symbol': symbol.strip(), 'market_type': market_type,
                    'entry_price': entry_price, 'quantity': quantity, 'investment': investment,
                    'reason': reason, 'memo': memo, 'stop_loss': stop_loss,
                    'target_price': target_price if target_price > 0 else None,
                    'status': '보유중',
                })
                st.success(f"✅ {symbol.strip()} 거래가 저장되었습니다!")
                st.rerun()

    # ── 보유 포지션 ──────────────────────────────────────────────────────────
    with j_tab2:
        st.subheader("📋 보유 중인 포지션")
        open_trades = load_trades(status='보유중')
        if open_trades.empty:
            st.info("보유 중인 포지션이 없습니다.")
        else:
            for _, row in open_trades.iterrows():
                unit = "원" if row['market_type'] == "주식" else ""
                rr_str = ""
                if row['target_price'] and row['stop_loss'] < row['entry_price']:
                    rr = (row['target_price'] - row['entry_price']) / (row['entry_price'] - row['stop_loss'])
                    rr_str = f" | 손익비 {rr:.2f}"
                with st.expander(f"#{row['id']} {row['symbol']} ({row['market_type']}) | "
                                 f"진입가 {row['entry_price']:,.2f}{unit} × {row['quantity']} | "
                                 f"{row['trade_date']}{rr_str}"):
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.write(f"**투입금액**: {row['investment']:,.0f}{unit}")
                        st.write(f"**진입 근거**: {row['reason']}")
                        st.write(f"**손절가**: {row['stop_loss']:,.2f}{unit}")
                        if row['target_price']:
                            st.write(f"**목표가**: {row['target_price']:,.2f}{unit}")
                        if row['memo']:
                            st.write(f"**메모**: {row['memo']}")
                    with ic2:
                        exit_price = st.number_input("청산가", min_value=0.0, value=0.0,
                                                     step=1.0, format="%.2f", key=f"ep_{row['id']}")
                        if exit_price > 0:
                            pnl     = (exit_price - row['entry_price']) * row['quantity']
                            pnl_pct = (exit_price - row['entry_price']) / row['entry_price'] * 100
                            clr     = "#00cc44" if pnl >= 0 else "#ff3333"
                            sign    = "+" if pnl >= 0 else ""
                            st.markdown(f"<div style='color:{clr};font-weight:bold;font-size:1.1em;'>"
                                        f"실현손익: {sign}{pnl:,.0f}{unit} ({sign}{pnl_pct:.2f}%)</div>",
                                        unsafe_allow_html=True)
                        exit_reason   = st.selectbox("청산 사유",
                                                     ["목표가 도달", "손절", "무효라인 이탈", "기타"],
                                                     key=f"er_{row['id']}")
                        stop_respected = st.checkbox("손절가를 지켰나요?", key=f"sr_{row['id']}")
                        review_memo    = st.text_area("복기 메모",
                                                      placeholder="계획대로 했나? 어디서 틀렸나?",
                                                      key=f"rv_{row['id']}", height=80)
                        if st.button("✅ 청산 완료", key=f"close_{row['id']}", type="primary"):
                            if exit_price <= 0:
                                st.error("청산가를 입력해주세요.")
                            else:
                                close_trade(row['id'], exit_price, exit_reason,
                                            int(stop_respected), review_memo)
                                st.success("청산 처리되었습니다!")
                                st.rerun()

    # ── 종료 거래 ────────────────────────────────────────────────────────────
    with j_tab3:
        st.subheader("📜 종료된 거래")
        closed_trades = load_trades(status='종료')
        if closed_trades.empty:
            st.info("종료된 거래가 없습니다.")
        else:
            disp = closed_trades[['trade_date','symbol','market_type','entry_price','exit_price',
                                  'quantity','realized_pnl','realized_pnl_pct','reason','exit_reason']].copy()
            disp.columns = ['날짜','종목','구분','진입가','청산가','수량','실현손익','손익률(%)','진입근거','청산사유']
            disp['손익률(%)'] = disp['손익률(%)'].round(2)
            disp['실현손익']   = disp['실현손익'].round(0)
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.markdown("---")
            st.subheader("📝 복기 메모")
            for _, row in closed_trades.iterrows():
                pnl_val = row['realized_pnl'] or 0
                pnl_pct = row['realized_pnl_pct'] or 0
                sign    = "+" if pnl_val >= 0 else ""
                with st.expander(f"#{row['id']} {row['symbol']} | "
                                 f"{sign}{pnl_val:,.0f} ({sign}{pnl_pct:.2f}%) | {row['trade_date']}"):
                    new_memo = st.text_area("복기 메모", value=row['review_memo'] or "",
                                            placeholder="계획대로 했나? 어디서 틀렸나?",
                                            key=f"crv_{row['id']}", height=100)
                    if st.button("저장", key=f"savememo_{row['id']}"):
                        update_review(row['id'], new_memo)
                        st.success("저장되었습니다.")
                        st.rerun()

    # ── 통계 ────────────────────────────────────────────────────────────────
    with j_tab4:
        st.subheader("📊 통계 대시보드")
        closed = load_trades(status='종료')
        if closed.empty:
            st.info("통계를 보려면 청산된 거래가 필요합니다.")
        else:
            closed['realized_pnl']     = pd.to_numeric(closed['realized_pnl'], errors='coerce').fillna(0)
            closed['realized_pnl_pct'] = pd.to_numeric(closed['realized_pnl_pct'], errors='coerce').fillna(0)
            wins   = closed[closed['realized_pnl'] > 0]
            losses = closed[closed['realized_pnl'] <= 0]
            total  = len(closed)
            win_rate       = len(wins) / total * 100
            avg_profit_pct = wins['realized_pnl_pct'].mean() if not wins.empty else 0
            avg_loss_pct   = losses['realized_pnl_pct'].mean() if not losses.empty else 0
            rr_stat        = abs(avg_profit_pct / avg_loss_pct) if avg_loss_pct != 0 else float('inf')

            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            kc1.metric("총 거래",     f"{total}건")
            kc2.metric("승률",        f"{win_rate:.1f}%")
            kc3.metric("평균 수익률",  f"+{avg_profit_pct:.2f}%")
            kc4.metric("평균 손실률",  f"{avg_loss_pct:.2f}%")
            kc5.metric("손익비",      f"{rr_stat:.2f}" if rr_stat != float('inf') else "∞")

            st.markdown("---")
            closed_sorted = closed.sort_values(['trade_date', 'id'])
            closed_sorted['cum_pnl'] = closed_sorted['realized_pnl'].cumsum()
            total_pnl = closed['realized_pnl'].sum()
            line_color = '#00cc44' if total_pnl >= 0 else '#ff3333'

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=list(range(1, len(closed_sorted) + 1)), y=closed_sorted['cum_pnl'],
                mode='lines+markers', line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor='rgba(0,204,68,0.1)' if total_pnl >= 0 else 'rgba(255,51,51,0.1)',
                customdata=list(zip(closed_sorted['symbol'], closed_sorted['trade_date'])),
                hovertemplate='%{customdata[0]} (%{customdata[1]})<br>누적: %{y:,.0f}원<extra></extra>',
            ))
            fig_pnl.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1)
            fig_pnl.update_layout(title="누적 실현손익 곡선", template='plotly_dark',
                                  height=300, xaxis_title="거래 순번", yaxis_title="누적 손익 (원)",
                                  margin=dict(t=50, b=40))
            st.plotly_chart(fig_pnl, use_container_width=True)

            st.subheader("📈 진입 근거별 성과")
            reason_stats = (
                closed.groupby('reason')
                .apply(lambda x: pd.Series({
                    '거래수': len(x),
                    '승': int((x['realized_pnl'] > 0).sum()),
                    '패': int((x['realized_pnl'] <= 0).sum()),
                    '승률(%)': round((x['realized_pnl'] > 0).mean() * 100, 1),
                    '평균손익(%)': round(x['realized_pnl_pct'].mean(), 2),
                    '총손익': round(x['realized_pnl'].sum(), 0),
                }))
                .reset_index().sort_values('승률(%)', ascending=False)
            )
            st.dataframe(reason_stats, use_container_width=True, hide_index=True)

            st.subheader("🔒 손절 준수 여부별 성과")
            comp_rows = []
            for label, flag in [("손절 준수", 1), ("손절 미준수", 0)]:
                subset = closed[closed['stop_respected'] == flag]
                if not subset.empty:
                    comp_rows.append({
                        '구분': label, '거래수': len(subset),
                        '승률(%)': round((subset['realized_pnl'] > 0).mean() * 100, 1),
                        '평균손익(%)': round(subset['realized_pnl_pct'].mean(), 2),
                        '총손익': round(subset['realized_pnl'].sum(), 0),
                    })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=closed['realized_pnl_pct'], nbinsx=20,
                marker_color=[('#00cc44' if v > 0 else '#ff3333') for v in closed['realized_pnl_pct']],
            ))
            fig_dist.add_vline(x=0, line_dash='dot', line_color='white')
            fig_dist.update_layout(title="손익률 분포", template='plotly_dark',
                                   height=260, xaxis_title="손익률 (%)", yaxis_title="거래 수",
                                   margin=dict(t=50, b=40))
            st.plotly_chart(fig_dist, use_container_width=True)

# ── 하단 고지 ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ 자동 카운팅은 하나의 해석일 뿐이며 투자 권유가 아닙니다. 모든 투자 결정은 본인 책임 하에 이루어져야 합니다.")
