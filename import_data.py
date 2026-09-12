"""
import_data.py
----------------
นำเข้าข้อมูลจากไฟล์ Dataset ทั้ง 5 ไฟล์ (งบการเงิน, ราคาหุ้นรายวัน, ความเสี่ยง)
เข้า SQLite (cis_database.db) เพื่อให้ calculate_scores.py และ app.py ใช้งานต่อ

ตาราง (tables) ที่จะถูกสร้าง:
- stock_financials      : งบการเงินปี 2023-2025 ของ 8 หุ้นเป้าหมาย (ตัวเลขถูกแปลงเป็น float แล้ว)
- stock_daily_prices    : ราคาหุ้นรายวัน + technical indicators ปี 2023-2025
- stock_risk_static     : Beta / Volatility / Max Drawdown รายหุ้น จาก stock_risk_metrics.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

DB_NAME = 'cis_database.db'
TARGET_STOCKS = ['ADVANC', 'CCET', 'DELTA', 'HANA', 'JMART', 'KCE', 'THCOM', 'TRUE']
TARGET_YEARS = [2023, 2024, 2025]

SEARCH_DIRS = ['.', 'Dataset', 'Dataset/train_test', 'dataset', 'dataset/train_test']


def find_file(pattern):
    """ค้นหาไฟล์ตาม pattern ในหลาย ๆ โฟลเดอร์ที่เป็นไปได้"""
    for d in SEARCH_DIRS:
        matches = glob.glob(os.path.join(d, pattern))
        if matches:
            return sorted(matches)[0]
    return None


def find_all_files(pattern):
    found = []
    for d in SEARCH_DIRS:
        found.extend(glob.glob(os.path.join(d, pattern)))
    # unique, keep order
    seen = set()
    out = []
    for f in found:
        rp = os.path.abspath(f)
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def read_csv_smart(path):
    """อ่าน CSV โดยลองหลาย encoding (ไฟล์งบการเงินเป็น cp874 เพราะมีหัวคอลัมน์ภาษาไทย)"""
    for enc in ['utf-8', 'cp874', 'cp1252', 'latin1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # last resort
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def clean_numeric_series(s):
    """แปลงคอลัมน์ตัวเลขที่มี comma / % / ช่องว่าง ให้เป็น float"""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (
        s.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.strip()
        .replace({'': None, '-': None, 'nan': None, 'None': None})
        .astype(float)
    )


def import_financial_data(engine):
    print("\n--- 1. กำลังประมวลผลไฟล์งบการเงิน (Financials) ---")

    # ไฟล์หลัก master_all_8_stocks_financials + ไฟล์ train/test (โครงสร้างคอลัมน์เดียวกัน)
    # ใช้ master เป็นหลักเพราะครอบคลุมทั้ง 2023-2025 อยู่แล้ว แต่รวมทุกไฟล์ที่เจอไว้กันตกหล่น
    candidate_files = find_all_files("*master_all_8_stocks_financials*.csv")
    train_test_files = find_all_files("*financials_train*.csv") + find_all_files("*financials_test*.csv")

    if not candidate_files and not train_test_files:
        print("❌ ไม่พบไฟล์งบการเงิน")
        return

    frames = []
    for fp in (candidate_files or train_test_files):
        try:
            frames.append(read_csv_smart(fp))
        except Exception as e:
            print(f"⚠️ อ่านไฟล์ {fp} ไม่ได้: {e}")

    if not frames:
        print("❌ ไม่สามารถอ่านไฟล์งบการเงินได้เลย")
        return

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()

    stock_col = [c for c in df.columns if 'Stock' in c][0]
    # หาปี ค.ศ. โดยตรงก่อน ถ้าไม่เจอค่อย fallback ไปแปลงจาก พ.ศ.
    year_ad_col = [c for c in df.columns if 'ค.ศ' in c or c.lower().strip() == 'year']
    year_be_col = [c for c in df.columns if 'พ.ศ' in c]

    df['stock_symbol'] = df[stock_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)

    def clean_year_generic(val):
        try:
            y = int(float(str(val).split('.')[0].strip()))
            return y - 543 if y > 2500 else y
        except Exception:
            return None

    if year_ad_col:
        df['year_clean'] = df[year_ad_col[0]].apply(clean_year_generic)
    elif year_be_col:
        df['year_clean'] = df[year_be_col[0]].apply(clean_year_generic)
    else:
        print("❌ ไม่พบคอลัมน์ปีในไฟล์งบการเงิน")
        return

    df = df.drop_duplicates(subset=['stock_symbol', 'year_clean'], keep='first')

    filtered_df = df[
        (df['stock_symbol'].isin(TARGET_STOCKS)) &
        (df['year_clean'].isin(TARGET_YEARS))
    ].copy()

    # แปลงคอลัมน์ตัวเลข (ที่มี comma/% ปนอยู่) ให้เป็น float ทั้งหมด
    non_numeric_cols = {stock_col, 'stock_symbol'}
    if year_ad_col:
        non_numeric_cols.add(year_ad_col[0])
    if year_be_col:
        non_numeric_cols.add(year_be_col[0])

    for col in filtered_df.columns:
        if col in non_numeric_cols or col == 'year_clean':
            continue
        try:
            filtered_df[col] = clean_numeric_series(filtered_df[col])
        except Exception:
            pass  # คอลัมน์ที่แปลงไม่ได้ ปล่อยไว้เป็น string เดิม

    rename_mapping = {
        'stock_symbol': 'ticker',
        'year_clean': 'year',
        'Total Revenue': 'total_revenue',
        'Operating Revenue': 'operating_revenue',
        'COGS': 'cogs',
        'Gross Profit': 'gross_profit',
        'EBIT': 'ebit',
        'EBITDA': 'ebitda',
        'Gross Margin (%)': 'gross_margin',
        'Operating Margin (%)': 'operating_margin',
        'EBITDA Margin (%)': 'ebitda_margin',
        'Net Margin (%)': 'net_margin',
        'ROE (%)': 'roe',
        'ROA (%)': 'roa',
        'D/E (x)': 'de_ratio',
        'Current Ratio (x)': 'current_ratio',
        'Quick Ratio (x)': 'quick_ratio',
        'Interest Coverage (x)': 'interest_coverage',
        'OCF_to_NI': 'ocf_to_ni',
        'Free Cash Flow': 'free_cash_flow',
        'Operating CF': 'operating_cash_flow',
        'Capital Expenditure': 'capex',
        'Total Assets': 'total_assets',
        'Total Equity': 'total_equity',
        'Total Liabilities': 'total_liabilities',
        'Current Assets': 'current_assets',
        'Current Liabilities': 'current_liabilities',
        'Cash Equivalent': 'cash_and_equivalents',
        'Inventory': 'inventory',
        'Accounts Receivable': 'accounts_receivable',
        'Accounts Payable': 'accounts_payable',
        'Fixed Assets': 'fixed_assets',
        'INT': 'interest_expense',
        'EPS': 'eps',
        'NI (Parent)': 'net_income',
    }

    save_df = filtered_df.rename(columns=rename_mapping)
    keep_cols = ['ticker', 'year'] + [v for v in rename_mapping.values() if v not in ('ticker', 'year') and v in save_df.columns]
    save_df = save_df[keep_cols].sort_values(['ticker', 'year']).reset_index(drop=True)

    save_df.to_sql('stock_financials', con=engine, if_exists='replace', index=False)
    print(f"✅ นำเข้าข้อมูลงบการเงินสำเร็จ: {len(save_df)} แถว "
          f"(ครอบคลุม {save_df['ticker'].nunique()} หุ้น, ปี {sorted(save_df['year'].unique())})")


def compute_technical_indicators(df):
    """คำนวณ EMA20, EMA50, RSI14, MACD, ADX, Volume Avg จากราคา OHLCV ดิบ (รายหุ้นเดียว, เรียงตามวันที่แล้ว)
    ใช้กรณีไฟล์ราคาต้นฉบับไม่มี Technical Indicators สำเร็จรูปมาให้ (เช่น ไฟล์ราคาดิบจาก SET/Settrade)
    สูตรมาตรฐาน: EMA (Exponential Moving Average), RSI14 แบบ Wilder's Smoothing,
    MACD = EMA12 - EMA26 (Gerald Appel), ADX14 แบบ Wilder's Directional Movement Index"""
    df = df.sort_values('date').reset_index(drop=True).copy()

    # EMA20 / EMA50
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    # MACD Line (EMA12 - EMA26)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # RSI14 (Wilder's Smoothing - สูตรดั้งเดิมมาตรฐาน)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['RSI14'] = df['RSI14'].fillna(50.0)  # ช่วงข้อมูลไม่พอ (warm-up) ให้เป็นค่ากลาง

    # ADX14 (Wilder's Directional Movement Index)
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df['ADX'] = df['ADX'].fillna(20.0)  # ค่ากลางช่วง warm-up

    # Volume Avg (ปริมาณซื้อขายเฉลี่ยย้อนหลัง 20 วัน)
    df['Volume Avg'] = df['volume'].rolling(window=20, min_periods=1).mean()

    return df


def import_price_data(engine):
    print("\n--- 2. กำลังประมวลผลไฟล์ราคาและดัชนีเทคนิคอล (Stock Prices) ---")
    # รองรับทั้งไฟล์เดิม (stock_cleaned_*) และไฟล์ใหม่ที่ผู้ใช้อัปโหลด (เช่น set_tech_stocks_combined_*)
    file_path = (find_file("*stock_cleaned*.csv")
                 or find_file("*set_tech_stocks*.csv")
                 or find_file("*combined*.csv"))
    if not file_path:
        print("❌ ไม่พบไฟล์ราคาหุ้น (รองรับชื่อไฟล์: *stock_cleaned*.csv, *set_tech_stocks*.csv, *combined*.csv)")
        return

    df = read_csv_smart(file_path)
    df.columns = df.columns.str.strip()

    # ปรับชื่อคอลัมน์ราคาให้เป็นตัวพิมพ์เล็กมาตรฐาน (รองรับไฟล์ใหม่ที่ใช้ Capitalized เช่น Open/High/Low/Close/Volume)
    standard_price_cols = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    rename_map = {c: standard_price_cols[c.strip().lower()] for c in df.columns
                  if c.strip().lower() in standard_price_cols and c != standard_price_cols[c.strip().lower()]}
    df = df.rename(columns=rename_map)

    # ตัดคอลัมน์ Adj Close ทิ้งเสมอ - ใช้ Close ดิบเท่านั้น เพื่อให้ตรงกับราคาที่ SET แสดงจริง (ไม่ใช่ราคาที่ปรับปันผลแล้ว)
    adj_cols = [c for c in df.columns if c.strip().lower() in ['adj close', 'adjclose', 'adjusted close']]
    if adj_cols:
        df = df.drop(columns=adj_cols)
        print(f"ℹ️ ตัดคอลัมน์ {adj_cols} ทิ้ง (ใช้ราคา Close ดิบเท่านั้น ไม่ใช่ Adjusted Close)")

    ticker_col = [c for c in df.columns if c.lower() in ['ticker', 'stock', 'symbol']][0]
    df['clean_ticker'] = df[ticker_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)

    date_col = [c for c in df.columns if c.lower() in ['date', 'datetime', 'วันที่']][0]
    # รองรับรูปแบบวันที่สากล YYYY-MM-DD (ไฟล์ใหม่) และ M/D/YYYY (ไฟล์เดิม)
    df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)

    if df['parsed_date'].notna().any() and df['parsed_date'].dt.year.max() > 2500:
        df['parsed_date'] = df['parsed_date'].apply(
            lambda x: x.replace(year=x.year - 543) if pd.notnull(x) and x.year > 2500 else x
        )

    mask_stock = df['clean_ticker'].isin(TARGET_STOCKS)
    mask_date = (df['parsed_date'] >= '2023-01-01') & (df['parsed_date'] <= '2025-12-31')
    filtered_df = df[mask_stock & mask_date].copy()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in filtered_df.columns:
            filtered_df[col] = clean_numeric_series(filtered_df[col])

    filtered_df = filtered_df.sort_values(by=['clean_ticker', 'parsed_date']).reset_index(drop=True)
    filtered_df['date'] = filtered_df['parsed_date']
    filtered_df['ticker'] = filtered_df['clean_ticker']

    # --- Technical Indicators: ใช้ค่าที่มีอยู่แล้วถ้าไฟล์ต้นฉบับมีมาให้ ไม่งั้นคำนวณเอง ---
    indicator_cols = ['EMA20', 'EMA50', 'RSI14', 'MACD', 'ADX', 'Volume Avg']
    has_all_indicators = all(c in filtered_df.columns for c in indicator_cols)

    if has_all_indicators:
        for col in indicator_cols:
            filtered_df[col] = clean_numeric_series(filtered_df[col])
        print("ℹ️ พบคอลัมน์ Technical Indicators ในไฟล์ต้นฉบับแล้ว ใช้ค่าที่มีโดยตรง")
    else:
        print("ℹ️ ไฟล์ราคาไม่มี Technical Indicators สำเร็จรูป - กำลังคำนวณ EMA20/EMA50/RSI14/MACD/ADX/Volume Avg เองจากราคา OHLCV จริง")
        processed = []
        for t in TARGET_STOCKS:
            sub = filtered_df[filtered_df['ticker'] == t].copy()
            if sub.empty:
                continue
            processed.append(compute_technical_indicators(sub))
        filtered_df = pd.concat(processed, ignore_index=True) if processed else filtered_df

    filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')

    drop_cols = ['clean_ticker', 'parsed_date']
    if ticker_col != 'ticker':
        drop_cols.append(ticker_col)
    if date_col != 'date':
        drop_cols.append(date_col)
    filtered_df = filtered_df.drop(columns=drop_cols, errors='ignore')

    filtered_df.to_sql('stock_daily_prices', con=engine, if_exists='replace', index=False)

    if len(filtered_df) > 0:
        print(f"✅ นำเข้าข้อมูลราคาหุ้นสำเร็จ: {len(filtered_df)} แถว "
              f"(ครอบคลุม {filtered_df['ticker'].nunique()} หุ้น, "
              f"วันที่ {filtered_df['date'].min()} ถึง {filtered_df['date'].max()}) "
              f"[แหล่งไฟล์: {os.path.basename(file_path)}]")
    else:
        print("⚠️ ไม่พบแถวที่ตรงตามเงื่อนไข ลองตรวจตัวอย่างข้อมูล:")
        print("ตัวอย่าง Ticker ในไฟล์:", df[ticker_col].unique()[:5])
        print("ตัวอย่าง Date ในไฟล์:", df[date_col].head(3).tolist())


def import_risk_metrics(engine):
    print("\n--- 3. กำลังประมวลผลไฟล์ Risk Metrics (Beta / Volatility / Max Drawdown) ---")
    file_path = find_file("*stock_risk_metrics*.csv")
    if not file_path:
        print("❌ ไม่พบไฟล์ stock_risk_metrics")
        return

    df = read_csv_smart(file_path)
    df.columns = df.columns.str.strip()

    stock_col = [c for c in df.columns if c.lower() in ['stock', 'ticker', 'symbol']][0]
    df['ticker'] = df[stock_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)

    beta_col = [c for c in df.columns if 'beta' in c.lower()][0]
    vol_col = [c for c in df.columns if 'volatility' in c.lower()][0]
    dd_col = [c for c in df.columns if 'drawdown' in c.lower()][0]

    out = pd.DataFrame({
        'ticker': df['ticker'],
        'beta': clean_numeric_series(df[beta_col]),
        'volatility_pct': clean_numeric_series(df[vol_col]),
        'max_drawdown_pct': clean_numeric_series(df[dd_col]),
    })
    out = out[out['ticker'].isin(TARGET_STOCKS)].drop_duplicates(subset=['ticker']).reset_index(drop=True)

    out.to_sql('stock_risk_static', con=engine, if_exists='replace', index=False)
    print(f"✅ นำเข้าข้อมูลความเสี่ยงสำเร็จ: {len(out)} หุ้น")


if __name__ == '__main__':
    engine = create_engine(f'sqlite:///{DB_NAME}')
    import_financial_data(engine)
    import_price_data(engine)
    import_risk_metrics(engine)
    print("\n" + "=" * 60)
    print("นำเข้าข้อมูลทั้งหมดเสร็จสมบูรณ์ -> รันต่อด้วย: python calculate_scores.py")
