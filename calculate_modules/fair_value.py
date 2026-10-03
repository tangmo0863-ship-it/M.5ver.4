"""
calculate_modules/fair_value.py — v4 (แก้ "ตัวเลขต้องถูกก่อน")
========================================================================
สูตรคำนวณโมดูล "Fair Value" (⚖️) — key ใน DATA CONTRACT เดิมครบทุกตัว เพิ่ม key ใหม่:
    net_debt_used, debt_basis, debt_excluded_payables

=== CHANGELOG v4 ===
- [FIX-F1] หนี้สุทธิที่หักออกจากมูลค่ากิจการ (Enterprise Value) เดิมใช้ total_liabilities - cash
  ซึ่งรวม "เจ้าหนี้การค้า" (หนี้จากการซื้อของ ไม่มีดอกเบี้ย และถูกนับไปแล้วใน Operating Cash Flow)
  เข้าไปเป็นหนี้ด้วย ทำให้ Fair Value ต่ำเกินจริง — หนักสุดที่ DELTA (เจ้าหนี้การค้า = 79% ของหนี้สินรวม),
  HANA 65%, CCET 57%, KCE 54%
  ตอนนี้: net_debt = total_liabilities - accounts_payable - cash
  ข้อจำกัดที่ยังเหลือ: หนี้สินไม่มีดอกเบี้ยอื่น (ค่าใช้จ่ายค้างจ่าย, รายได้รับล่วงหน้า, ประมาณการหนี้สิน)
  ยังถูกนับเป็นหนี้อยู่ เพราะ Dataset ไม่แยกคอลัมน์หนี้ที่มีดอกเบี้ย → ผลยังอนุรักษนิยม (ประเมินต่ำไว้ก่อน)
  ส่วนหนี้สินตามสัญญาเช่า/ค่าใบอนุญาตคลื่นของ ADVANC/TRUE ยังนับเป็นหนี้ ซึ่งถูกต้องตามหลัก
  ถ้าไม่มีคอลัมน์ accounts_payable → ใช้วิธีเดิมและติด debt_basis = 'total_liabilities'
- [FIX-F2] เลิก clip คะแนนย่อยที่ 25-95 และ safety_score ที่ 10-98 → ใช้ช่วง 0-100
  (การหนีบราคาประเมินไว้ 65%-185% ของราคาตลาดตอนแปลงเป็นคะแนนยังคงไว้ตามดีไซน์เดิม)
- [FIX-F3] หุ้นที่ "ไม่มีข้อมูล" (NO_DATA เช่น ไม่มีงบ/ไม่มีราคา/ไม่มีจำนวนหุ้น) เดิมได้ valuation_score = 0.0
  ซึ่งถูกนับเป็น "แพงที่สุด" ในคะแนนรวม → ตอนนี้เป็น None (ไม่นับในคะแนนรวม) เพราะไม่ใช่ความผิดของบริษัท
  ส่วนหุ้นที่ขาดทุน ยังคงกติกาเดิมแต่เขียนให้ชัดเจน: NOT_RATED (EPS และ FCF ติดลบ) = 0 คะแนน และ PARTIAL
  (คำนวณได้โมเดลเดียว) = คะแนนโมเดลที่มี x น้ำหนักของมัน (โมเดลที่หายไปนับ 0) เพราะการขาดทุนเป็นสัญญาณลบจริง
  (ทดสอบแล้ว: ถ้าถ่วงน้ำหนักใหม่ HANA/JMART ที่ขาดทุนจะได้ 92/100 จาก DCF ตัวเดียว ซึ่งชวนเข้าใจผิดกว่า)
"""

import numpy as np
import pandas as pd
from calculate_modules.common import clean_float, SECTOR_MAP

SHARES_OUTSTANDING = {
    'ADVANC': 2974209736,
    'CCET':   10400000000,
    'DELTA':  12473816149,
    'HANA':   885366460,
    'JMART':  1458518485,
    'KCE':    1182429485,
    'THCOM':  1096000000,
    'TRUE':   34552100801
}

STOCK_VALUATION_PARAMS = {
    'ADVANC': {'target_pe': 22.0, 'wacc': 0.078, 'terminal_g': 0.020, 'group': 'Technology & Telecommunication'},
    'TRUE':   {'target_pe': 22.0, 'wacc': 0.078, 'terminal_g': 0.020, 'group': 'Technology & Telecommunication'},
    'THCOM':  {'target_pe': 22.0, 'wacc': 0.078, 'terminal_g': 0.020, 'group': 'Technology & Telecommunication'},
    'DELTA':  {'target_pe': 18.0, 'wacc': 0.088, 'terminal_g': 0.015, 'group': 'Electronic Components'},
    'HANA':   {'target_pe': 18.0, 'wacc': 0.088, 'terminal_g': 0.015, 'group': 'Electronic Components'},
    'KCE':    {'target_pe': 18.0, 'wacc': 0.088, 'terminal_g': 0.015, 'group': 'Electronic Components'},
    'CCET':   {'target_pe': 18.0, 'wacc': 0.088, 'terminal_g': 0.015, 'group': 'Electronic Components'},
    'JMART':  {'target_pe': 22.0, 'wacc': 0.092, 'terminal_g': 0.025, 'group': 'Commerce & Technology'},
}

SECTOR_WACC = {'Technology & Telecomm': 0.078, 'Technology & Telecommunication': 0.078,
               'Electronic Components': 0.088, 'Commerce & Technology': 0.092}
SECTOR_TERMINAL_G = {'Technology & Telecomm': 0.020, 'Technology & Telecommunication': 0.020,
                     'Electronic Components': 0.015, 'Commerce & Technology': 0.025}
SECTOR_TARGET_PE = {'Technology & Telecomm': 22.0, 'Technology & Telecommunication': 22.0,
                    'Electronic Components': 18.0, 'Commerce & Technology': 22.0}

DEFAULT_WACC = 0.082
DEFAULT_TERMINAL_G = 0.020
DEFAULT_TARGET_PE = 20.0
NEAR_TERM_GROWTH_PREMIUM = 0.015
DCF_WEIGHT, PE_WEIGHT = 0.55, 0.45

VALUATION_METHODOLOGY_NOTE = (
    "WACC และ Target P/E อ้างอิงตาม Sector Baseline Benchmark 3 กลุ่มอุตสาหกรรมใน SET: "
    "กลุ่ม Tech & Telecom (WACC 7.8%, Target P/E 22.0x, g 2.0%), "
    "กลุ่ม Electronic Components (WACC 8.8%, Target P/E 18.0x, g 1.5%), "
    "และกลุ่ม Commerce & Tech (WACC 9.2%, Target P/E 22.0x, g 2.5%) "
    "หนี้สุทธิ = หนี้สินรวม - เจ้าหนี้การค้า - เงินสด (Dataset ไม่แยกหนี้ที่มีดอกเบี้ย จึงยังเป็นค่าประมาณแบบอนุรักษนิยม)"
)

_EMPTY_KEYS = [
    'valuation_score', 'fair_value', 'dcf_fair_value', 'pe_fair_value', 'margin_of_safety',
    'pe_ratio', 'pb_ratio', 'market_cap_mb', 'eps',
    'wacc', 'wacc_used', 'wacc_str', 'terminal', 'terminal_growth', 'terminal_growth_used', 'terminal_str',
    'target', 'target_pe', 'target_pe_used', 'target_pe_str', 'target_pb',
    'fcf_growth', 'fcf_growth_assumed', 'fcf_growth_yr1', 'fcf_base', 'fcf_base_used',
    'raw_dcf_fair_value', 'raw_pe_fair_value', 'dcf_fair_value_raw', 'pe_fair_value_raw',
    'dcf_is_clipped', 'pe_is_clipped', 'is_clipped',
    'pe_score', 'dcf_score', 'intrinsic_score', 'relative_score', 'safety_score',
    'dcf_is_estimate', 'pe_is_estimate', 'book_value_per_share', 'shares_outstanding',
    'net_debt_used', 'debt_basis', 'debt_excluded_payables',
]


def _empty_output(message):
    out = {k: None for k in _EMPTY_KEYS}
    out['valuation_score'] = None          # FIX-F3: ไม่มีข้อมูล = ไม่ให้คะแนน (เดิม 0.0)
    out['valuation_status'] = 'NO_DATA'
    out['confidence_level'] = 'Low'
    out['valuation_methodology_note'] = VALUATION_METHODOLOGY_NOTE
    out['warning_message'] = message
    return out


def calc_sub_score(raw_fair_value, current_price):
    """คะแนนย่อย 0-100 ของแต่ละโมเดล (DCF หรือ P/E)
    คืน (score, is_clipped) — score = None ถ้าโมเดลคำนวณไม่ได้"""
    if raw_fair_value is None or current_price is None or current_price <= 0:
        return None, False
    lower_bound = current_price * 0.65
    upper_bound = current_price * 1.85
    fair_for_score = float(np.clip(raw_fair_value, lower_bound, upper_bound))
    is_clipped = bool(raw_fair_value < lower_bound - 1e-4 or raw_fair_value > upper_bound + 1e-4)
    margin = (fair_for_score - current_price) / fair_for_score * 100 if fair_for_score > 0 else 0.0
    score = round(float(np.clip((margin + 20) * 1.4, 0.0, 100.0)), 1)   # FIX-F2 (เดิม 25-95)
    return score, is_clipped


def _net_debt(r):
    """FIX-F1: หนี้สุทธิไม่รวมเจ้าหนี้การค้า"""
    total_liab = clean_float(r.get('total_liabilities'), default=0.0)
    cash = clean_float(r.get('cash_and_equivalents'), default=0.0)
    ap = clean_float(r.get('accounts_payable'), default=None)
    if ap is not None and 0 <= ap <= total_liab:
        return total_liab - ap - cash, 'total_liabilities_less_payables', ap
    return total_liab - cash, 'total_liabilities', 0.0


def calculate_valuation_module(df_fin_ticker, current_price, ticker, beta=None, **kwargs):
    """Module 2: Fair Value (DCF + Relative P/E) — v4"""
    if df_fin_ticker is None or df_fin_ticker.empty or current_price is None or current_price <= 0:
        return _empty_output('ไม่มีข้อมูลงบการเงิน หรือราคาหุ้นไม่ถูกต้อง ไม่สามารถประเมินมูลค่าได้')

    fin_sorted = df_fin_ticker.sort_values(by='year')
    r = fin_sorted.iloc[-1]

    ticker_clean = str(ticker).replace('.BK', '').strip().upper() if ticker else ''
    param = STOCK_VALUATION_PARAMS.get(ticker_clean)
    sector = SECTOR_MAP.get(ticker_clean, '')
    if param:
        wacc, g, target_pe = param['wacc'], param['terminal_g'], param['target_pe']
    else:
        wacc = SECTOR_WACC.get(sector, DEFAULT_WACC)
        g = SECTOR_TERMINAL_G.get(sector, DEFAULT_TERMINAL_G)
        target_pe = SECTOR_TARGET_PE.get(sector, DEFAULT_TARGET_PE)
    near_term_growth = g + NEAR_TERM_GROWTH_PREMIUM

    shares = SHARES_OUTSTANDING.get(ticker_clean)
    if not shares:
        return _empty_output(f'ไม่มีจำนวนหุ้นจดทะเบียนของ {ticker_clean} ไม่สามารถคำนวณมูลค่าต่อหุ้นได้')

    net_inc = clean_float(r.get('net_income'), default=0.0)
    eps = clean_float(r.get('eps'), default=0.0)

    fcf_hist = fin_sorted['free_cash_flow'].apply(clean_float).tail(2)
    fcf_base = float(fcf_hist.mean()) if len(fcf_hist) > 0 else (net_inc * 0.75)

    net_debt, debt_basis, ap_excluded = _net_debt(r)          # FIX-F1

    if fcf_base > 0 and (wacc - g) > 0:
        enterprise_value = (fcf_base * (1 + near_term_growth)) / (wacc - g)
        dcf_equity = enterprise_value - net_debt
        raw_dcf_fair_value = (dcf_equity / shares) if dcf_equity > 0 else None
    else:
        raw_dcf_fair_value = None

    raw_pe_fair_value = eps * target_pe if eps > 0 else None

    dcf_score, dcf_is_clipped = calc_sub_score(raw_dcf_fair_value, current_price)
    pe_score, pe_is_clipped = calc_sub_score(raw_pe_fair_value, current_price)
    is_clipped = bool(dcf_is_clipped or pe_is_clipped)

    fair_components = [(raw_dcf_fair_value, dcf_score, DCF_WEIGHT), (raw_pe_fair_value, pe_score, PE_WEIGHT)]
    valid = [(v, s, w) for v, s, w in fair_components if v is not None]

    if not valid:
        blended_fair, mos, val_score = None, None, 0.0     # NOT_RATED: ขาดทุนทั้ง EPS และ FCF
        valuation_status = 'NOT_RATED'
    else:
        total_w = sum(w for _, _, w in valid)
        blended_fair = round(sum(v * w for v, _, w in valid) / total_w, 2)
        mos = round((blended_fair - current_price) / blended_fair * 100, 1)
        # FIX-F3: โมเดลที่คำนวณไม่ได้เพราะขาดทุนนับ 0 คะแนน (ไม่หารด้วย total_w)
        val_score = round(sum(s * w for _, s, w in valid), 1)
        valuation_status = 'OK' if len(valid) == 2 else 'PARTIAL'

    is_profitable = (eps > 0 and fcf_base > 0)
    if (not is_profitable) or valuation_status in ('NOT_RATED', 'NO_DATA'):
        confidence_level = 'Low'
    elif is_clipped:
        confidence_level = 'Medium'
    elif mos is not None and mos > 15:
        confidence_level = 'High'
    else:
        confidence_level = 'Medium'

    warning_message = None
    if valuation_status == 'NOT_RATED':
        warning_message = 'ไม่สามารถประเมินมูลค่าได้เนื่องจากบริษัทมีผลการดำเนินงานขาดทุน (EPS และ/หรือ FCF ติดลบ)'
    elif valuation_status == 'PARTIAL':
        warning_message = 'ประเมินได้เพียงโมเดลเดียว (อีกโมเดลขาดทุนหรือไม่สามารถคำนวณได้) ควรใช้ความระมัดระวังเป็นพิเศษ'
    elif is_clipped:
        warning_message = 'ราคาประเมินดิบชนหรือทะลุกรอบปกติ (65%-185% ของราคาตลาด) ตัวเลข Fair Value ที่แสดงเป็นค่าดิบแท้จริง'

    pe_ratio_now = round(float(current_price / eps), 2) if eps > 0 else None
    book_value_per_share = clean_float(r.get('total_equity'), 0.0) / shares
    pb_ratio_now = round(float(current_price / book_value_per_share), 2) if book_value_per_share > 0 else None
    market_cap = round(current_price * shares / 1e6, 1)
    target_pb = 1.50

    safety_score = (round(float(np.clip(50.0 + mos * 1.2, 0.0, 100.0)), 1)    # FIX-F2 (เดิม 10-98)
                    if mos is not None else None)

    wacc_pct, g_pct, fcf_g_pct = round(wacc * 100, 1), round(g * 100, 1), round(near_term_growth * 100, 1)
    dcf_r = round(raw_dcf_fair_value, 2) if raw_dcf_fair_value is not None else None
    pe_r = round(raw_pe_fair_value, 2) if raw_pe_fair_value is not None else None

    return {
        'valuation_score': val_score,
        'fair_value': blended_fair,
        'dcf_fair_value': dcf_r, 'pe_fair_value': pe_r,
        'raw_dcf_fair_value': dcf_r, 'raw_pe_fair_value': pe_r,
        'dcf_fair_value_raw': dcf_r, 'pe_fair_value_raw': pe_r,
        'margin_of_safety': mos,
        'pe_ratio': pe_ratio_now,
        'pb_ratio': pb_ratio_now,
        'market_cap_mb': market_cap,
        'eps': round(eps, 2),

        'wacc': wacc_pct, 'wacc_used': round(wacc * 100, 2), 'wacc_pct': wacc_pct,
        'wacc_rate': wacc, 'wacc_str': f"{wacc_pct:.1f}%",

        'terminal': g_pct, 'terminal_growth': g_pct, 'terminal_growth_used': round(g * 100, 2),
        'terminal_g': g_pct, 'terminal_g_used': round(g * 100, 2), 'terminal_growth_rate': g,
        'g': g_pct, 'g_used': round(g * 100, 2),
        'terminal_str': f"{g_pct:.1f}%", 'terminal_growth_str': f"{g_pct:.1f}%",

        'target': target_pe, 'target_pe': target_pe, 'target_pe_used': target_pe, 'pe_target': target_pe,
        'pe_target_used': target_pe, 'target_p_e': target_pe, 'target_pe_ratio': target_pe,
        'target_pe_str': f"{target_pe:.1f}x", 'target_pb': target_pb,

        'fcf_growth': fcf_g_pct, 'fcf_growth_assumed': round(near_term_growth * 100, 2),
        'fcf_growth_yr1': fcf_g_pct, 'fcf_growth_y1': fcf_g_pct, 'fcf_growth_1': fcf_g_pct,
        'near_term_growth': fcf_g_pct, 'fcf_growth_str': f"{fcf_g_pct:.1f}%",
        'fcf_base': round(fcf_base, 1), 'fcf_base_used': round(fcf_base, 1),

        'dcf_is_clipped': bool(dcf_is_clipped), 'pe_is_clipped': bool(pe_is_clipped), 'is_clipped': is_clipped,
        'pe_score': pe_score, 'dcf_score': dcf_score,
        'intrinsic_score': dcf_score, 'relative_score': pe_score,
        'safety_score': safety_score,

        'dcf_is_estimate': (fcf_base <= 0),
        'pe_is_estimate': (eps <= 0),
        'valuation_status': valuation_status,
        'confidence_level': confidence_level,
        'valuation_methodology_note': VALUATION_METHODOLOGY_NOTE,
        'warning_message': warning_message,
        'book_value_per_share': round(book_value_per_share, 2),
        'shares_outstanding': shares,

        # ใหม่ (FIX-F1) — ให้หน้า UI/รายงานแสดงที่มาของหนี้สุทธิได้
        'net_debt_used': round(net_debt, 1),
        'debt_basis': debt_basis,
        'debt_excluded_payables': round(ap_excluded, 1),
    }


def build_fair_value_yearly(df_fin_ticker, df_price_ticker, ticker):
    """Fair Value ย้อนหลังรายปี เทียบราคาปิดสิ้นปี (ไม่เปลี่ยนในรอบนี้)"""
    rows = []
    if df_fin_ticker is None or df_fin_ticker.empty or df_price_ticker is None or df_price_ticker.empty:
        return pd.DataFrame(rows)
    price_df = df_price_ticker.copy()
    price_df['date'] = pd.to_datetime(price_df['date'])
    for yr in sorted(df_fin_ticker['year'].unique()):
        fin_upto = df_fin_ticker[df_fin_ticker['year'] <= yr]
        year_end_prices = price_df[price_df['date'] <= f'{yr}-12-31']
        if fin_upto.empty or year_end_prices.empty:
            continue
        year_end_price = clean_float(year_end_prices.sort_values('date').iloc[-1]['close'], default=None)
        if year_end_price is None:
            continue
        try:
            val = calculate_valuation_module(fin_upto, year_end_price, ticker)
            rows.append({'year': int(yr), 'price': round(year_end_price, 2), 'fair_value': val.get('fair_value')})
        except Exception:
            continue
    return pd.DataFrame(rows)
