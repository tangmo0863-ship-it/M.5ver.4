"""
calculate_modules/company_health.py
-------------------------------------
สูตรคำนวณโมดูล "Company Health" (💚) — คู่กับ pages_content/company_health.py

=== DATA CONTRACT (key เดิมครบ + key ใหม่) ===
calculate_health_module(df_fin_ticker) คืน dict:
    health_score, roe, roa, de_ratio, current_ratio,
    s_profitability, s_liquidity, s_debt                      (key เดิม — อาจเป็น None ถ้าข้อมูลไม่พอ)
    roe_reported, roa_reported, ratio_mismatch, health_components_used, health_data_status   (ใหม่)
build_health_score_yearly(df_fin_ticker) คืน DataFrame [year, health_score]

=== CHANGELOG (รอบแก้ "ตัวเลขต้องถูกก่อน") ===
- [FIX-H1] คำนวณ ROE = NI / Total Equity และ ROA = NI / Total Assets จากงบเอง แทนคอลัมน์ roe/roa ในไฟล์
  เหตุผล: คอลัมน์ในไฟล์ขัดกับกำไรจริง เช่น HANA 2025 ขาดทุน (NI < 0) แต่ไฟล์ให้ ROE +2.51%, ROA +2.31%
  และ JMART 2025 ขาดทุนแต่ ROA +2.67% ทำให้หุ้นขาดทุนได้คะแนนกำไรเป็นบวก
  ค่าในไฟล์ยังเก็บไว้ที่ roe_reported / roa_reported และติด ratio_mismatch=True ถ้าต่างกันเกิน 1 จุด %
- [FIX-H2] เลิกใช้ค่า default ปลอม (roe=10, roa=5, de=1.0, cr=1.2) ตอนข้อมูลหาย — ใช้ None แทน
  แล้วคำนวณคะแนนจากองค์ประกอบที่มีจริง (ถ่วงน้ำหนักใหม่ตามที่มี) ถ้ามีไม่ถึง 3 ใน 4 องค์ประกอบ → health_score = None
- [FIX-H3] เลิก clip คะแนนที่ 25-98 (ซ่อนหุ้นที่แย่จริง) — คะแนนอยู่ในช่วง 0-100 ตามธรรมชาติของสูตรอยู่แล้ว
- [FIX-H4] D/E ติดลบ (ส่วนผู้ถือหุ้นติดลบ) เดิมได้คะแนนหนี้เต็ม 100 → ตอนนี้ให้ 0 (ความเสี่ยงสูงสุด)
- [FIX-H5] กัน DataFrame ว่าง (เดิม IndexError) และรวมสูตรไว้ที่ _score_row() ที่เดียว ไม่เขียนซ้ำ 2 ที่
"""

import numpy as np
import pandas as pd

from calculate_modules.common import clean_float

# น้ำหนักเดิมของทีม (ไม่เปลี่ยนในรอบนี้)
W_ROE, W_ROA, W_LIQ, W_DEBT = 0.30, 0.25, 0.20, 0.25
MIN_COMPONENTS = 3            # ต้องมีอย่างน้อย 3 ใน 4 องค์ประกอบถึงจะให้คะแนนรวม
MISMATCH_TOLERANCE = 1.0      # จุด % ที่ยอมให้ค่าในไฟล์ต่างจากที่คำนวณเอง


def _ratio_pct(num, den):
    """num/den*100 ถ้าคำนวณได้ ไม่งั้น None"""
    if num is None or den is None or den == 0:
        return None
    return num / den * 100.0


def _score_row(r):
    """คำนวณคะแนนของงบ 1 ปี — ใช้ทั้งปีล่าสุดและกราฟรายปี (สูตรเดียวกันเป๊ะ)"""
    ni = clean_float(r.get('net_income'), default=None)
    eq = clean_float(r.get('total_equity'), default=None)
    ta = clean_float(r.get('total_assets'), default=None)
    roe_rep = clean_float(r.get('roe'), default=None)
    roa_rep = clean_float(r.get('roa'), default=None)

    # FIX-H1: คำนวณจากงบเองก่อน ถ้าคำนวณไม่ได้ค่อยใช้ค่าในไฟล์
    roe_calc = _ratio_pct(ni, eq) if (eq is not None and eq > 0) else None
    roa_calc = _ratio_pct(ni, ta)
    roe = roe_calc if roe_calc is not None else roe_rep
    roa = roa_calc if roa_calc is not None else roa_rep

    de = clean_float(r.get('de_ratio'), default=None)
    cr = clean_float(r.get('current_ratio'), default=None)

    mismatch = any(
        c is not None and rep is not None and abs(c - rep) > MISMATCH_TOLERANCE
        for c, rep in [(roe_calc, roe_rep), (roa_calc, roa_rep)]
    )

    s_roe = float(np.clip(roe * 3.5, 0, 100)) if roe is not None else None
    s_roa = float(np.clip(roa * 7.0, 0, 100)) if roa is not None else None
    s_liq = float(np.clip(cr * 45.0, 0, 100)) if cr is not None else None
    if de is None:
        s_debt = None
    elif de < 0:                       # FIX-H4
        s_debt = 0.0
    else:
        s_debt = float(np.clip((2.5 - de) * 40.0, 0, 100))

    parts = [(s_roe, W_ROE), (s_roa, W_ROA), (s_liq, W_LIQ), (s_debt, W_DEBT)]
    avail = [(s, w) for s, w in parts if s is not None]
    if len(avail) >= MIN_COMPONENTS:   # FIX-H2 + FIX-H3 (ไม่มี clip 25-98)
        score = round(sum(s * w for s, w in avail) / sum(w for _, w in avail), 1)
    else:
        score = None

    if s_roe is not None and s_roa is not None:
        s_prof = round(s_roe * 0.6 + s_roa * 0.4, 1)
    else:
        s_prof = s_roe if s_roe is not None else s_roa

    return {
        'health_score': score,
        'roe': None if roe is None else round(roe, 2),
        'roa': None if roa is None else round(roa, 2),
        'de_ratio': None if de is None else round(de, 2),
        'current_ratio': None if cr is None else round(cr, 2),
        's_profitability': None if s_prof is None else round(float(s_prof), 1),
        's_liquidity': None if s_liq is None else round(s_liq, 1),
        's_debt': None if s_debt is None else round(s_debt, 1),
        'roe_reported': roe_rep,
        'roa_reported': roa_rep,
        'ratio_mismatch': bool(mismatch),
        'health_components_used': len(avail),
        'health_data_status': 'OK' if score is not None else 'INSUFFICIENT_DATA',
    }


def calculate_health_module(df_fin_ticker):
    """Module 1: Company Health (ใช้งบปีล่าสุดที่มีจริง)"""
    if df_fin_ticker is None or df_fin_ticker.empty:      # FIX-H5
        out = _score_row({})
        out['health_data_status'] = 'NO_DATA'
        return out
    r = df_fin_ticker.sort_values(by='year').iloc[-1]
    return _score_row(r)


def build_health_score_yearly(df_fin_ticker):
    """คะแนน Health รายปี (ใช้ _score_row ตัวเดียวกับปีล่าสุด)"""
    if df_fin_ticker is None or df_fin_ticker.empty:
        return pd.DataFrame(columns=['year', 'health_score'])
    rows = [{'year': int(r['year']), 'health_score': _score_row(r)['health_score']}
            for _, r in df_fin_ticker.sort_values('year').iterrows()]
    return pd.DataFrame(rows)
