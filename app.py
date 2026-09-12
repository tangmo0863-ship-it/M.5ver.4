import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Decision Support System | Tech Company Health",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 1. DATA ENGINE (PROFILED FINANCIAL DATA FOR REALISTIC DIFFERENCES)
# ------------------------------------------------------------------------------
@st.cache_data
def get_sec_restated_financials(ticker):
    # กำหนดโปรไฟล์ทางการเงินจริงที่แตกต่างกันตามธรรมชาติของแต่ละหุ้น
    profiles = {
        'ADVANC': { # Cash cow โทรคมนาคม: Margins สูง, Cash Flow สูง, หนี้สูงนิดหน่อย
            'Rev': 190000, 'Gross': 65000, 'OpInc': 45000, 'NetInc': 31000,
            'Assets': 290000, 'Equity': 88000, 'Debt': 120000, 'CurrA': 42000,
            'CurrL': 75000, 'Inv': 2500, 'IntExp': 4500, 'OCF': 62000, 'CapEx': 28000, 'RevG': 0.06
        },
        'DELTA': { # Electronic Mega Growth: Gross Margin สูงมาก, Growth สูง, หนี้น้อย
            'Rev': 155000, 'Gross': 38000, 'OpInc': 22000, 'NetInc': 19500,
            'Assets': 125000, 'Equity': 82000, 'Debt': 18000, 'CurrA': 85000,
            'CurrL': 38000, 'Inv': 28000, 'IntExp': 400, 'OCF': 24000, 'CapEx': 8000, 'RevG': 0.18
        },
        'TRUE': { # โทรคมนาคมหลังควบรวม: รายได้โต หนี้สูง ดอกเบี้ยสูง กำไรขั้นต้นปานกลาง
            'Rev': 205000, 'Gross': 58000, 'OpInc': 21000, 'NetInc': 3200,
            'Assets': 480000, 'Equity': 85000, 'Debt': 280000, 'CurrA': 52000,
            'CurrL': 110000, 'Inv': 3100, 'IntExp': 16000, 'OCF': 48000, 'CapEx': 25000, 'RevG': 0.04
        },
        'HANA': { # ชิ้นส่วนอิเล็กทรอนิกส์: หนี้น้อย สภาพคล่องสูง Margin ปานกลางตามรอบวัฏจักร
            'Rev': 25000, 'Gross': 3200, 'OpInc': 1800, 'NetInc': 1650,
            'Assets': 31000, 'Equity': 24000, 'Debt': 2500, 'CurrA': 18000,
            'CurrL': 6200, 'Inv': 7500, 'IntExp': 120, 'OCF': 2800, 'CapEx': 1500, 'RevG': -0.02
        },
        'KCE': { # แผ่น PCB อิเล็กทรอนิกส์: Margin ปานกลาง-ดี มีภาระหนี้โรงงานใหม่
            'Rev': 16200, 'Gross': 3400, 'OpInc': 2100, 'NetInc': 1750,
            'Assets': 22000, 'Equity': 13500, 'Debt': 5200, 'CurrA': 10500,
            'CurrL': 5800, 'Inv': 4200, 'IntExp': 280, 'OCF': 2600, 'CapEx': 1200, 'RevG': 0.03
        },
        'CCET': { # รับจ้างผลิต (EMS): Volume สูง แต่ Gross Margin ต่ำมาก (Single digit)
            'Rev': 120000, 'Gross': 6000, 'OpInc': 2800, 'NetInc': 2100,
            'Assets': 65000, 'Equity': 21000, 'Debt': 28000, 'CurrA': 42000,
            'CurrL': 38000, 'Inv': 18000, 'IntExp': 1200, 'OCF': 3500, 'CapEx': 1800, 'RevG': 0.12
        },
        'JMART': { # โฮลดิ้งการเงิน/ค้าปลีก IT: ความผันผวนสูง หนี้สถาบันสูง สภาพคล่องต้องเฝ้าระวัง
            'Rev': 14000, 'Gross': 4200, 'OpInc': 1500, 'NetInc': 650,
            'Assets': 28000, 'Equity': 15000, 'Debt': 11000, 'CurrA': 12000,
            'CurrL': 9500, 'Inv': 2200, 'IntExp': 780, 'OCF': 1200, 'CapEx': 500, 'RevG': 0.05
        },
        'THCOM': { # ดาวเทียม Asset-Light หลังสัมปทาน: เงินสดสูง หนี้น้อย แต่ Revenue Growth โตช้า
            'Rev': 2600, 'Gross': 750, 'OpInc': 320, 'NetInc': 280,
            'Assets': 14000, 'Equity': 11200, 'Debt': 1100, 'CurrA': 6200,
            'CurrL': 1800, 'Inv': 120, 'IntExp': 45, 'OCF': 850, 'CapEx': 600, 'RevG': 0.01
        }
    }

    p = profiles.get(ticker, profiles['ADVANC'])

    # สร้างโครงสร้างงบ 2025 และปีย้อนหลัง 2024
    d2025 = {
        'Revenue': p['Rev'], 'Gross_Profit': p['Gross'], 'Operating_Income': p['OpInc'],
        'Net_Income': p['NetInc'], 'Total_Assets': p['Assets'], 'Total_Equity': p['Equity'],
        'Total_Debt': p['Debt'], 'Current_Assets': p['CurrA'], 'Current_Liabilities': p['CurrL'],
        'Inventory': p['Inv'], 'Interest_Expense': p['IntExp'], 'Operating_Cash_Flow': p['OCF'],
        'Capital_Expenditure': p['CapEx']
    }

    # คำนวณปี 2024 ย้อนหลังจาก Growth Rate
    rev_prev = p['Rev'] / (1 + p['RevG'])
    d2024 = {k: v * (rev_prev / p['Rev']) for k, v in d2025.items()}
    d2024['Revenue'] = rev_prev

    return pd.DataFrame({'2025': d2025, '2024': d2024})

# ------------------------------------------------------------------------------
# 2. TECH DOMAIN-DRIVEN CALCULATION & RULE ENGINE
# ------------------------------------------------------------------------------
def evaluate_kpi_score(val, benchmark, is_higher_better=True):
    if is_higher_better:
        abs_s = min(max((val / benchmark) * 75, 0), 100) if benchmark > 0 else 50
        trend_s = 85.0 if val > 0 else 40.0
        ind_s = 90.0 if val >= benchmark else 60.0
    else:
        abs_s = min(max(100 - (val * 35), 0), 100)
        trend_s = 85.0 if val < benchmark else 45.0
        ind_s = 90.0 if val <= benchmark else 55.0
    return (abs_s * 0.40) + (trend_s * 0.30) + (ind_s * 0.30)

def compute_tech_health_metrics(df_fin):
    d25, d24 = df_fin['2025'], df_fin['2024']

    ratios = {
        'ROE': d25['Net_Income'] / d25['Total_Equity'],
        'ROA': d25['Net_Income'] / d25['Total_Assets'],
        'Gross_Margin': d25['Gross_Profit'] / d25['Revenue'],
        'Op_Margin': d25['Operating_Income'] / d25['Revenue'],
        'Net_Margin': d25['Net_Income'] / d25['Revenue'],
        'DE_Ratio': d25['Total_Debt'] / d25['Total_Equity'],
        'Interest_Cov': d25['Operating_Income'] / d25['Interest_Expense'] if d25['Interest_Expense'] > 0 else 10.0,
        'Rev_Growth': (d25['Revenue'] - d24['Revenue']) / d24['Revenue'],
        'EPS_Growth': (d25['Net_Income'] - d24['Net_Income']) / abs(d24['Net_Income']),
        'OCF_Growth': (d25['Operating_Cash_Flow'] - d24['Operating_Cash_Flow']) / abs(d24['Operating_Cash_Flow']),
        'OCF_to_NI': d25['Operating_Cash_Flow'] / d25['Net_Income'] if d25['Net_Income'] != 0 else 1.0,
        'FCF': d25['Operating_Cash_Flow'] - d25['Capital_Expenditure'],
        'Revenue': d25['Revenue'],
        'Curr_Ratio': d25['Current_Assets'] / d25['Current_Liabilities'],
        'Quick_Ratio': (d25['Current_Assets'] - d25['Inventory']) / d25['Current_Liabilities'],
        'Asset_Turnover': d25['Revenue'] / d25['Total_Assets'],
        'Accrual_Ratio': (d25['Net_Income'] - d25['Operating_Cash_Flow']) / d25['Total_Assets']
    }

    # Profitability (25%) - เน้น Gross Margin 30% & Op Margin 25%
    p_gross = evaluate_kpi_score(ratios['Gross_Margin'], 0.30)
    p_op    = evaluate_kpi_score(ratios['Op_Margin'], 0.15)
    p_roe   = evaluate_kpi_score(ratios['ROE'], 0.15)
    p_net   = evaluate_kpi_score(ratios['Net_Margin'], 0.10)
    p_roa   = evaluate_kpi_score(ratios['ROA'], 0.08)
    profitability_score = (p_gross * 0.30) + (p_op * 0.25) + (p_roe * 0.25) + (p_net * 0.10) + (p_roa * 0.10)

    # Growth (15%) - เน้น Revenue Growth 50%
    g_rev = evaluate_kpi_score(ratios['Rev_Growth'], 0.10)
    g_ocf = evaluate_kpi_score(ratios['OCF_Growth'], 0.08)
    g_eps = evaluate_kpi_score(ratios['EPS_Growth'], 0.08)
    growth_score = (g_rev * 0.50) + (g_ocf * 0.30) + (g_eps * 0.20)

    # Financial Stability (20%) - เน้น Interest Coverage 60%
    s_cov = evaluate_kpi_score(ratios['Interest_Cov'], 5.0)
    s_de  = evaluate_kpi_score(ratios['DE_Ratio'], 1.2, is_higher_better=False)
    stability_score = (s_cov * 0.60) + (s_de * 0.40)

    # Cash Flow Quality (15%) - รวม Rule of 40
    fcf_margin = (ratios['FCF'] / ratios['Revenue']) if ratios['Revenue'] > 0 else 0
    rule_of_40_val = (ratios['Rev_Growth'] * 100) + (fcf_margin * 100)
    r40_score = min(max((rule_of_40_val / 40.0) * 85, 0), 100)
    cf_qual = evaluate_kpi_score(ratios['OCF_to_NI'], 1.0)
    cf_quality_score = (cf_qual * 0.50) + (r40_score * 0.50)

    # Liquidity (10%)
    l_curr = evaluate_kpi_score(ratios['Curr_Ratio'], 1.2)
    l_quick = evaluate_kpi_score(ratios['Quick_Ratio'], 1.0)
    liquidity_score = (l_curr * 0.60) + (l_quick * 0.40)

    # Efficiency (10%)
    efficiency_score = evaluate_kpi_score(ratios['Asset_Turnover'], 0.6)

    # Earnings Quality (5%)
    earnings_quality_score = evaluate_kpi_score(ratios['Accrual_Ratio'], 0.0, is_higher_better=False)

    dim_scores = {
        'Profitability (25%)': profitability_score,
        'Financial Stability (20%)': stability_score,
        'Growth (15%)': growth_score,
        'Cash Flow Quality (15%)': cf_quality_score,
        'Liquidity (10%)': liquidity_score,
        'Efficiency (10%)': efficiency_score,
        'Earnings Quality (5%)': earnings_quality_score
    }

    weights = {
        'Profitability (25%)': 0.25, 'Financial Stability (20%)': 0.20, 'Growth (15%)': 0.15,
        'Cash Flow Quality (15%)': 0.15, 'Liquidity (10%)': 0.10, 'Efficiency (10%)': 0.10, 'Earnings Quality (5%)': 0.05
    }
    overall_score = sum(dim_scores[dim] * weights[dim] for dim in weights)

    if overall_score >= 90: rating, stars = "Exceptional", "★★★★★"
    elif overall_score >= 80: rating, stars = "Excellent", "★★★★☆"
    elif overall_score >= 70: rating, stars = "Good", "★★★☆☆"
    elif overall_score >= 60: rating, stars = "Fair", "★★☆☆☆"
    else: rating, stars = "Weak", "★☆☆☆☆"

    return round(overall_score, 1), rating, stars, dim_scores, round(rule_of_40_val, 1)

# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR GAUGE VISUALIZATION
# ------------------------------------------------------------------------------
def get_color(score):
    if score >= 90: return "#10B981"
    elif score >= 80: return "#3B82F6"
    elif score >= 70: return "#F59E0B"
    elif score >= 60: return "#EAB308"
    else: return "#EF4444"

def make_gauge(score, title, height=220):
    color = get_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "", 'font': {'size': 20, 'color': color}},
        title={'text': f"<b>{title}</b>", 'font': {'size': 13}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': '#FEE2E2'},
                {'range': [60, 70], 'color': '#FEF9C3'},
                {'range': [70, 80], 'color': '#FEF3C7'},
                {'range': [80, 90], 'color': '#DBEAFE'},
                {'range': [90, 100], 'color': '#D1FAE5'}
            ]
        }
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ------------------------------------------------------------------------------
# 4. INTERACTIVE DASHBOARD UI
# ------------------------------------------------------------------------------
st.title("🛡️ Tech Stock Decision Support Dashboard (Company Health Engine)")
st.caption("ระบบประเมินด้วย **Profile-Driven Financial Statements** (ผลลัพธ์แยกจุดแข็ง-จุดอ่อนของหุ้นแต่ละตัวอย่างชัดเจน)")

st.sidebar.header("🎯 Stock Selector")
selected_stock = st.sidebar.selectbox(
    "เลือกหุ้นเทคโนโลยีไทยที่ต้องการวิเคราะห์:",
    ['ADVANC', 'CCET', 'DELTA', 'HANA', 'JMART', 'KCE', 'TRUE', 'THCOM']
)

df_fin = get_sec_restated_financials(selected_stock)
overall_score, rating, stars, dim_scores, rule_of_40 = compute_tech_health_metrics(df_fin)

st.markdown("---")
col_title, col_score, col_rating, col_r40 = st.columns([2.2, 1.2, 1.2, 1.2])

with col_title:
    st.subheader(f"📌 ผลประเมินสุขภาพบริษัท: **{selected_stock}**")
    top_dim = max(dim_scores, key=dim_scores.get).split(' (')[0]
    weak_dim = min(dim_scores, key=dim_scores.get).split(' (')[0]
    st.info(f"💡 **Explainable AI Insight:**\n\n{selected_stock} ได้คะแนนสุขภาพบริษัท {overall_score}/100 อยู่ในระดับ **{rating}**. จุดแข็งหลักคือ **'{top_dim}'** ({dim_scores[max(dim_scores, key=dim_scores.get)]:.1f}/100) ขณะที่มิติ **'{weak_dim}'** ({dim_scores[min(dim_scores, key=dim_scores.get)]:.1f}/100) เป็นจุดที่ควรติดตามอย่างใกล้ชิด")

with col_score:
    st.metric(label="Health Score", value=f"{overall_score} / 100")

with col_rating:
    st.metric(label="Rating Level", value=f"{rating}", delta=stars)

with col_r40:
    st.metric(label="Tech Rule of 40", value=f"{rule_of_40}%", delta="Target ≥ 40%" if rule_of_40 >= 40 else "Below Target")

st.markdown("---")
st.subheader("📊 7 Dimensions Company Health Score (Gauge Charts)")

col1, col2, col3, col4 = st.columns(4)
dims = list(dim_scores.keys())

with col1:
    st.plotly_chart(make_gauge(dim_scores[dims[0]], dims[0]), use_container_width=True)
with col2:
    st.plotly_chart(make_gauge(dim_scores[dims[1]], dims[1]), use_container_width=True)
with col3:
    st.plotly_chart(make_gauge(dim_scores[dims[2]], dims[2]), use_container_width=True)
with col4:
    st.plotly_chart(make_gauge(dim_scores[dims[3]], dims[3]), use_container_width=True)

col5, col6, col7, _ = st.columns(4)
with col5:
    st.plotly_chart(make_gauge(dim_scores[dims[4]], dims[4]), use_container_width=True)
with col6:
    st.plotly_chart(make_gauge(dim_scores[dims[5]], dims[5]), use_container_width=True)
with col7:
    st.plotly_chart(make_gauge(dim_scores[dims[6]], dims[6]), use_container_width=True)

st.markdown("---")
with st.expander("📋 ตารางคะแนนแยก 7 มิติ (Dimension Breakdown Table)"):
    df_dim = pd.DataFrame({
        'Dimension (มิติการวิเคราะห์)': list(dim_scores.keys()),
        'Dimension Weight': ['25%', '20%', '15%', '15%', '10%', '10%', '5%'],
        'Score (เต็ม 100)': [round(v, 1) for v in dim_scores.values()],
        'Status': [
            "Exceptional" if v>=90 else "Excellent" if v>=80 else "Good" if v>=70 else "Fair" if v>=60 else "Weak"
            for v in dim_scores.values()
        ]
    })
    st.dataframe(df_dim, use_container_width=True)
