"""
Reusable UI component helpers for Streamlit application.
"""

import streamlit as st

def render_disclaimer():
    """Renders prominent non-clinical prototype disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ RESEARCH / PORTFOLIO PROTOTYPE:</strong> This platform estimates 30-day hospital readmission risk 
            from historical encounter records for operational and research evaluation. It is <strong>NOT</strong> a clinical 
            decision-making or diagnostic tool and has not undergone prospective clinical trials. 
            All clinical treatment choices require licensed physician evaluation.
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi(col, label: str, value: str, sub: str = "", delta_color: str = "#0d9488"):
    """Renders a polished KPI metric card in the specified column."""
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub" style="color: {delta_color};">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_risk_badge(prob: float, threshold: float, risk_cat: str):
    """Renders high-visibility patient risk banner."""
    pct = prob * 100
    if risk_cat == "HIGH":
        css_class = "risk-banner-high"
        color = "#e11d48"
        icon = "🔴"
        title = "HIGH READMISSION RISK"
    elif risk_cat == "MODERATE":
        css_class = "risk-banner-moderate"
        color = "#d97706"
        icon = "🟡"
        title = "MODERATE READMISSION RISK"
    else:
        css_class = "risk-banner-low"
        color = "#059669"
        icon = "🟢"
        title = "LOW READMISSION RISK"
        
    st.markdown(
        f"""
        <div class="{css_class}">
            <div style="font-size: 0.9rem; font-weight: 700; color: {color};">{icon} {title}</div>
            <div style="display: flex; align-items: baseline; gap: 1rem; margin-top: 0.3rem;">
                <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{pct:.1f}%</div>
                <div style="font-size: 0.95rem; color: #475569;">
                    Operating Threshold: <strong>{threshold:.3f}</strong> | Decision: 
                    <span style="color: {color}; font-weight: 700;">
                        {'FLAGGED FOR REVIEW' if prob >= threshold else 'STANDARD DISCHARGE'}
                    </span>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.35rem;">
                Calibrated posterior probability reflecting observed 30-day readmission likelihood.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
