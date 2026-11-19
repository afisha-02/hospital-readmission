"""
Healthcare design system styles and custom CSS for Streamlit dashboard.
Clean clinical palette: Slate, Deep Navy, Clinical Teal, Pure White.
"""

def get_custom_css() -> str:
    return """
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    
    /* Top header clean up */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Healthcare Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.80rem;
        color: #0d9488;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Risk Status Banners */
    .risk-banner-high {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-banner-moderate {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-banner-low {
        background-color: #f0fdf4;
        border-left: 5px solid #059669;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Disclaimer badge */
    .disclaimer-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        font-size: 0.82rem;
        color: #475569;
        margin: 1rem 0;
        line-height: 1.45;
    }
    
    /* Section dividers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e3a8a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """
