"""
Custom CSS styles for the Streamlit application.

Implements a premium dark theme with:
- Zinc/Slate neutral base (adapted from Frontend Skill)
- Single Emerald accent color
- Outfit typography from Google Fonts
- Glassmorphism panels with backdrop-blur
- Smooth transitions and pulse animations
- No generic Streamlit look
"""


def inject_custom_css():
    """Inject custom CSS into the Streamlit app."""
    import streamlit as st

    st.markdown(
        """
        <style>
        /* ============================================
           GOOGLE FONTS — Outfit + JetBrains Mono
           ============================================ */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ============================================
           ROOT VARIABLES
           ============================================ */
        :root {
            --bg-primary: #0c0c0f;
            --bg-secondary: #141419;
            --bg-card: #1a1a22;
            --bg-card-hover: #1f1f29;
            --border-color: rgba(255, 255, 255, 0.06);
            --border-hover: rgba(255, 255, 255, 0.12);
            --text-primary: #f0f0f3;
            --text-secondary: #9898a6;
            --text-muted: #5c5c6e;
            --accent: #34d399;
            --accent-muted: rgba(52, 211, 153, 0.15);
            --accent-glow: rgba(52, 211, 153, 0.08);
            --danger: #f87171;
            --warning: #fbbf24;
            --info: #60a5fa;
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --radius-xl: 28px;
            --shadow-card: 0 4px 24px -4px rgba(0, 0, 0, 0.4);
            --shadow-glow: 0 0 40px -12px rgba(52, 211, 153, 0.15);
            --transition-fast: 0.15s cubic-bezier(0.16, 1, 0.3, 1);
            --transition-smooth: 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* ============================================
           BASE OVERRIDES
           ============================================ */
        .stApp {
            background-color: var(--bg-primary) !important;
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--text-primary) !important;
        }

        /* Main content area */
        .main .block-container {
            padding: 2rem 3rem !important;
            max-width: 1400px !important;
        }

        /* ============================================
           SIDEBAR
           ============================================ */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        section[data-testid="stSidebar"] .block-container {
            padding: 1.5rem 1rem !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
        }

        section[data-testid="stSidebar"] label {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }

        /* ============================================
           TYPOGRAPHY
           ============================================ */
        h1 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            color: var(--text-primary) !important;
        }

        h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            color: var(--text-primary) !important;
        }

        p, li, div {
            font-family: 'Outfit', sans-serif;
        }

        code, pre, .stCodeBlock {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* ============================================
           BUTTONS
           ============================================ */
        .stButton > button {
            background: linear-gradient(135deg, #34d399, #10b981) !important;
            color: #0c0c0f !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1.8rem !important;
            transition: all var(--transition-smooth) !important;
            box-shadow: 0 2px 12px -2px rgba(52, 211, 153, 0.3) !important;
            letter-spacing: -0.01em;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px -4px rgba(52, 211, 153, 0.4) !important;
        }

        .stButton > button:active {
            transform: translateY(0px) scale(0.98) !important;
        }

        /* Secondary buttons */
        .stDownloadButton > button {
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            transition: all var(--transition-smooth) !important;
        }

        .stDownloadButton > button:hover {
            background: var(--bg-card-hover) !important;
            border-color: var(--border-hover) !important;
        }

        /* ============================================
           TABS
           ============================================ */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-secondary) !important;
            border-radius: var(--radius-lg) !important;
            padding: 4px !important;
            gap: 4px !important;
            border: 1px solid var(--border-color) !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: var(--radius-md) !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            padding: 0.5rem 1.2rem !important;
            transition: all var(--transition-fast) !important;
            border: none !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--bg-card) !important;
            color: var(--accent) !important;
            box-shadow: 0 2px 8px -2px rgba(0,0,0,0.3) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* ============================================
           INPUT ELEMENTS
           ============================================ */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-family: 'Outfit', sans-serif !important;
            transition: border-color var(--transition-fast) !important;
        }

        .stSelectbox > div > div:hover,
        .stTextInput > div > div > input:focus {
            border-color: var(--accent) !important;
        }

        /* Toggle switches */
        .stCheckbox label span {
            color: var(--text-secondary) !important;
        }

        /* Slider */
        .stSlider > div > div > div > div {
            background-color: var(--accent) !important;
        }

        /* ============================================
           FILE UPLOADER
           ============================================ */
        .stFileUploader {
            border: 2px dashed var(--border-color) !important;
            border-radius: var(--radius-lg) !important;
            background: var(--bg-secondary) !important;
            transition: all var(--transition-smooth) !important;
        }

        .stFileUploader:hover {
            border-color: var(--accent) !important;
            background: var(--accent-glow) !important;
        }

        .stFileUploader label {
            color: var(--text-secondary) !important;
        }

        /* Sửa triệt để lỗi hiển thị text đè của nút uploader */
        .stFileUploader button {
            color: transparent !important;
            position: relative !important;
            min-width: 110px !important;
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-sm) !important;
            transition: all var(--transition-fast) !important;
        }

        .stFileUploader button::after {
            content: "Browse files" !important;
            position: absolute !important;
            left: 50% !important;
            top: 50% !important;
            transform: translate(-50%, -50%) !important;
            color: var(--text-primary) !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            white-space: nowrap !important;
        }

        .stFileUploader button:hover {
            background-color: var(--bg-card-hover) !important;
            border-color: var(--accent) !important;
        }

        /* ============================================
           EXPANDER
           ============================================ */
        .streamlit-expanderHeader {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
        }

        .streamlit-expanderContent {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            border-top: none !important;
            border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        }

        /* ============================================
           METRICS
           ============================================ */
        [data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem 1.2rem !important;
            transition: all var(--transition-smooth) !important;
        }

        [data-testid="stMetric"]:hover {
            border-color: var(--border-hover) !important;
            box-shadow: var(--shadow-card) !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--accent) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
        }

        /* ============================================
           PROGRESS BAR
           ============================================ */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #34d399, #10b981) !important;
            border-radius: 999px !important;
        }

        .stProgress > div > div {
            background: var(--bg-card) !important;
            border-radius: 999px !important;
        }

        /* ============================================
           ALERTS & STATUS
           ============================================ */
        .stAlert {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border-color) !important;
            font-family: 'Outfit', sans-serif !important;
        }

        /* ============================================
           GLASSMORPHISM CARD (custom component)
           ============================================ */
        .glass-card {
            background: rgba(26, 26, 34, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            box-shadow:
                var(--shadow-card),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            transition: all var(--transition-smooth);
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow:
                var(--shadow-card),
                var(--shadow-glow),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }

        .glass-card h3 {
            margin-top: 0;
            font-size: 1rem;
            color: var(--text-primary);
        }

        .glass-card p {
            color: var(--text-secondary);
            font-size: 0.88rem;
            line-height: 1.6;
        }

        /* ============================================
           STATUS DOT (pulse animation)
           ============================================ */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            position: relative;
        }

        .status-dot.active {
            background: var(--accent);
            box-shadow: 0 0 8px rgba(52, 211, 153, 0.5);
            animation: pulse-dot 2s infinite;
        }

        .status-dot.inactive {
            background: var(--text-muted);
        }

        .status-dot.warning {
            background: var(--warning);
            box-shadow: 0 0 8px rgba(251, 191, 36, 0.4);
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(52, 211, 153, 0.5); }
            50% { opacity: 0.6; box-shadow: 0 0 16px rgba(52, 211, 153, 0.3); }
        }

        /* ============================================
           PIPELINE FLOW DIAGRAM
           ============================================ */
        .pipeline-flow {
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 1rem 0;
        }

        .pipeline-stage {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-left: 2px solid var(--border-color);
            margin-left: 12px;
            transition: all var(--transition-smooth);
            position: relative;
        }

        .pipeline-stage::before {
            content: '';
            position: absolute;
            left: -7px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--bg-primary);
            border: 2px solid var(--accent);
        }

        .pipeline-stage:hover {
            background: var(--accent-glow);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        }

        .pipeline-stage .stage-name {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.9rem;
            min-width: 160px;
        }

        .pipeline-stage .stage-model {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--accent);
            background: var(--accent-muted);
            padding: 2px 10px;
            border-radius: 999px;
        }

        .pipeline-stage .stage-status {
            font-size: 0.78rem;
            color: var(--text-muted);
        }

        /* ============================================
           FIGURE GALLERY
           ============================================ */
        .figure-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            overflow: hidden;
            transition: all var(--transition-smooth);
        }

        .figure-card:hover {
            border-color: var(--border-hover);
            box-shadow: var(--shadow-card);
            transform: translateY(-2px);
        }

        .figure-card img {
            width: 100%;
            height: auto;
            display: block;
        }

        .figure-card .figure-meta {
            padding: 12px 16px;
            border-top: 1px solid var(--border-color);
        }

        .figure-card .figure-meta .meta-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 2px;
        }

        .figure-card .figure-meta .meta-value {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        /* ============================================
           HEADER AREA
           ============================================ */
        .app-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .app-header .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #34d399, #10b981);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 16px -4px rgba(52, 211, 153, 0.4);
        }

        .app-header .header-text h1 {
            font-size: 1.6rem !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
        }

        .app-header .header-text p {
            margin: 4px 0 0 0;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        /* ============================================
           MARKDOWN VIEWER
           ============================================ */
        .markdown-viewer {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 2rem;
            line-height: 1.7;
            max-height: 70vh;
            overflow-y: auto;
        }

        .markdown-viewer h1, .markdown-viewer h2, .markdown-viewer h3 {
            color: var(--text-primary) !important;
            margin-top: 1.5em;
        }

        .markdown-viewer p {
            color: var(--text-secondary) !important;
        }

        .markdown-viewer code {
            background: var(--bg-secondary) !important;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
        }

        .markdown-viewer pre {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 1rem;
            overflow-x: auto;
        }

        .markdown-viewer table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }

        .markdown-viewer th, .markdown-viewer td {
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            text-align: left;
            font-size: 0.88rem;
        }

        .markdown-viewer th {
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Scrollbar */
        .markdown-viewer::-webkit-scrollbar {
            width: 6px;
        }
        .markdown-viewer::-webkit-scrollbar-track {
            background: transparent;
        }
        .markdown-viewer::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 999px;
        }
        .markdown-viewer::-webkit-scrollbar-thumb:hover {
            background: var(--border-hover);
        }

        /* ============================================
           ANIMATIONS
           ============================================ */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        .animate-in {
            animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        .shimmer-loading {
            background: linear-gradient(
                90deg,
                var(--bg-card) 25%,
                var(--bg-card-hover) 50%,
                var(--bg-card) 75%
            );
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: var(--radius-sm);
        }

        /* ============================================
           HIDE STREAMLIT DEFAULTS
           ============================================ */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: var(--bg-primary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }

        /* Divider */
        hr {
            border-color: var(--border-color) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
