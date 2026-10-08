"""Shared dark-theme styling for the clinic Streamlit interface."""


def get_theme_css() -> str:
    """Return the UI theme used by the Streamlit clinic app."""
    return """
    <style>
        .stApp {
            background: #090d14;
            color: #f4f7fb;
        }

        .stApp, .stApp * {
            font-family: "Segoe UI", sans-serif;
        }

        .stSidebar {
            background: #0f1726;
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: #101827;
            color: #dfe7fb;
            border-radius: 0.6rem 0.6rem 0 0;
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 0.5rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: #19324c;
            color: #f8fbff;
            border-color: rgba(125, 211, 252, 0.4);
        }

        .stAlert, .stSuccess, .stWarning, .stInfo, .stError {
            border-radius: 0.75rem;
        }

        div[data-testid="stHorizontalBlock"] > div {
            border-radius: 0.75rem;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > textarea,
        .stSelectbox > div > div,
        .stDateInput > div > div,
        .stTimeInput > div > div,
        .stMultiSelect > div > div,
        .stCheckbox > label,
        .stRadio > div {
            background: #111827 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
            border-radius: 0.6rem !important;
        }

        .stTextInput label,
        .stNumberInput label,
        .stTextArea label,
        .stSelectbox label,
        .stDateInput label,
        .stTimeInput label,
        .stMultiSelect label,
        .stCheckbox label,
        .stRadio label {
            color: #e2e8f0 !important;
        }

        .stButton > button {
            border-radius: 0.7rem;
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
            color: white;
            border: none;
            font-weight: 600;
            padding: 0.65rem 1.1rem;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
        }

        .stDataFrame, .stDataFrame * {
            background: #0b1220;
            color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """
