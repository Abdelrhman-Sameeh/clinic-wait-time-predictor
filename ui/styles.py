"""Shared healthcare styling for the clinic Streamlit interface."""


def get_theme_css() -> str:
    """Return the UI theme used by the Streamlit clinic app."""
    return """
    <style>
        .stApp {
            background: #f4f8fa;
            color: #153b4a;
        }

        .stApp, .stApp * {
            font-family: "Segoe UI", sans-serif;
        }

        .stSidebar {
            background: #ffffff;
            border-right: 1px solid #d8e7eb;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: #eaf3f5;
            color: #245264;
            border-radius: 0.6rem 0.6rem 0 0;
            border: 1px solid #d5e5e9;
            padding: 0.5rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: #087e8b;
            border-color: #90cbd0;
        }

        .stAlert, .stSuccess, .stWarning, .stInfo, .stError {
            border-radius: 0.75rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #dce9ec;
            border-radius: 0.9rem;
            box-shadow: 0 2px 10px rgba(25, 75, 88, 0.04);
            padding: 0.7rem;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > textarea,
        .stDateInput > div > div > input,
        .stTimeInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        [data-testid="stBaseInputContainer"],
        input[type="text"],
        input[type="number"],
        textarea,
        select,
        .stSelectbox [data-baseweb="select"],
        .stTextArea [data-baseweb="textarea"] {
            background: #ffffff !important;
            color: #173d4b !important;
            border: 1px solid #bfd6dc !important;
            border-radius: 0.6rem !important;
            -webkit-text-fill-color: #173d4b !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stNumberInput input::placeholder,
        input::placeholder,
        textarea::placeholder {
            color: #6d8993 !important;
        }

        .stTextInput label,
        .stNumberInput label,
        .stTextArea label,
        .stSelectbox label,
        .stDateInput label,
        .stTimeInput label,
        .stMultiSelect label,
        .stCheckbox label,
        .stRadio label,
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .stMarkdown h4,
        .stMarkdown h5,
        .stMarkdown h6 {
            color: #173d4b !important;
        }

        .stButton > button {
            border-radius: 0.7rem;
            background: linear-gradient(135deg, #087e8b, #126b9a);
            color: #ffffff;
            border: 1px solid #087e8b;
            font-weight: 600;
            padding: 0.65rem 1.1rem;
        }

        .stButton > button:hover {
            filter: brightness(1.12);
        }

        .stDataFrame, .stDataFrame * {
            background: #ffffff;
            color: #173d4b;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        [data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #dce9ec;
            border-radius: 0.9rem;
        }

        .stCaption {
            color: #587681 !important;
        }
    </style>
    """
