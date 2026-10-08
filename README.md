# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | <your full name>                     |
| Project Name     | Clinic Next-Day Smart Booking & Waiting-Time Recommendation System |
| GitHub Username  | <your GitHub username>               |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

Patients often arrive early at clinics and wait for an unknown amount of time. This project helps a clinic (one doctor, one specialty) reduce that waiting. Patients book for **the next day**, and the system tells each patient:

* their queue / appointment number
* the expected consultation time
* the recommended time to arrive at the clinic

The system works in **two phases**, because a new clinic has no historical data:

**Phase 1: Cold Start (rules + RAG)**
The doctor fills one structured registration form (working hours, breaks, appointment duration, policies, peak hours, services). The form is validated with Pydantic and stored in a database. It is also converted into a clinic knowledge document, which is chunked, embedded, and stored in a vector database. Patient questions such as *"What happens if I arrive 20 minutes late?"* are answered with RAG from that specific clinic's rules. Booking uses queue/scheduling logic based on the clinic configuration and current bookings (no ML). Every appointment is logged with what the system predicted and what actually happened.

**Phase 2: Clinic-Specific ML**
After enough real, high-quality records are collected (checked by record count and data quality, not by date alone), a regression model is trained on that clinic's own data to predict actual waiting time. It replaces the rule-based estimate only if it performs better.

**Separation of responsibilities**

| Component | Question it answers |
|---|---|
| RAG | What are this clinic's rules and information? |
| Queue / Scheduling Logic | What is the patient's position and expected schedule? |
| ML Model (Phase 2) | Based on this clinic's history, how long will this patient wait? |
| Recommendation | When should the patient come to the clinic? |

---

# ✨ Features

* Structured clinic registration form with validation (Pydantic)
* Automatic clinic knowledge document generation, with chunking, embeddings, and a vector database
* Clinic-specific question answering with RAG (late arrival, cancellation, walk-in, services, working hours)
* Next-day booking with queue number, expected consultation time, and recommended arrival time
* Real operational data collection (booking, arrival, consultation, outcome, and context data)
* Predicted vs. actual comparison for every appointment
* Clinic-specific waiting-time regression model trained on real data (Phase 2)

---

# 🛠️ Technologies Used

* **Python 3.10+**
* **Pydantic v2**: structured input/output and validation
* **SQLite**: clinic configuration and appointment log
* **ChromaDB**: vector database
* **Sentence embeddings + LLM API**: RAG pipeline
* **scikit-learn / pandas**: feature engineering and regression
* **Jupyter Notebooks (VS Code)**: development and demo
* **Git & GitHub**: version control

---

# ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Recommended) keep notebook outputs out of Git
nbstripout --install

# 5. Add your API keys
cp .env.example .env     # then edit .env
```

Run the tests:

```bash
pytest
```

---

# 🚀 Usage

Run the notebooks in order from the `notebooks/` folder:

| Notebook | What it does |
|---|---|
| `01_schemas_and_config.ipynb` | Defines and validates the clinic configuration and saves it to SQLite |
| `02_phase1_scheduler.ipynb` | Queue number, expected time, and recommended arrival time |
| `03_rag_pipeline.ipynb` | Knowledge document, chunking, embeddings, vector DB, and Q&A |
| `04_data_collection_simulation.ipynb` | Simulates a clinic day and fills the appointment log |
| `05_phase2_ml_training.ipynb` | Data-sufficiency check, features, regression training, and evaluation |
| `06_full_demo.ipynb` | End-to-end demo of both phases |

## Clinic management and RAG assistant

Start the API and Streamlit UI in separate terminals from the repository root:

```powershell
uvicorn app.api:app --reload
streamlit run ui/streamlit_app.py
```

The sidebar contains the Assistant, Knowledge Base, Clinic Configuration, and
System Status pages. Enter the configured `STAFF_API_KEY` in the sidebar to
manage clinic settings, add or remove free-form knowledge, or use staff
operations. Configuration is saved through the API to SQLite. Admin-added
knowledge is chunked and embedded into the existing persistent Chroma
collection; configuration updates rebuild the clinic sections while preserving
those admin entries.

Chroma uses one process-wide persistent client. Set `CHROMA_PATH` in `.env` to
an absolute path for the shared index; relative values are resolved from the
repository root. The legacy `CHROMA_DIR` setting is still accepted when
`CHROMA_PATH` is not set. For local Chroma storage, run one API process only:
the normal development command below uses a single reloader worker; do not add
multiple Uvicorn workers or start another API process against the same index.

To migrate existing clinic settings and admin knowledge, then replace stale
clinic collections with rebuilt indexes, stop the API and run this once from
the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_chroma_indexes.py
```

The script backfills missing settings from `data/sample_clinic_config.json`,
migrates legacy vector-only admin entries to SQLite, and rebuilds each saved
clinic index at the configured Chroma path. Restart the API after it completes.

The assistant uses the configured `LLM_PROVIDER`. For local Qwen generation,
install its separate dependencies using a PyTorch build appropriate for the
machine:

```powershell
pip install -r requirements-qwen.txt
$env:LLM_PROVIDER = "qwen"
```

The model `Qwen/Qwen2.5-1.5B-Instruct` is downloaded on first use and cached
for the running process. It selects CUDA with float16 only when at least 4 GiB
of VRAM is free; otherwise it selects CPU with float32. The selected device
and dtype are logged when the API starts and reported by `/health`. CPU
inference is considerably slower. If the model or its weights are unavailable,
RAG answers fall back to the existing extractive generator.
The existing retrieval evaluation remains in `notebooks/03_rag_demo.ipynb`.

**Project structure**

```
├── app/                  # reusable code
│   ├── schemas.py        # Pydantic models
│   ├── db.py             # SQLite helpers
│   ├── scheduler.py      # Phase 1 queue logic
│   ├── rag.py            # knowledge doc, chunking, vector DB
│   ├── ml.py             # features, training, prediction
│   └── qwen.py           # optional cached local answer generation
├── notebooks/            # one notebook per stage
├── data/                 # sample clinic config
├── tests/
├── requirements.txt
├── requirements-qwen.txt # optional local model dependencies
└── README.md
```

---

# 📸 Demo

<!-- Add screenshots, GIFs, or a demo video here, for example: -->
<!-- ![Booking result](docs/booking_result.png) -->
<!-- [Demo video](link) -->

*Coming soon.*

---

# 📈 Results

<!-- Fill this in at the end with your real numbers, for example: -->
<!-- - Number of appointment records collected -->
<!-- - Phase 1 rule-based wait estimate error (MAE) -->
<!-- - Phase 2 ML model error (MAE) and improvement over the rules -->
<!-- - Example RAG answers -->

*To be completed after testing.*

---

# 🔮 Future Improvements

* Retrain the model periodically as new appointments are logged
* Patient notifications (SMS / WhatsApp) with the recommended arrival time
* Handle walk-ins and emergencies dynamically in the live queue
* Extend to multiple doctors and specialties

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
