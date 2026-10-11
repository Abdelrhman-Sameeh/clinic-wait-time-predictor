# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Abdelrhman Sameeh Zahran             |
| Project Name     |clinic-wait-time-predictor            |
| GitHub Username  | Abdelrhman-Sameeh                    |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview
## The Story Behind the Project

A friend of mine had been missing college for two days. When I asked him why, he told me he was accompanying his mother to a clinic.

I was confused. How could a clinic visit take two entire days? What about his lectures and exams?

He explained that the doctor was well-known for his expertise, attracting patients from different governorates. Some patients arrived as early as dawn just to secure their place in the queue. My friend had spent two days waiting for his mother to be examined, putting his education and daily life on hold because someone he loved needed medical care.

**This made me realize that the problem is not always the quality of healthcare, but also how patients access it.**

### The Solution

The **Clinic Waiting-Time Prediction and Best Visit-Time Recommendation System** aims to make clinic visits more predictable and convenient. Patients can book appointments online for the next day and receive their queue number, estimated consultation time, and recommended arrival time. They can also identify peak hours and understand the clinic's late-arrival policies, helping them avoid unnecessary waiting.

The system also uses **Retrieval-Augmented Generation (RAG)** to answer questions about the clinic's services, specialties, and policies using information provided by the doctor, even outside working hours.

The project follows two phases: **Phase 1 — Cold Start**, where scheduling relies on the doctor's configuration, current bookings, and predefined rules without ML; and **Phase 2 — Clinic-Specific ML**, where a regression model learns from real appointment records to improve waiting-time estimates, replacing the initial method only if it performs better.

### Our Goal

**Help patients spend less time waiting and more time living.** By combining smarter scheduling, data-driven predictions, and AI-powered information retrieval, the system aims to make healthcare access more organized, transparent, and convenient for both patients and clinic owners.
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
<img width="975" height="470" alt="image" src="https://github.com/user-attachments/assets/d8ac1fc3-d705-4988-a40e-3c5ede598e81" />

RAG answering:
<img width="975" height="467" alt="image" src="https://github.com/user-attachments/assets/57268ef2-196d-43b1-8c59-a9e929db20fe" />

Adding knowledge base:
<img width="975" height="466" alt="image" src="https://github.com/user-attachments/assets/1dc9af4a-cd06-4b56-9b56-e1df969cd64f" />

clinic config:
<img width="863" height="1090" alt="image" src="https://github.com/user-attachments/assets/8eb80341-5155-40bf-9b82-2d31f7f09334" />
<img width="853" height="1088" alt="image" src="https://github.com/user-attachments/assets/d7fd0c4d-9863-4e36-a68f-da178b8545b1" />

system status: 
<img width="752" height="507" alt="image" src="https://github.com/user-attachments/assets/799a0936-c353-4d5a-a356-869ce65c0f93" />

Booking:
<img width="937" height="852" alt="image" src="https://github.com/user-attachments/assets/b744208b-8b69-4dd8-b606-3540bde91c48" />


---

# 📈 Results
Results

Current Status

The project currently implements a rule-based appointment scheduling system and a Retrieval-Augmented Generation (RAG) pipeline for answering clinic-related questions. Quantitative evaluation is still pending.

1. Appointment Data

Appointment records collected: Not yet measured.

Data source: Appointments generated or recorded through the application.

Data collection status: The application supports appointment persistence and visit-event tracking. The actual number of collected records must be measured from the database.

2. Phase 1 — Rule-Based Wait-Time Estimation

Method: Rule-based scheduling and wait-time estimation.

Mean Absolute Error (MAE): Not yet evaluated.

Evaluation status: Requires recorded actual waiting times and corresponding estimates.

The MAE will be calculated as:

[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|
]

where (\hat{y}_i) is the estimated waiting time and (y_i) is the actual waiting time, both measured in the same unit.

3. Phase 2 — Machine Learning Wait-Time Prediction

Model: Not yet implemented as a trained wait-time prediction model.

MAE: Not available.

Improvement over the rule-based baseline: Not available.

A trained model will be evaluated against the Phase 1 baseline using the same held-out test data. Improvement will be reported only after both approaches have been evaluated on comparable observations.

4. RAG Evaluation

Implementation: Clinic knowledge retrieval using embeddings and ChromaDB, followed by answer generation or extraction, depending on configuration.


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
