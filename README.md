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

**Project structure**

```
├── app/                  # reusable code
│   ├── schemas.py        # Pydantic models
│   ├── db.py             # SQLite helpers
│   ├── scheduler.py      # Phase 1 queue logic
│   ├── rag.py            # knowledge doc, chunking, vector DB
│   └── ml.py             # features, training, prediction
├── notebooks/            # one notebook per stage
├── data/                 # sample clinic config
├── tests/
├── requirements.txt
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
