# The Dark Pattern Detective
An advanced Deep Learning framework that identifies, categorizes, and mitigates deceptive "Dark Patterns" in legal contracts and Terms of Service.

![Main Dashboard & Final Verdict](screenshots/web1.png)
*The main dashboard calculating a document safety score and generating a Final Verdict banner.*

## 🚀 Project Overview
Traditional NLP models struggle to understand the complex, deceptive "legalese" used by companies to trick users. This project utilizes a **Hybrid LLM Architecture** to provide context-aware detection and Explainable AI (XAI) risk analysis.

## 🧠 Tech Stack
* **High-Speed Filtering:** Fine-tuned `RoBERTa` for binary sequence classification.
* **Explainable AI (XAI) & Categorization:** `FLAN-T5` Generative AI for zero-shot multi-class categorization and human-readable risk summaries.
* **Frontend:** Streamlit 
* **Document Processing:** PyPDF2, docx2txt

## 📸 System Output & Features

### 1. Explainable AI (XAI) & Clause Categorization
Instead of acting as a "black box," the system isolates predatory sentences, categorizes the psychological tactic being used, and generates a plain-English explanation of the legal risk.
![XAI Expanders](screenshots/web3.png)

### 2. Smart Document Extraction
Upload `.txt`, `.pdf`, or `.docx` files, and the system instantly extracts and formats the text for pipeline processing.
![Document Extraction](screenshots/doc_inp.png)

### 3. Dynamic Traditional ML vs. LLM Comparison
The system runs a real-time Traditional "Bag-of-Words" baseline against the RoBERTa model, mathematically demonstrating the LLM's superiority in avoiding false positives and understanding contextual intent.
![Architecture Comparison](screenshots/image_d55979.png)

### 4. Backend Model Testing Pipeline
The repository includes the complete machine learning lifecycle scripts, from data preparation to terminal-based model confidence testing.
![Terminal Output](screenshots/test_model_output.png)

## ⚙️ How to Run Locally
1. Clone the repository: `git clone https://github.com/yourusername/fine-print-detective.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
*(Note: Requires the local `./legal_shield_model` weights to execute the RoBERTa pipeline).*
