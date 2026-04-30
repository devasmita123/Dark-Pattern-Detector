# The Dark Pattern Detective
An advanced Deep Learning framework that identifies, categorizes, and mitigates deceptive "Dark Patterns" in legal contracts and Terms of Service.

## 🚀 Project Overview
Traditional NLP models struggle to understand the complex, deceptive "legalese" used by companies to trick users. This project utilizes a **Hybrid LLM Architecture** to provide context-aware detection and Explainable AI (XAI) risk analysis.

## 🧠 Tech Stack
* **High-Speed Filtering:** Fine-tuned `RoBERTa` for binary sequence classification.
* **Explainable AI (XAI) & Categorization:** `FLAN-T5` Generative AI for zero-shot multi-class categorization and human-readable risk summaries.
* **Frontend:** Streamlit 
* **Document Processing:** PyPDF2, docx2txt

## 📸 System Output & Features
*(Insert screenshot of your dashboard here)*

### 1. Smart Document Extraction
Upload `.txt`, `.pdf`, or `.docx` files, and the system instantly extracts and formats the text for pipeline processing.

### 2. Dynamic Traditional ML vs. LLM Comparison
*(Insert screenshot of your live comparison metrics here)*
The system runs a real-time Traditional "Bag-of-Words" baseline against the RoBERTa model, mathematically demonstrating the LLM's superiority in avoiding false positives and understanding contextual intent.

## ⚙️ How to Run Locally
1. Clone the repository: `git clone https://github.com/yourusername/fine-print-detective.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
*(Note: Requires the local `./legal_shield_model` weights to execute the RoBERTa pipeline).*
