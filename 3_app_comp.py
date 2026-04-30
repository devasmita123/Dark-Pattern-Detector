import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import torch
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
import re
import PyPDF2
import docx2txt

# 1. PROFESSIONAL PAGE CONFIGURATION & CUSTOM CSS
st.set_page_config(page_title="Fine-Print Detective Pro", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .verdict-safe { background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; border: 2px solid #28a745; margin-bottom: 20px;}
    .verdict-danger { background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; border: 2px solid #dc3545; margin-bottom: 20px;}
    .verdict-warning { background-color: #fff3cd; color: #856404; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; border: 2px solid #ffc107; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        tokenizer = RobertaTokenizer.from_pretrained('./legal_shield_model')
        model = RobertaForSequenceClassification.from_pretrained('./legal_shield_model')
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error("Model not found. Please ensure the model folder exists.")
        return None, None

    explainer = pipeline("text2text-generation", model="google/flan-t5-base")
    return classifier, explainer

classifier, explainer = load_models()

# Helper Function: Extract Text from Files
def extract_text(uploaded_file):
    try:
        if uploaded_file.name.endswith('.txt'):
            return uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return text
        elif uploaded_file.name.endswith('.docx'):
            return docx2txt.process(uploaded_file)
    except Exception as e:
        # If extraction fails, print the error directly inside the text box!
        return f"⚠️ Error extracting text from document: {str(e)}"
    return ""

# Initialize bulletproof session state variables
if "doc_content" not in st.session_state:
    st.session_state.doc_content = ""
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# 2. HEADER SECTION
st.title("🛡️ The Fine-Print Detective")
st.markdown("Advanced Deep Learning framework for identifying, categorizing, and mitigating deceptive patterns in legal contracts.")
st.divider()

# 3. INPUT SECTION (BULLETPROOF EXTRACTION)
st.subheader("Input Legal Document")

# File uploader
uploaded_file = st.file_uploader("Upload a file to auto-extract text (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

# Logic to handle new uploads or file removals
if uploaded_file is not None:
    # If a NEW file is uploaded
    if uploaded_file.name != st.session_state.current_file:
        st.session_state.doc_content = extract_text(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.rerun() # Force UI refresh instantly
else:
    # If the user clicks the 'X' to remove the file
    if st.session_state.current_file is not None:
        st.session_state.doc_content = ""
        st.session_state.current_file = None
        st.rerun()

# The Text Area (using 'value' instead of 'key' avoids the Streamlit bug)
document = st.text_area("Paste text here, or review your uploaded document:", value=st.session_state.doc_content, height=250)

# If the user manually edits the text in the box, save it to the session state
if document != st.session_state.doc_content:
    st.session_state.doc_content = document

# Clear Button Logic
if st.button("Clear Text"):
    st.session_state.doc_content = ""
    st.rerun()

# 4. ANALYSIS PIPELINE
if st.button("Analyze Document", type="primary") and document and classifier:
    with st.spinner("Executing NLP Pipeline..."):
        
        raw_sentences = re.split(r'\n+|(?<=[.!?])\s+', document)
        
        results = []
        traditional_flags = []
        llm_flags = []
        
        for sentence in raw_sentences:
            clean_sentence = sentence.strip()
            if len(clean_sentence) < 10:
                continue
            
            # --- TRADITIONAL NLP PIPELINE ---
            traditional_keywords = ["subscription", "fee", "free", "discount", "offer", "gps", "charge", "refund"]
            trad_is_trap = any(keyword in clean_sentence.lower() for keyword in traditional_keywords)
            if trad_is_trap:
                traditional_flags.append(clean_sentence)

            # --- HYBRID LLM PIPELINE ---
            pred = classifier(clean_sentence)[0]
            llm_is_trap = pred['label'] == 'LABEL_1' 
            
            red_flag_keywords = ["ip address", "microphone", "gps", "deleted forever", "contacts", "precise location"]
            if any(keyword in clean_sentence.lower() for keyword in red_flag_keywords):
                llm_is_trap = True
            
            if llm_is_trap:
                llm_flags.append(clean_sentence)
                
                cat_prompt = f"Categorize this deceptive text into exactly ONE of these types: [Hidden Cost, Fake Urgency, Forced Action, Privacy Zuckering, Misdirection]. Text: '{clean_sentence}'"
                category = explainer(cat_prompt, max_length=15, do_sample=False)[0]['generated_text']
                
                exp_prompt = f"Briefly explain the danger of this specific clause: '{clean_sentence}'"
                explanation = explainer(
                    exp_prompt, max_length=60, do_sample=False, 
                    repetition_penalty=2.0, no_repeat_ngram_size=2
                )[0]['generated_text']
                
                results.append({"text": clean_sentence, "status": "Trap", "category": category, "explanation": explanation})
            else:
                results.append({"text": clean_sentence, "status": "Safe"})

        total_clauses = len(results)
        safe_count = total_clauses - len(llm_flags)
        safety_score = (safe_count / total_clauses) * 100 if total_clauses > 0 else 0

        # 5. FINAL VERDICT LOGIC
        st.subheader("Final Verdict")
        if len(llm_flags) == 0:
            st.markdown('<div class="verdict-safe">✅ VERDICT: SAFE TO ACCEPT.<br><span style="font-size: 16px; font-weight: normal;">No dark patterns or manipulative clauses detected.</span></div>', unsafe_allow_html=True)
        elif safety_score >= 80:
            st.markdown('<div class="verdict-warning">⚠️ VERDICT: PROCEED WITH CAUTION.<br><span style="font-size: 16px; font-weight: normal;">A few manipulative tactics detected. Review the flagged clauses before accepting.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-danger">🚨 VERDICT: DO NOT ACCEPT.<br><span style="font-size: 16px; font-weight: normal;">High concentration of Dark Patterns. This agreement is highly predatory.</span></div>', unsafe_allow_html=True)

        # 6. DASHBOARD METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clauses Analyzed", total_clauses)
        col2.metric("Dark Patterns Found", len(llm_flags), delta=f"-{len(llm_flags)} risks", delta_color="inverse")
        col3.metric("Document Safety Score", f"{safety_score:.1f}%")
        
        st.divider()

        # 7. LIVE DYNAMIC COMPARISON: TRADITIONAL VS LLM
        st.subheader("📊 Live Architecture Comparison")
        st.markdown("Comparing our Hybrid LLM against a Traditional NLP 'Bag-of-Words' Baseline algorithm running in real-time.")
        
        comp_col1, comp_col2 = st.columns(2)
        comp_col1.metric("🚩 Flags by Traditional NLP", len(traditional_flags))
        comp_col2.metric("🎯 Flags by Hybrid LLM", len(llm_flags))

        # Side-by-side expanding boxes for clean reading
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander("🔍 View Clauses Flagged by Traditional NLP"):
                if len(traditional_flags) > 0:
                    for clause in traditional_flags:
                        st.write(f"- {clause}")
                else:
                    st.write("No clauses flagged by Traditional NLP.")

        with exp_col2:
            with st.expander("🔍 View Clauses Flagged by Hybrid LLM"):
                if len(llm_flags) > 0:
                    for clause in llm_flags:
                        st.write(f"- {clause}")
                else:
                    st.write("No clauses flagged by Hybrid LLM.")

        if len(traditional_flags) > len(llm_flags):
            st.info("💡 **Comparison Verdict:** The Traditional NLP model over-flagged the document. It likely flagged safe sentences (like 'free shipping') as traps because of isolated keywords. The LLM successfully ignored safe keywords and focused on context.")
        elif len(traditional_flags) < len(llm_flags):
            st.info("💡 **Comparison Verdict:** The Traditional NLP model missed psychological traps. Because it only looks at historical keywords, it fails to recognize new manipulation. Our LLM recognized these threats through deeper contextual understanding.")
        elif len(traditional_flags) == len(llm_flags) and len(llm_flags) > 0:
            st.success("💡 **Comparison Verdict:** Both models found the same number of traps, but the LLM provides reasoning and categorization, whereas Traditional NLP remains a 'black box'.")

        st.divider()

        # 8. DETAILED BREAKDOWN
        st.subheader("Detailed Clause Analysis")
        for res in results:
            if res["status"] == "Trap":
                with st.expander(f"🚨 SUSPICIOUS: {res['text'][:80]}...", expanded=True):
                    st.error(f"**Clause:** {res['text']}")
                    st.warning(f"**Detected Tactic:** {res['category'].upper()}")
                    st.info(f"**AI Risk Analysis:** {res['explanation']}")
            else:
                with st.expander(f"✅ SAFE: {res['text'][:80]}..."):
                    st.success(res['text'])