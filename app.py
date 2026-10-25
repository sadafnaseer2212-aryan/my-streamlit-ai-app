import hashlib
import html
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

st.set_page_config(page_title="RAG Studio | AI Document Assistant", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--blue:#2146D0;--red:#EE364F;--navy:#0F1C3E;--muted:#64718D;--border:#E4E9F3;--ice:#F5F7FC;}
html,body,[data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#F7F9FD;color:var(--navy);}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1240px;padding-top:2.3rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.04em!important;color:#101C3E!important;}
h1 {font-weight:800!important;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #E8EDF5;}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem}
.stButton button, .stFormSubmitButton button, .stDownloadButton button {border-radius:12px!important;min-height:44px;font-weight:700!important;transition:all .2s ease!important;}
.stButton button[kind="primary"],.stFormSubmitButton button[kind="primary"] {background:#2146D0!important;border:1px solid #2146D0!important;color:#fff!important;box-shadow:0 7px 20px rgba(33,70,208,.18)!important;}
.stButton button[kind="primary"]:hover,.stFormSubmitButton button[kind="primary"]:hover{background:#1837AE!important;transform:translateY(-1px);}
.stTextInput input,.stSelectbox [data-baseweb="select"],.stFileUploader {border-radius:12px!important;}
[data-testid="stForm"]{border:0!important;padding:0!important;background:transparent!important;}
[data-testid="stChatMessage"] {border:1px solid #E7EBF4!important;border-radius:17px!important;background:#fff!important;padding:1rem!important;box-shadow:0 5px 20px rgba(30,50,100,.025);}
[data-testid="stFileUploader"] section {background:#fff;border:1.5px dashed #B8C6E9;border-radius:17px;padding:1rem;}
[data-testid="stMetric"] {background:#fff;border:1px solid #E6EBF3;border-radius:15px;padding:16px 18px;}
[data-testid="stMetricLabel"] {font-size:.77rem;color:#64718D;}
[data-testid="stMetricValue"] {font-size:1.55rem;font-family:'Manrope',sans-serif;font-weight:800;}
.hero-shell {position:relative;overflow:hidden;background:#fff;border:1px solid #E6EBF4;border-radius:26px;padding:46px 48px;box-shadow:0 20px 55px rgba(16,35,90,.045);}
.hero-shell:before {content:'';position:absolute;right:-130px;top:-210px;width:490px;height:490px;border-radius:50%;background:radial-gradient(circle,#E9EDFF 0%,rgba(233,237,255,0) 70%);}
.hero-shell:after {content:'';position:absolute;right:95px;bottom:-135px;width:300px;height:300px;border:40px solid #FBE7EA;border-radius:45px;transform:rotate(-28deg);opacity:.7;}
.eyebrow {display:inline-flex;align-items:center;gap:8px;font-weight:800;letter-spacing:.14em;font-size:11px;color:#2146D0;text-transform:uppercase;}
.dot {width:8px;height:8px;border-radius:100%;background:#EE364F;display:inline-block;}
.hero-title {font-family:'Manrope',sans-serif;font-size:clamp(39px,5vw,66px);font-weight:800;color:#101D3E;line-height:1.09;letter-spacing:-.065em;max-width:750px;margin:25px 0 19px;position:relative;z-index:1;}
.blue {color:#2146D0;} .red {color:#EE364F;}
.hero-sub {color:#68758D;font-size:16px;line-height:1.85;max-width:560px;position:relative;z-index:1;}
.pills {margin-top:27px;display:flex;gap:8px;flex-wrap:wrap;position:relative;z-index:1;}
.pills span {background:#F5F7FC;border:1px solid #E7ECF6;color:#465775;padding:7px 13px;border-radius:100px;font-size:12px;font-weight:700;}
.topbar {display:flex;align-items:center;justify-content:space-between;margin:0 0 24px;gap:12px;}
.logomark {background:#2146D0;color:#fff;border-radius:13px;display:inline-flex;align-items:center;justify-content:center;width:43px;height:43px;font-size:25px;font-weight:900;box-shadow:inset -7px -7px 0 rgba(255,255,255,.07);}
.brand {display:flex;align-items:center;gap:12px;font-family:'Manrope',sans-serif;font-weight:800;font-size:18px;color:#17244A;letter-spacing:-.045em;}
.status-pill {background:#F0F9F4;border:1px solid #D5F0E0;color:#21764B;font-size:12px;font-weight:800;border-radius:100px;padding:8px 13px;}
.card-white {background:#fff;border:1px solid #E6EBF4;border-radius:20px;padding:25px 28px;margin:15px 0;box-shadow:0 10px 28px rgba(21,38,87,.025);}
.card-title {font:800 19px 'Manrope',sans-serif;color:#101D3E;letter-spacing:-.04em;margin-bottom:7px;}
.card-sub {font-size:13px;color:#71809A;line-height:1.7;}
.step-head {font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:#EE364F;font-weight:800;margin-bottom:6px;}
.feature-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:15px;}
.feature-card {background:#fff;border:1px solid #E8EDF5;border-radius:17px;padding:21px 19px;}
.feature-icon {font-size:22px;background:#EFF2FF;border-radius:12px;width:43px;height:43px;display:flex;align-items:center;justify-content:center;margin-bottom:15px;}
.feature-name {font-size:14px;font-weight:800;color:#1A294C;}
.feature-desc {font-size:12px;color:#7B879B;line-height:1.7;margin-top:5px;}
.section-label {font-size:12px;font-weight:800;color:#2146D0;text-transform:uppercase;letter-spacing:.14em;margin:14px 0 4px;}
.workspace-banner {background:#fff;border:1px solid #E7ECF4;border-radius:22px;padding:29px 33px;position:relative;overflow:hidden;margin-bottom:22px;}
.workspace-banner:after {content:'';position:absolute;right:-55px;top:-75px;width:260px;height:260px;border:38px solid #F0F3FF;border-radius:50%;}
.workspace-banner h1 {font-size:33px!important;margin:8px 0!important;}
.workspace-banner p {color:#6A7891;font-size:14px;margin-bottom:0;}
.upload-heading {font-size:16px;font-family:'Manrope',sans-serif;font-weight:800;color:#17244A;}
.file-chip {display:inline-block;padding:7px 11px;background:#F1F4FF;color:#2850BD;border:1px solid #DFE7FF;border-radius:8px;font-size:12px;font-weight:700;margin:4px 6px 4px 0;}
.chat-empty {text-align:center;padding:27px 16px;color:#7B879B;font-size:14px;}
.chat-empty b {display:block;color:#19274B;font-size:20px;font-family:'Manrope',sans-serif;margin:9px 0;}
.caption-line {font-size:12px;color:#97A1B5;text-align:center;margin-top:20px;}
@media(max-width:760px){.hero-shell{padding:28px 22px}.hero-title{font-size:39px}.feature-grid{grid-template-columns:1fr}.workspace-banner{padding:22px}.workspace-banner h1{font-size:27px!important}}
</style>
""", unsafe_allow_html=True)

DEFAULT_MODEL = "gpt-4o-mini"


def verify_key(key: str) -> tuple[bool, str]:
    """Make a real, inexpensive API call; listing models alone may not verify billing."""
    try:
        client = OpenAI(api_key=key, timeout=20.0, max_retries=0)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=5,
        )
        return True, "API key verified successfully."
    except Exception as exc:
        # Never expose an exception containing potentially sensitive details.
        name = type(exc).__name__
        if name == "AuthenticationError":
            return False, "Invalid API key. Please check it and try again."
        if name == "RateLimitError":
            return False, "Rate limit or insufficient API quota. Please review your OpenAI billing and limits."
        if name == "PermissionDeniedError":
            return False, "This API key does not have permission to use the selected model."
        return False, f"Verification failed ({name}). Check your connection and API account."


def docs_from_upload(upload) -> list[Document]:
    name = upload.name
    suffix = Path(name).suffix.lower()
    data = upload.getvalue()
    docs = []
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as file:
            file.write(data)
            path = file.name
        try:
            docs = PyPDFLoader(path).load()
        finally:
            os.unlink(path)
        for d in docs:
            d.metadata["source"] = name
            d.metadata["location"] = f"Page {int(d.metadata.get('page', 0)) + 1}"
    elif suffix == ".docx":
        from docx import Document as WordDocument
        word = WordDocument(io.BytesIO(data))
        paragraphs = [p.text.strip() for p in word.paragraphs if p.text.strip()]
        for table in word.tables:
            paragraphs.extend(" | ".join(c.text for c in row.cells) for row in table.rows)
        docs = [Document(page_content="\n".join(paragraphs), metadata={"source": name, "location": "Document"})]
    elif suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        for sheet, frame in sheets.items():
            frame = frame.fillna("")
            for start in range(0, len(frame), 25):
                batch = frame.iloc[start:start+25]
                text = "\n".join(f"Row {i+2}: " + "; ".join(f"{col}: {value}" for col,value in row.items()) for i,row in batch.iterrows())
                docs.append(Document(page_content=text, metadata={"source": name, "location": f"Sheet {sheet}, rows {start+2}–{start+len(batch)+1}"}))
    elif suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(data)).fillna("")
        for start in range(0, len(frame), 25):
            batch = frame.iloc[start:start+25]
            text = "\n".join(f"Row {i+2}: " + "; ".join(f"{col}: {value}" for col,value in row.items()) for i,row in batch.iterrows())
            docs.append(Document(page_content=text, metadata={"source": name, "location": f"Rows {start+2}–{start+len(batch)+1}"}))
    elif suffix == ".json":
        parsed = json.loads(data.decode("utf-8-sig"))
        if isinstance(parsed, list):
            for start in range(0, len(parsed), 20):
                docs.append(Document(page_content=json.dumps(parsed[start:start+20], ensure_ascii=False, indent=2), metadata={"source": name, "location": f"Records {start+1}–{min(start+20,len(parsed))}"}))
        else:
            docs = [Document(page_content=json.dumps(parsed, ensure_ascii=False, indent=2), metadata={"source": name, "location": "JSON"})]
    elif suffix == ".txt":
        docs = [Document(page_content=data.decode("utf-8-sig"), metadata={"source": name, "location": "Text"})]
    return [doc for doc in docs if doc.page_content.strip()]


def build_vectorstore(uploads, api_key, chunk_size, chunk_overlap):
    docs = []
    for upload in uploads:
        docs.extend(docs_from_upload(upload))
    if not docs:
        raise ValueError("No extractable text found. Scanned PDFs may need OCR.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings), len(docs), len(chunks)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions ONLY using the supplied document context. "
               "If it is not in the context, say: I couldn't find that in the uploaded documents. "
               "Cite the document name and page, sheet, row, or section when available. "
               "Do not invent sources.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')} | {d.metadata.get('location', 'Unknown')}]\n{d.page_content}" for d in docs)


def get_chain(store, api_key, model, k):
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | RAG_PROMPT | llm | StrOutputParser())
    return chain, retriever


if "verified" not in st.session_state:
    st.session_state.verified = False
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.verified:
    st.markdown('<div class="topbar"><div class="brand"><span class="logomark">R</span> RAG STUDIO <span style="color:#EE364F">.</span></div><span style="font-size:12px;color:#8A96AB;font-weight:700">AI DOCUMENT WORKSPACE</span></div>', unsafe_allow_html=True)
    st.markdown('''<div class="hero-shell"><div class="eyebrow"><span class="dot"></span> WORK SMARTER WITH YOUR FILES</div>
    <div class="hero-title">Your knowledge.<br><span class="blue">One intelligent</span><br><span class="red">conversation.</span></div>
    <div class="hero-sub">Turn scattered documents into clear answers. Upload reports, spreadsheets, Word files and structured data — and let AI find the details that matter.</div>
    <div class="pills"><span>PDF</span><span>WORD</span><span>EXCEL</span><span>JSON</span><span>CSV</span><span>TXT</span></div></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="feature-grid">
    <div class="feature-card"><div class="feature-icon">◈</div><div class="feature-name">Everything in one place</div><div class="feature-desc">Search multiple file types together in one clean workspace.</div></div>
    <div class="feature-card"><div class="feature-icon">◎</div><div class="feature-name">Answers with evidence</div><div class="feature-desc">Trace responses back to the source page, sheet or record.</div></div>
    <div class="feature-card"><div class="feature-icon">✦</div><div class="feature-name">Private key connection</div><div class="feature-desc">Connect using your own OpenAI API key before entering.</div></div></div>''', unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.35, 1], gap="large")
    with center:
        st.markdown('''<div class="card-white"><div class="step-head">STEP 01 / ACCESS</div><div class="card-title">Connect your AI workspace</div><div class="card-sub">Enter your OpenAI API key. We'll check it with a small live request before taking you to the dashboard.</div></div>''', unsafe_allow_html=True)
        with st.form("connect_form"):
            entered_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", help="Your key is held in session state, not displayed on the dashboard.")
            submitted = st.form_submit_button("Verify key and launch workspace  →", type="primary", use_container_width=True)
        if submitted:
            if not entered_key.strip():
                st.error("Enter your OpenAI API key to continue.")
            else:
                with st.spinner("Verifying connection..."):
                    valid, feedback = verify_key(entered_key.strip())
                if valid:
                    st.session_state.api_key = entered_key.strip()
                    st.session_state.verified = True
                    st.rerun()
                else:
                    st.error(feedback)
        st.caption("API verification uses a small billable request. Avoid uploading confidential files unless your provider and deployment meet your privacy needs.")
    st.markdown('<div class="caption-line">RAG STUDIO · YOUR DOCUMENTS, DECODED</div>', unsafe_allow_html=True)
    st.stop()



# PREMIUM V3 WORKSPACE — the API connection screen above is intentionally unchanged.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&display=swap');
:root{--midnight:#080F26;--navy:#111C40;--panel:#111B39;--blue:#345DFF;--red:#FF455B;--violet:#8972FF;--paper:#E8EEFF;--muted:#9CA9CE}
html,body,[data-testid="stApp"],.stApp{font-family:'Outfit',sans-serif!important}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse 850px 600px at 77% 3%,rgba(47,83,229,.23),transparent 65%),radial-gradient(ellipse 550px 400px at 10% 80%,rgba(255,64,98,.075),transparent 70%),#080F26!important;color:#E8EEFF!important}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1650px!important;padding:1.1rem 2.1rem 2rem!important}
[data-testid="stSidebar"]{background:#0B1430!important;border-right:1px solid #253359!important}
[data-testid="stSidebar"] *{color:#DCE6FF}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#AAB8DC}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{color:#BCC9E8!important}
[data-testid="stSidebar"] .stButton button{background:#121E40!important;border:1px solid #30416E!important;color:#F0F4FF!important;box-shadow:none!important}
[data-testid="stSidebar"] .stButton button:hover{border-color:#5175FF!important;background:#1C2C59!important}
[data-testid="stSidebar"] hr{border-color:#26345A!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#162345!important;border-color:#34446F!important;color:#fff!important}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:#121D3C!important;border:1px solid #30416E;border-radius:12px}
[data-testid="stSidebar"] .stSlider{color:#fff}
.side-brand{font:800 20px Outfit,sans-serif;color:white;letter-spacing:-.6px;display:flex;align-items:center;gap:11px;margin:12px 0 21px}
.side-logo{width:40px;height:40px;display:grid;place-items:center;background:linear-gradient(145deg,#4167FF,#2339AC);border-radius:13px;box-shadow:0 8px 28px #284DFF3F;color:#fff}
.side-status{color:#85F1C5!important;background:#0E302F;border:1px solid #245B50;border-radius:12px;padding:13px;font-size:12px;font-weight:700;letter-spacing:.2px}
.side-eyebrow{color:#7F9CFB!important;font-size:10px;font-weight:800;letter-spacing:2.1px;margin:26px 0 12px}
.topline{display:flex;justify-content:space-between;align-items:center;padding:10px 0 24px;border-bottom:1px solid #273556;gap:12px}
.topline-left{display:flex;align-items:center;gap:12px;font-weight:800;font-size:18px;color:#EAF0FF;letter-spacing:-.6px}
.topicon{height:39px;width:39px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(140deg,#3156F4,#792BDA);box-shadow:0 0 28px #4560FB55;font-size:22px}
.topline-right{display:flex;align-items:center;gap:13px;font-size:11px;font-weight:700;color:#B2C0E4}
.live-circle{width:8px;height:8px;background:#36DD9A;border-radius:50%;box-shadow:0 0 0 6px #36DD9A15;animation:live 1.7s infinite}
@keyframes live{50%{box-shadow:0 0 0 11px #36DD9A00}}
.hero-pro{position:relative;overflow:hidden;min-height:230px;margin:23px 0 22px;padding:35px 39px;background:linear-gradient(105deg,#243ECB 0%,#304FE0 51%,#5138A5 100%);border-radius:25px;box-shadow:0 25px 80px #080E2B88;border:1px solid #6379FF5B;isolation:isolate;animation:appear .7s ease both}
.hero-pro:before{content:'';position:absolute;right:-75px;top:-175px;width:485px;height:485px;border:2px solid #FFFFFF55;border-radius:50%;box-shadow:0 0 0 37px #FFFFFF0C,0 0 0 94px #FFFFFF09;animation:orbit 15s linear infinite;z-index:-1}
.hero-pro:after{content:'';position:absolute;left:-100px;bottom:-165px;width:450px;height:280px;border-radius:50%;background:#FD3A64;filter:blur(100px);opacity:.18;z-index:-1}
@keyframes orbit{to{transform:rotate(360deg)}}
@keyframes appear{from{opacity:0;transform:translateY(17px)}to{opacity:1;transform:translateY(0)}}
.hero-kicker{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:800;letter-spacing:2px;color:#E9ECFF;margin-bottom:12px}
.hero-kicker:before{content:'';width:24px;height:4px;border-radius:5px;background:#FF4D62}
.hero-title-pro{font-size:clamp(31px,3.2vw,47px);font-weight:900;line-height:1.13;letter-spacing:-1.9px;color:white;max-width:700px}
.hero-title-pro .pop{color:#FF6073}
.hero-desc{font-size:14px;color:#DFE6FF;max-width:550px;line-height:1.8;margin-top:12px}
.hero-orb{position:absolute;right:14%;top:24%;width:145px;height:145px;border-radius:39px;transform:rotate(-14deg);background:linear-gradient(140deg,#FFFFFF33,#FFFFFF07);border:1px solid #FFFFFF77;box-shadow:inset 0 0 30px #FFFFFF22,0 20px 65px #04116477;display:grid;place-items:center;font-size:78px;color:#fff;animation:float 4.5s ease-in-out infinite;backdrop-filter:blur(12px)}
@keyframes float{0%,100%{transform:rotate(-14deg) translateY(0)}50%{transform:rotate(-10deg) translateY(-15px)}}
.v3-card{background:linear-gradient(165deg,#151F42,#101A37);border:1px solid #2F3E68;border-radius:22px;box-shadow:0 15px 45px #03091B5E;padding:22px 23px 24px;animation:appear .65s both;transition:border-color .25s,box-shadow .25s}
.v3-card:hover{border-color:#536BB3;box-shadow:0 20px 55px #02081B80}
.v3-title{color:#EFF3FF;font-size:17px;font-weight:800;letter-spacing:-.4px}
.v3-sub{color:#9EAED3;font-size:12px;line-height:1.65;margin:6px 0 18px}
.v3-tag{font-size:10px;font-weight:800;color:#89A5FF;letter-spacing:1.8px}
.division-header{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:6px}
.st-key-upload_card [data-testid="stFileUploader"] section{background:linear-gradient(135deg,#1E315F,#18294D)!important;border:2px dashed #6B8DFF!important;border-radius:18px!important;min-height:145px!important;display:flex;align-items:center;justify-content:center;animation:glowborder 2.8s ease-in-out infinite}
@keyframes glowborder{0%,100%{border-color:#5776E5;box-shadow:0 0 0 0 #446EFF0B}50%{border-color:#96ABFF;box-shadow:0 0 36px #3C66F63B}}
.st-key-upload_card [data-testid="stFileUploader"] section:hover{background:#26396A!important;transform:translateY(-2px);box-shadow:0 0 38px #5076FB55!important}
.st-key-upload_card [data-testid="stFileUploader"] section button{background:#FF455B!important;border:0!important;color:#fff!important;border-radius:11px!important;font-weight:800!important}
.st-key-upload_card [data-testid="stFileUploader"] section *{color:#EBF0FF!important}
.st-key-upload_card [data-testid="stWidgetLabel"] p{color:#E8EFFF!important;font-weight:700}
.formats{font-size:11px;color:#A3B2DC;letter-spacing:.3px;margin:13px 0 0}
.format-tag{padding:5px 8px;background:#263865;border:1px solid #354B7F;border-radius:7px;display:inline-block;margin:3px 2px;font-size:10px;font-weight:800;color:#C3D0FF}
.file-row{display:flex;align-items:center;gap:12px;background:#1D2A51;border:1px solid #344875;border-radius:12px;padding:13px;margin:9px 0;transition:transform .2s,background .2s;animation:appear .4s both}
.file-row:hover{transform:translateX(5px);background:#263968}
.file-badge{background:#3159DA;border-radius:9px;display:grid;place-items:center;min-width:45px;height:45px;font-weight:800;color:#fff;font-size:10px}
.file-name{font-size:12px;font-weight:800;color:#EEF3FF;overflow-wrap:anywhere}
.file-size{font-size:11px;color:#9FAED2;margin-top:3px}
.file-check{margin-left:auto;color:#44E7AC}
.st-key-chat_card{min-height:555px}
.chat-head{display:flex;align-items:center;gap:12px;padding-bottom:20px;border-bottom:1px solid #30416D;margin-bottom:18px}
.chat-avatar{height:48px;width:48px;border-radius:15px;background:linear-gradient(135deg,#FF4B60,#ED2B63);display:grid;place-items:center;color:white;font-size:26px;box-shadow:0 0 23px #FF355555;animation:avatar 3s ease-in-out infinite}
@keyframes avatar{50%{box-shadow:0 0 34px #FF355588;transform:scale(1.05)}}
.chat-title{font-size:17px;font-weight:800;color:#F4F6FF}
.chat-sub{font-size:11px;color:#97A9D1;margin-top:3px}
.intro-box{padding:30px 20px 24px;text-align:center;background:radial-gradient(circle at top,#30438A88,#152448 57%,#152045);border:1px solid #40558A;border-radius:17px;min-height:230px;display:flex;flex-direction:column;align-items:center;justify-content:center;animation:appear .6s .13s both}
.intro-icon{font-size:40px;width:78px;height:78px;background:linear-gradient(145deg,#365CF5,#6845D6);display:grid;place-items:center;border-radius:22px;margin-bottom:14px;box-shadow:0 0 40px #4268F555;animation:float2 4s ease-in-out infinite}
@keyframes float2{50%{transform:translateY(-10px)}}
.intro-title{color:#F3F6FF;font-size:22px;font-weight:900;letter-spacing:-.5px}
.intro-copy{color:#B0BCE0;max-width:400px;font-size:13px;line-height:1.8;margin-top:9px}
.step-row{display:flex;gap:9px;margin-top:17px;flex-wrap:wrap;justify-content:center}
.step-pill{background:#243663;border:1px solid #43578C;border-radius:50px;padding:7px 12px;color:#BFD0FF;font-size:11px;font-weight:700}
.small-caption{color:#A6B5D8;font-size:11px;margin:18px 0 10px;font-weight:800;letter-spacing:1.3px}
[data-testid="stChatMessage"]{background:#19294F!important;border:1px solid #304576!important;border-radius:18px!important;padding:16px 18px!important;box-shadow:none!important;animation:appear .4s ease both;margin-bottom:9px!important}
[data-testid="stChatMessage"] p{color:#EFF3FF!important}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:#233A78!important;border-color:#466BD0!important}
[data-testid="stChatInput"]{background:#182850!important;border:1px solid #4869B0!important;border-radius:16px!important;box-shadow:0 0 20px #2647CC22!important}
[data-testid="stChatInput"] textarea{color:#FFF!important}
[data-testid="stChatInput"] textarea::placeholder{color:#A5B4DA!important}
[data-testid="stChatInput"]:focus-within{border-color:#7895FF!important;box-shadow:0 0 0 4px #3C60EE33!important}
.st-key-chat_card .stButton button{border:1px solid #425C9E!important;background:#203466!important;color:#EDF3FF!important;font-weight:700!important;min-height:49px!important;transition:all .22s!important}
.st-key-chat_card .stButton button:hover{background:#3153A2!important;border-color:#92A9FF!important;transform:translateY(-3px);box-shadow:0 10px 20px #0B1A5480!important}
.st-key-chat_card [data-testid="stExpander"]{background:#1C2B50!important;border:1px solid #364B78!important;border-radius:12px}
.st-key-chat_card [data-testid="stExpander"] *{color:#D9E4FF!important}
.stats{display:flex;gap:10px;margin-bottom:16px}
.stat{flex:1;background:#20325D;border:1px solid #385287;padding:12px;border-radius:12px;text-align:center}
.stat strong{display:block;color:#fff;font-size:22px;font-weight:900}.stat small{color:#A6B9E2;font-size:10px;font-weight:800}
footer{color:#6073A5;text-align:center;font-size:11px;padding:25px 0}
@media(max-width:920px){.hero-orb{display:none}.hero-pro{padding:29px 24px}.block-container{padding:1rem!important}.topline-right{display:none}.stats{flex-wrap:wrap}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="side-brand"><span class="side-logo">✦</span> RAG STUDIO<span style="color:#FF455B">.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-status">● &nbsp; OpenAI securely connected</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-eyebrow">INTELLIGENCE SETTINGS</div>', unsafe_allow_html=True)
    model_name=st.selectbox('AI model',['gpt-4o-mini','gpt-4o','gpt-4.1-mini'])
    top_k=st.slider('Relevant sources',2,10,4)
    with st.expander('⚙ Advanced retrieval'):
        chunk_size=st.slider('Chunk size',500,2000,1000,step=100)
        chunk_overlap=st.slider('Chunk overlap',0,400,150,step=50)
    st.divider()
    st.markdown('<div class="side-eyebrow">YOUR SESSION</div>', unsafe_allow_html=True)
    if st.button('＋ New conversation',use_container_width=True):
        st.session_state.messages=[]
        st.rerun()
    if st.session_state.messages:
        st.download_button('↓ Export chat history', data=json.dumps(st.session_state.messages,ensure_ascii=False,indent=2),file_name='rag_chat.json',mime='application/json',use_container_width=True)
    if st.button('↪ Disconnect API',use_container_width=True):
        for item in ('verified','api_key','vectorstore','file_sig','messages','doc_count','chunk_count'):
            st.session_state.pop(item,None)
        st.rerun()
    st.caption('API key stays in this session. Files are processed using OpenAI embeddings and answers.')

st.markdown('<div class="topline"><div class="topline-left"><span class="topicon">✦</span> RAG Studio <span style="font-size:11px;color:#899DCF;letter-spacing:1.5px;margin-left:10px">/ INTELLIGENCE WORKSPACE</span></div><div class="topline-right"><span class="live-circle"></span> SYSTEM ONLINE <span style="opacity:.35">│</span> DOCUMENT AI</div></div>',unsafe_allow_html=True)
st.markdown("""<div class="hero-pro"><div class="hero-kicker">YOUR PERSONAL AI RESEARCH SPACE</div><div class="hero-title-pro">Your files. Your questions.<br><span class="pop">Brilliant answers.</span></div><div class="hero-desc">Upload documents, spreadsheets or data. Search, discover, and chat with all your knowledge in one place.</div><div class="hero-orb">✧</div></div>""",unsafe_allow_html=True)

left,right=st.columns([1,1.8],gap='large')
with left:
    with st.container(key='upload_card'):
        st.markdown('<div class="division-header"><div class="v3-title">◈ &nbsp; Knowledge library</div><div class="v3-tag">01 / UPLOAD</div></div><div class="v3-sub">Your workspace starts with your documents.</div>',unsafe_allow_html=True)
        uploads=st.file_uploader('Drag files here, or choose from your computer',type=['pdf','docx','xlsx','xls','csv','json','txt'],accept_multiple_files=True)
        st.markdown('<div class="formats">SUPPORTED FORMATS</div><div>'+''.join(f'<span class="format-tag">{v}</span>' for v in ('PDF','DOCX','XLSX','XLS','CSV','JSON','TXT'))+'</div>',unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="division-header"><div class="v3-title">Files in this session</div><div class="v3-tag">LIBRARY</div></div>',unsafe_allow_html=True)
        if uploads:
            for file in uploads:
                suffix=Path(file.name).suffix.upper().lstrip('.')
                st.markdown(f'<div class="file-row"><span class="file-badge">{html.escape(suffix)}</span><div style="min-width:0"><div class="file-name">{html.escape(file.name)}</div><div class="file-size">{file.size/1024:,.1f} KB · uploaded</div></div><div class="file-check">✓</div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="v3-sub">No files added yet. Drop documents into the glowing upload box above.</div>',unsafe_allow_html=True)
with right:
    with st.container(key='chat_card'):
        st.markdown('<div class="chat-head"><span class="chat-avatar">✦</span><div><div class="chat-title">AI Document Assistant</div><div class="chat-sub">Source-grounded answers · Powered by your documents</div></div></div>',unsafe_allow_html=True)
        if uploads:
            signature=(tuple((u.name,hashlib.sha256(u.getvalue()).hexdigest()) for u in uploads),chunk_size,chunk_overlap,hashlib.sha256(st.session_state.api_key.encode()).hexdigest())
            if st.session_state.get('file_sig')!=signature:
                with st.spinner('Building your searchable knowledge base…'):
                    try:
                        store,doc_count,chunk_count=build_vectorstore(uploads,st.session_state.api_key,chunk_size,chunk_overlap)
                        st.session_state.vectorstore=store
                        st.session_state.doc_count=doc_count
                        st.session_state.chunk_count=chunk_count
                        st.session_state.file_sig=signature
                        st.session_state.messages=[]
                    except Exception as exc:
                        st.error(f'Unable to index files ({type(exc).__name__}). Check document content and API quota.')
                        st.stop()
            st.markdown(f'<div class="stats"><div class="stat"><strong>{len(uploads)}</strong><small>FILES LOADED</small></div><div class="stat"><strong>{st.session_state.doc_count}</strong><small>SECTIONS</small></div><div class="stat"><strong>{st.session_state.chunk_count}</strong><small>INDEXED CHUNKS</small></div></div>',unsafe_allow_html=True)
            question=None
            if not st.session_state.messages:
                st.markdown('<div class="intro-box"><div class="intro-icon">✦</div><div class="intro-title">Knowledge unlocked.</div><div class="intro-copy">Your documents are ready. Ask a question below or use a quick action to explore their contents.</div><div class="step-row"><span class="step-pill">✓ Uploaded</span><span class="step-pill">✓ Indexed</span><span class="step-pill">✧ Ready to chat</span></div></div><div class="small-caption">QUICK ACTIONS</div>',unsafe_allow_html=True)
                a,b=st.columns(2)
                with a:
                    if st.button('✦ Summarize documents',use_container_width=True):question='Summarize the main points of the uploaded documents.'
                    if st.button('↔ Compare sources',use_container_width=True):question='Compare the main points across the uploaded documents.'
                with b:
                    if st.button('◇ Find key insights',use_container_width=True):question='What are the key findings in the uploaded documents?'
                    if st.button('▤ Extract important facts',use_container_width=True):question='List the important facts with their sources.'
            for msg in st.session_state.messages:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])
                    if msg.get('sources'):
                        with st.expander(f"◈  View {len(msg['sources'])} sources"):
                            for source in msg['sources']:
                                st.markdown(f"**{source['label']}**")
                                st.text(source['preview'])
                                st.divider()
            typed=st.chat_input('Ask your files anything…')
            if typed:question=typed
            if question:
                st.session_state.messages.append({'role':'user','content':question})
                with st.chat_message('user'):
                    st.markdown(question)
                with st.chat_message('assistant'):
                    with st.spinner('Searching files and generating an answer…'):
                        try:
                            chain,retriever=get_chain(st.session_state.vectorstore,st.session_state.api_key,model_name,top_k)
                            answer=chain.invoke(question)
                            found=retriever.invoke(question)
                            sources=[{'label':f"{d.metadata.get('source','Document')} · {d.metadata.get('location','Source')}", 'preview':d.page_content[:700]} for d in found]
                        except Exception as exc:
                            answer=f"Unable to answer ({type(exc).__name__}). Check API access and quota."
                            sources=[]
                    st.markdown(answer)
                    if sources:
                        with st.expander(f'◈ View {len(sources)} sources'):
                            for source in sources:
                                st.markdown(f"**{source['label']}**")
                                st.text(source['preview'])
                                st.divider()
                st.session_state.messages.append({'role':'assistant','content':answer,'sources':sources})
        else:
            for item in ('vectorstore','file_sig','doc_count','chunk_count'):
                st.session_state.pop(item,None)
            st.markdown('<div class="intro-box"><div class="intro-icon">▤</div><div class="intro-title">Ready when you are.</div><div class="intro-copy">Upload a PDF, Excel spreadsheet, Word document, JSON or CSV to transform your data into a searchable AI knowledge base.</div><div class="step-row"><span class="step-pill">01 · Upload files</span><span class="step-pill">02 · Index content</span><span class="step-pill">03 · Ask questions</span></div></div>',unsafe_allow_html=True)
st.markdown('<footer>RAG STUDIO · YOUR AI KNOWLEDGE WORKSPACE · LANGCHAIN × FAISS</footer>',unsafe_allow_html=True)
