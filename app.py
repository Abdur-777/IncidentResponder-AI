import streamlit as st
import PyPDF2
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from fpdf import FPDF

# === BRANDING ===
COUNCIL_NAME = "Wyndham City Council"
COUNCIL_LOGO = "https://www.wyndham.vic.gov.au/themes/custom/wyndham/logo.png"
GOV_ICON = "https://cdn-icons-png.flaticon.com/512/3209/3209872.png"

# PAGE STYLING
st.set_page_config(page_title="PolicySimplify AI", page_icon="✅", layout="centered")
st.markdown("""
    <style>
    body, .stApp { background-color: #eaf3fa; }
    .main { background-color: #eaf3fa; }
    .reportview-container { background-color: #eaf3fa; }
    .reminder { color: #fff; background:#e65c5c; padding: 6px 12px; border-radius: 6px; margin-right: 8px; }
    .reminder-upcoming { color: #fff; background:#f3c852; padding: 6px 12px; border-radius: 6px; margin-right: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown(f"""
<div style="background:#1764a7;padding:20px 0 10px 0;border-radius:16px 16px 0 0;">
    <div style="display:flex;flex-direction:column;align-items:center;gap:0;">
        <img src="{GOV_ICON}" width="40" style="margin-bottom:8px" />
        <div style="font-size:2.1em;font-weight:700;color:white;">PolicySimplify AI</div>
        <span style="color:#bfe2ff;font-size:1.08em;">Council: <b>Wyndham City Council</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:12px; margin-bottom:12px;">
    <div style="text-align:center;font-size:1.13em;color:#1764a7;">
        Upload council policies & instantly see what matters.<br>
        <span style="color:#59c12a;font-weight:500;">Australian-hosted • Secure • Unlimited uploads</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# === OPENAI KEY ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# === SESSION STATE ===
if 'obligations' not in st.session_state:
    st.session_state['obligations'] = {}  # key: filename, value: list of dicts

if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = []  # {action, file, obligation, who, time}

if 'search_text' not in st.session_state:
    st.session_state['search_text'] = ""

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("📄 Upload Policy PDF(s)", type=["pdf"], accept_multiple_files=True)

# --- EXTRACT, AI SUMMARIZE, AND PARSE OBLIGATIONS ---
def extract_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def ai_summarize(text):
    prompt = f"""
You are a compliance AI assistant for Australian councils.
Given the following policy document, provide:

1. A plain-English summary (3-5 sentences).
2. A bullet-point list of every compliance obligation, including:
   - What must be done
   - Deadline (if any)
   - Who is responsible (if possible)
   - If no deadline, suggest one if appropriate (e.g., "every year", "within 30 days")
Format your response as:
Summary:
...
Obligations:
- Obligation (deadline, responsible)
- ...
Policy text:
\"\"\"
{text[:5000]}
\"\"\"
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

def ai_chat(query, all_policy_text):
    prompt = f"""
You are a helpful AI compliance assistant. Here is the combined text of all policies uploaded:

\"\"\"{all_policy_text[:6000]}\"\"\"

Answer this council staff question using ONLY the info above. If unsure, say "Not specified in current policies."

Question: {query}
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# --- REMINDER LOGIC ---
def get_deadline_color(deadline_str):
    if not deadline_str: return None
    try:
        # Interpret as "YYYY-MM-DD" or "within 30 days" etc.
        if "within" in deadline_str or "every" in deadline_str:
            return "reminder-upcoming"
        date = pd.to_datetime(deadline_str, errors="coerce")
        if pd.isnull(date): return None
        today = pd.Timestamp.now()
        if date < today:
            return "reminder"  # Overdue
        elif (date - today).days <= 7:
            return "reminder-upcoming"  # Due soon
    except:
        return None
    return None

# --- PDF Processing ---
if uploaded_files:
    all_policy_text = ""
    for uploaded_file in uploaded_files:
        pdf_text = extract_pdf_text(uploaded_file)
        all_policy_text += "\n\n" + pdf_text

        if uploaded_file.name not in st.session_state['obligations']:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                ai_response = ai_summarize(pdf_text)
                summary_part, obligations_part = ai_response.split("Obligations:", 1) if "Obligations:" in ai_response else (ai_response, "")
                obligations_list = []
                for line in obligations_part.strip().split("\n"):
                    if line.strip().startswith("-"):
                        # Attempt to extract deadline from obligation
                        text = line.strip()[1:].strip()
                        deadline = ""
                        for kw in ["by ", "before ", "within ", "every ", "on ", "due ", "deadline:"]:
                            if kw in text.lower():
                                deadline = text[text.lower().find(kw):]
                                break
                        obligations_list.append({
                            "text": text,
                            "done": False,
                            "assigned_to": "",
                            "deadline": deadline,
                            "timestamp": None
                        })
                st.session_state['obligations'][uploaded_file.name] = {
                    "summary": summary_part.strip(),
                    "obligations": obligations_list
                }
                st.session_state['audit_log'].append({
                    "action": "upload",
                    "file": uploaded_file.name,
                    "obligation": "",
                    "who": "You",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                })

    # === REMINDERS BAR ===
    st.markdown("### ⏰ Reminders")
    reminder_count, upcoming_count = 0, 0
    for fname, doc in st.session_state['obligations'].items():
        for obl in doc["obligations"]:
            color = get_deadline_color(obl.get("deadline", ""))
            if color == "reminder":
                st.markdown(f'<span class="reminder">Overdue:</span> <b>{obl["text"]}</b>', unsafe_allow_html=True)
                reminder_count += 1
            elif color == "reminder-upcoming":
                st.markdown(f'<span class="reminder-upcoming">Due Soon:</span> <b>{obl["text"]}</b>', unsafe_allow_html=True)
                upcoming_count += 1
    if reminder_count == 0 and upcoming_count == 0:
        st.info("No overdue or upcoming deadlines!")

    # === FULL-TEXT SEARCH ===
    st.markdown("---")
    st.markdown("### 🔍 Full-Text Search")
    search_text = st.text_input("Search all obligations, summaries, and policies...", key="search")
    st.session_state['search_text'] = search_text
    dashboard_data = []
    for fname, doc in st.session_state['obligations'].items():
        for obl in doc["obligations"]:
            match = (
                (search_text.lower() in obl["text"].lower() if search_text else True) or
                (search_text.lower() in doc["summary"].lower() if search_text else True)
            )
            if match:
                dashboard_data.append({
                    "Filename": fname,
                    "Summary": doc["summary"][:100]+"..." if len(doc["summary"]) > 100 else doc["summary"],
                    "Obligation": obl["text"],
                    "Done": "✅" if obl["done"] else "⬜️",
                    "Assigned to": obl.get("assigned_to",""),
                    "Deadline": obl.get("deadline",""),
                    "Timestamp": obl.get("timestamp","")
                })
    if dashboard_data:
        df = pd.DataFrame(dashboard_data)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="Download Obligations CSV",
            data=df.to_csv(index=False),
            file_name="policy_obligations.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching obligations found.")

    # === OBLIGATION CARDS ===
    st.markdown("---")
    for fname, doc in st.session_state['obligations'].items():
        with st.expander(f"📑 {fname}", expanded=False):
            st.markdown(f"**Summary:**<br>{doc['summary']}", unsafe_allow_html=True)
            st.markdown("**Obligations & Actions:**")
            for idx, obl in enumerate(doc['obligations']):
                cols = st.columns([0.07,0.68,0.13,0.12])
                with cols[0]:
                    checked = st.checkbox("", value=obl['done'], key=f"{fname}_check_{idx}")
                    if checked != obl['done']:
                        doc['obligations'][idx]['done'] = checked
                        doc['obligations'][idx]['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.session_state['audit_log'].append({
                            "action": "check" if checked else "uncheck",
                            "file": fname,
                            "obligation": obl['text'],
                            "who": "You",
                            "time": doc['obligations'][idx]['timestamp']
                        })
                with cols[1]:
                    st.markdown(obl['text'])
                with cols[2]:
                    assigned_to = st.text_input(
                        "Assign", value=obl.get("assigned_to",""), key=f"{fname}_assign_{idx}", label_visibility="collapsed", placeholder="Assign to"
                    )
                    if assigned_to != obl.get("assigned_to",""):
                        doc['obligations'][idx]['assigned_to'] = assigned_to
                        st.session_state['audit_log'].append({
                            "action": "assign",
                            "file": fname,
                            "obligation": obl['text'],
                            "who": assigned_to,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                with cols[3]:
                    deadline = st.text_input(
                        "Deadline", value=obl.get("deadline",""), key=f"{fname}_deadline_{idx}", label_visibility="collapsed", placeholder="Deadline (YYYY-MM-DD)"
                    )
                    if deadline != obl.get("deadline",""):
                        doc['obligations'][idx]['deadline'] = deadline
                        st.session_state['audit_log'].append({
                            "action": "deadline_change",
                            "file": fname,
                            "obligation": obl['text'],
                            "who": "You",
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
            st.caption("AI-generated. Please review obligations before action.")

    # === POLICY Q&A CHAT ===
    st.markdown("---")
    st.markdown("## 🤖 Ask Your Policies (AI Chat)")
    st.caption("Type a question about your policies. The AI answers ONLY using your uploaded documents.")
    query = st.text_input("Ask a policy/compliance question", key="policy_qa")
    if query:
        with st.spinner("Getting answer..."):
            answer = ai_chat(query, all_policy_text)
        st.success(answer)

    # === AUDIT LOG + PDF EXPORT ===
    st.markdown("---")
    st.markdown("## 🕵️ Audit Log & One-Click Audit Pack")
    st.caption("All major actions are tracked for compliance and audit reporting.")
    audit_df = pd.DataFrame(st.session_state['audit_log'])
    st.dataframe(audit_df, use_container_width=True)
    st.download_button(
        label="Download Audit Log CSV",
        data=audit_df.to_csv(index=False),
        file_name="audit_log.csv",
        mime="text/csv"
    )

    # --- Branded PDF Export ---
    def export_pdf(dataframe, filename="policy_obligations.pdf", title="Obligation Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(23, 100, 167)
        pdf.cell(0, 10, COUNCIL_NAME, ln=1, align="C")
        pdf.image(COUNCIL_LOGO, x=10, y=12, w=30)
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(23, 100, 167)
        pdf.cell(0, 10, title, ln=1)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        for i, row in dataframe.iterrows():
            pdf.cell(0, 8, f"{i+1}. {row['Obligation']} | {row['Done']} | Assigned: {row['Assigned to']} | Deadline: {row['Deadline']}", ln=1)
        pdf.output(filename)
        with open(filename, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes

    st.markdown("### 📄 Download Pretty PDF Export")
    if len(dashboard_data) > 0:
        if st.button("Download Compliance PDF"):
            pdf_bytes = export_pdf(pd.DataFrame(dashboard_data), title="Obligations Compliance Pack")
            st.download_button("Download PDF", pdf_bytes, file_name="policy_obligations.pdf", mime="application/pdf")

    st.markdown("### 🏛️ Download Branded Audit Pack PDF")
    if len(audit_df) > 0:
        if st.button("Download Audit Pack PDF"):
            pdf_bytes = export_pdf(audit_df.rename(columns={"action":"Obligation"}), filename="audit_pack.pdf", title="Audit Log Pack")
            st.download_button("Download PDF", pdf_bytes, file_name="audit_pack.pdf", mime="application/pdf")

else:
    st.info("Upload one or more council policy PDFs to begin.")

st.markdown("---")
st.markdown("""
<span style='color: #59c12a; font-weight:bold;'>PolicySimplify AI – Built for Australian councils. All data hosted securely in Australia.</span>
""", unsafe_allow_html=True)
