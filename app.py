import streamlit as st
import fitz
import requests
import pandas as pd
import json
import re
import io
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak
)

from similarity import compare_papers

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="ResearchGap AI",
    page_icon="📚",
    layout="wide"
)

MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
HISTORY_FILE = Path("research_history.json")

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fc;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #172554;
}
[data-testid="stSidebar"] * {
    color: white;
}
.hero {
    background: linear-gradient(120deg,#172554,#4338ca,#6366f1);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

defaults = {
    "papers": [],
    "topic": "",
    "literature": [],
    "evidence": [],
    "research_gap": "",
    "problem_statement": "",
    "future_directions": [],
    "analysis_complete": False,
    "analysis_time": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- PDF EXTRACTION ----------------

def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_title_year(doc):
    metadata = doc.metadata or {}

    year = ""

    text = doc[0].get_text("text") if len(doc) else ""

    match = re.search(r"\b(19[89]\d|20[0-2]\d)\b", text[:6000])

    if match:
        year = match.group(1)

    if not year:
        match = re.search(
            r"\b(19[89]\d|20[0-2]\d)\b",
            str(metadata)
        )
        if match:
            year = match.group(1)

    title = clean_text(metadata.get("title", ""))

    if title and len(title) > 12:
        return title, year or "Year not detected"

    # Find the largest title-like text on the first page.
    spans = []

    page = doc[0]
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                value = clean_text(span.get("text", ""))
                size = span.get("size", 0)
                y_position = span.get("bbox", [0, 0, 0, 0])[1]

                if (
                    value
                    and len(value) > 8
                    and y_position < page.rect.height * 0.45
                    and not re.search(
                        r"doi:|https?://|received|accepted|copyright",
                        value,
                        re.I
                    )
                ):
                    spans.append((size, y_position, value))

    if spans:
        largest = max(item[0] for item in spans)

        title_spans = [
            item for item in spans
            if item[0] >= largest * 0.75
        ]

        title_spans.sort(key=lambda item: item[1])

        title = clean_text(
            " ".join(item[2] for item in title_spans)
        )

    if not title or len(title) < 12:
        title = "Title not detected in PDF"

    return title, year or "Year not detected"


def process_pdf(uploaded_file):
    doc = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    title, year = extract_title_year(doc)

    pages = []

    for index, page in enumerate(doc):
        pages.append({
            "page": index + 1,
            "text": page.get_text("text")
        })

    full_text = "\n".join(
        page["text"] for page in pages
    )

    doc.close()

    return {
        "title": title,
        "year": year,
        "pages": pages,
        "text": full_text
    }


# ---------------- EVIDENCE EXTRACTION ----------------

PATTERNS = {
    "Limitations": r"\blimitation(s)?\b|\bshortcoming(s)?\b|\bunable to\b",
    "Future Work": r"\bfuture work\b|\bfuture research\b|\bfurther investigation\b",
    "Challenges": r"\bchallenge(s)?\b|\bdifficult(y|ies)\b",
    "Unresolved Issues": r"\bunresolved\b|\bremains unclear\b|\bnot yet addressed\b"
}


def extract_evidence(papers):
    evidence = []
    seen = set()

    for paper in papers:
        for page in paper["pages"]:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                re.sub(r"\s+", " ", page["text"])
            )

            for sentence in sentences:

                sentence = clean_text(sentence)

                if not 50 <= len(sentence) <= 450:
                    continue

                for category, pattern in PATTERNS.items():

                    if re.search(pattern, sentence, re.I):

                        key = (
                            paper["title"],
                            page["page"],
                            sentence
                        )

                        if key not in seen:
                            seen.add(key)

                            evidence.append({
                                "Paper Title": paper["title"],
                                "Year": paper["year"],
                                "Page": page["page"],
                                "Category": category,
                                "Evidence": sentence,
                                "Review Status": "Needs verification"
                            })

                        break

    return evidence[:100]


# ---------------- OLLAMA AI ----------------

def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3.5:4b",
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 3000
                }
            },
            timeout=900
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            answer = data.get("message", {}).get("content", "").strip()

        if not answer:
            raise ValueError("Qwen returned an empty response.")

        return answer

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama connection failed: {e}")


def parse_ai_json(answer):

    answer = re.sub(
        r"^```(?:json)?\s*",
        "",
        answer.strip(),
        flags=re.I
    )

    answer = re.sub(r"\s*```$", "", answer)

    start = answer.find("{")
    end = answer.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("AI did not return valid JSON")

    return json.loads(answer[start:end + 1])


def generate_analysis(papers, topic, evidence):

    paper_context = []

    for index, paper in enumerate(papers, 1):

        paper_context.append(f"""
PAPER {index}

TITLE: {paper['title']}
YEAR: {paper['year']}

CONTENT:
{paper['text'][:6500]}
""")

    evidence_context = "\n".join(
        f"{item['Paper Title']} ({item['Year']}), "
        f"Page {item['Page']}, {item['Category']}: "
        f"{item['Evidence']}"
        for item in evidence[:35]
    )

    prompt = f"""
You are an academic research analysis assistant.

Research topic: {topic}

UPLOADED PAPERS:

{chr(10).join(paper_context)}

EXTRACTED EVIDENCE:

{evidence_context}

Return ONLY valid JSON.

Use exactly this structure:

{{
 "literature_survey": [
  {{
   "advantages": "Advantages of the paper",
   "limitations": "Limitations of the paper",
   "tools_technologies": "Tools and technologies used"
  }}
 ],
 "research_gap": "One concise paragraph of 80 to 110 words",
 "problem_statement": "Academic problem statement of 100 to 130 words",
 "future_directions": [
  "Direction 1",
  "Direction 2",
  "Direction 3"
 ]
}}

STRICT RULES:

1. Return exactly one literature survey entry per uploaded paper.
2. Do not include authors.
3. Do not include filenames.
4. Do not invent titles, years, methods or results.
5. The paper title and year will be added from the PDF extraction by the application.
6. Literature survey must contain advantages, limitations and tools/technologies.
7. Do not use the word disadvantages.
8. Research gap must be ONE short paragraph.
9. Do not create a long report.
10. Use only information supported by the uploaded papers.
11. If information is missing, say it is not clearly stated.
"""

    answer = ask_ollama(prompt)

    return parse_ai_json(answer)


# ---------------- HISTORY ----------------

def load_history():

    try:
        data = json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )

        return data if isinstance(data, list) else []

    except Exception:
        return []


def save_history():

    history = load_history()

    record = {
        "timestamp": st.session_state.analysis_time,
        "topic": st.session_state.topic,
        "papers": [
            {
                "title": p["title"],
                "year": p["year"]
            }
            for p in st.session_state.papers
        ],
        "literature": st.session_state.literature,
        "evidence": st.session_state.evidence,
        "research_gap": st.session_state.research_gap,
        "problem_statement": st.session_state.problem_statement,
        "future_directions": st.session_state.future_directions
    }

    history.insert(0, record)

    HISTORY_FILE.write_text(
        json.dumps(history[:50], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ---------------- PDF REPORT ----------------

def create_pdf_report():

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="SmallCell",
            parent=styles["BodyText"],
            fontSize=7,
            leading=9
        )
    )

    story = []

    story.append(
        Paragraph(
            "ResearchGap AI – Research Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            f"<b>Research Topic:</b> "
            f"{escape(st.session_state.topic)}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Analysis Date:</b> "
            f"{escape(st.session_state.analysis_time)}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph("Literature Survey", styles["Heading1"])
    )

    data = [[
        "Paper (Title & Year)",
        "Advantages",
        "Limitations",
        "Tools & Technologies"
    ]]

    for item in st.session_state.literature:

        data.append([
            Paragraph(
                escape(item["paper"]),
                styles["SmallCell"]
            ),
            Paragraph(
                escape(item["advantages"]),
                styles["SmallCell"]
            ),
            Paragraph(
                escape(item["limitations"]),
                styles["SmallCell"]
            ),
            Paragraph(
                escape(item["tools_technologies"]),
                styles["SmallCell"]
            )
        ])

    table = Table(
        data,
        colWidths=[40*mm, 42*mm, 42*mm, 42*mm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#233876")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2ff")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph("Research Gap", styles["Heading1"])
    )

    story.append(
        Paragraph(
            escape(st.session_state.research_gap),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph("Problem Statement", styles["Heading1"])
    )

    story.append(
        Paragraph(
            escape(st.session_state.problem_statement),
            styles["BodyText"]
        )
    )

    if st.session_state.future_directions:

        story.append(Spacer(1, 18))

        story.append(
            Paragraph(
                "Future Research Directions",
                styles["Heading1"]
            )
        )

        for direction in st.session_state.future_directions:
            story.append(
                Paragraph(
                    "• " + escape(str(direction)),
                    styles["BodyText"]
                )
            )

    if st.session_state.evidence:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Extracted Research Evidence",
                styles["Heading1"]
            )
        )

        for item in st.session_state.evidence:

            story.append(
                Paragraph(
                    f"<b>{escape(item['Paper Title'])} "
                    f"({escape(str(item['Year']))}), "
                    f"Page {item['Page']} – "
                    f"{escape(item['Category'])}</b><br/>"
                    f"{escape(item['Evidence'])}",
                    styles["BodyText"]
                )
            )

            story.append(Spacer(1, 8))

    pdf.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("# 📚 ResearchGap AI")

    st.caption("Automated Research Gap Identification using QwenAI")

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Upload & Analyze",
            "Compare Papers",
            "Research Guides",
            "Problem Statement",
            "History",
            "Reports"
        ]
    )

    st.divider()

    st.caption("LOCAL AI MODEL")

    st.write("Qwen 3.5 4B · Ollama")


# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to ResearchGap AI</h1>
        <p>
        An intelligent automated research gap identification platform.
        Discover existing studies, understand research limitations,
        identify possible research gaps and develop problem statements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📚 Literature Survey")
        st.write(
            "Organize research papers, methods, advantages, "
            "limitations and technologies."
        )

    with col2:
        st.subheader("🔍 Research Gap Discovery")
        st.write(
            "Examine limitations, challenges and unresolved "
            "research problems."
        )

    with col3:
        st.subheader("🧠 Evidence Knowledge Graph")
        st.write(
            "Explore relationships between papers, evidence, "
            "limitations and research gaps."
        )

    st.subheader("How to use ResearchGap AI")

    st.markdown("""
    **Step 1:** Enter your research topic and upload research papers.

    **Step 2:** Process PDFs and generate the literature survey.

    **Step 3:** Explore research gaps and extracted evidence.

    **Step 4:** Compare papers and explore the Evidence Knowledge Graph.

    **Step 5:** Review your problem statement and download your PDF report.
    """)

    a, b, c = st.columns(3)

    a.metric("Papers uploaded", len(st.session_state.papers))

    b.metric("Evidence passages", len(st.session_state.evidence))

    c.metric(
        "Analysis status",
        "Complete" if st.session_state.analysis_complete else "Not started"
    )


# =====================================================
# UPLOAD AND ANALYZE
# =====================================================

elif page == "Upload & Analyze":

    st.title("Upload & Analyze Research Papers")

    st.write("Follow the steps below to analyze your research papers.")

    topic = st.text_input(
        "1. Enter your research topic",
        value=st.session_state.topic,
        placeholder="Enter your research topic"
    )

    files = st.file_uploader(
        "2. Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("3. Process PDF Files", type="primary"):

        if not topic.strip():
            st.warning("Enter your research topic.")

        elif not files:
            st.warning("Upload at least one PDF.")

        else:

            papers = []

            with st.spinner("Extracting PDF content and paper titles..."):

                for uploaded_file in files:

                    try:
                        papers.append(process_pdf(uploaded_file))

                    except Exception as error:
                        st.error(
                            f"Could not process a PDF: {error}"
                        )

            if papers:

                st.session_state.papers = papers
                st.session_state.topic = topic.strip()

                st.session_state.literature = []
                st.session_state.evidence = []
                st.session_state.research_gap = ""
                st.session_state.problem_statement = ""
                st.session_state.future_directions = []
                st.session_state.analysis_complete = False

                st.success("PDF processing completed.")

    if st.session_state.papers:

        st.subheader("Uploaded Research Papers")

        uploaded_table = pd.DataFrame([
            {
                "Paper Title": paper["title"],
                "Year": paper["year"]
            }
            for paper in st.session_state.papers
        ])

        st.dataframe(
            uploaded_table,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Analyze Papers with Qwen AI", type="primary"):

            try:

                with st.spinner(
                    "Qwen is analyzing your papers. "
                    "Please wait..."
                ):

                    evidence = extract_evidence(
                        st.session_state.papers
                    )

                    result = generate_analysis(
                        st.session_state.papers,
                        st.session_state.topic,
                        evidence
                    )

                ai_rows = result.get("literature_survey", [])

                literature = []

                for index, paper in enumerate(
                    st.session_state.papers
                ):

                    ai = (
                        ai_rows[index]
                        if index < len(ai_rows)
                        else {}
                    )

                    literature.append({
                        "paper": (
                            f"{paper['title']} ({paper['year']})"
                        ),
                        "advantages": ai.get(
                            "advantages",
                            "Not clearly stated in the paper"
                        ),
                        "limitations": ai.get(
                            "limitations",
                            "Not clearly stated in the paper"
                        ),
                        "tools_technologies": ai.get(
                            "tools_technologies",
                            "Not clearly stated in the paper"
                        )
                    })

                st.session_state.literature = literature
                st.session_state.evidence = evidence

                st.session_state.research_gap = clean_text(
                    result.get("research_gap", "")
                )

                st.session_state.problem_statement = clean_text(
                    result.get("problem_statement", "")
                )

                st.session_state.future_directions = (
                    result.get("future_directions", [])
                )

                st.session_state.analysis_time = (
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                )

                st.session_state.analysis_complete = True

                save_history()

                st.success("AI analysis completed successfully!")

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to Ollama. "
                    "Make sure Ollama is running."
                )

            except Exception as error:

                st.error(f"Analysis failed: {error}")

        if st.session_state.analysis_complete:

            st.subheader("Literature Survey")

            st.dataframe(
                pd.DataFrame(st.session_state.literature),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "paper": st.column_config.TextColumn(
                        "Paper (Title & Year)",
                        width="large"
                    ),
                    "advantages": st.column_config.TextColumn(
                        "Advantages",
                        width="large"
                    ),
                    "limitations": st.column_config.TextColumn(
                        "Limitations",
                        width="large"
                    ),
                    "tools_technologies": st.column_config.TextColumn(
                        "Tools & Technologies",
                        width="large"
                    )
                }
            )


# =====================================================
# COMPARE PAPERS
# =====================================================

elif page == "Compare Papers":

    st.title("Compare Research Papers")

    if len(st.session_state.papers) < 2:

        st.info(
            "Upload at least two papers in Upload & Analyze first."
        )

    else:

        paper_names = [
            f"{p['title']} ({p['year']})"
            for p in st.session_state.papers
        ]

        selected = st.multiselect(
            "Select papers",
            paper_names,
            default=paper_names
        )

        if st.button("Compare Selected Papers"):

            if len(selected) < 2:

                st.warning("Select at least two papers.")

            else:

                chosen = [
                    p for p in st.session_state.papers
                    if f"{p['title']} ({p['year']})" in selected
                ]

                with st.spinner("Calculating similarity..."):

                    scores = compare_papers([
                        paper["text"][:12000]
                        for paper in chosen
                    ])

                labels = [
                    f"{p['title']} ({p['year']})"
                    for p in chosen
                ]

                result = pd.DataFrame(
                    scores * 100,
                    index=labels,
                    columns=labels
                ).round(2)

                st.subheader("Semantic Similarity (%)")

                st.dataframe(
                    result,
                    use_container_width=True
                )

                st.caption(
                    "Similarity indicates textual closeness, "
                    "not research quality or novelty."
                )


# =====================================================
# RESEARCH GUIDES
# =====================================================

elif page == "Research Guides":

    st.title("Research Guides")

    if not st.session_state.analysis_complete:

        st.info(
            "Run AI analysis in Upload & Analyze first."
        )

    else:

        # Research gap appears ONLY here.
        st.subheader("AI Research Gap")

        st.write(st.session_state.research_gap)

        st.divider()

        # Evidence explorer
        st.subheader("Evidence Explorer")

        if st.session_state.evidence:

            df = pd.DataFrame(
                st.session_state.evidence
            )

            categories = [
                "All"
            ] + sorted(df["Category"].unique().tolist())

            selected_category = st.selectbox(
                "Filter evidence category",
                categories
            )

            if selected_category != "All":

                df = df[
                    df["Category"] == selected_category
                ]

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Evidence CSV",
                df.to_csv(index=False).encode("utf-8"),
                "ResearchGapAI_Evidence.csv",
                "text/csv"
            )

        else:

            st.info(
                "No evidence passages were detected."
            )

        # Actual knowledge graph
        st.divider()

        st.subheader("Evidence Knowledge Graph")

        st.write(
            "This graph connects paper titles to extracted "
            "evidence, issue categories and the candidate research gap."
        )

        if st.session_state.evidence:

            graph = [
                "digraph ResearchGap {",
                'rankdir="LR";',
                'node [shape=box, style="rounded,filled", fontname="Arial"];',
                'edge [color="#64748b"];'
            ]

            categories_seen = set()

            for i, item in enumerate(
                st.session_state.evidence[:25]
            ):

                paper_id = f"paper{i}"
                evidence_id = f"evidence{i}"

                category_id = re.sub(
                    r"[^a-zA-Z0-9]",
                    "_",
                    item["Category"]
                )

                paper_label = (
                    f"{item['Paper Title']} ({item['Year']})"
                )

                evidence_label = (
                    item["Evidence"][:90]
                )

                graph.append(
                    f'{paper_id} [label="{escape(paper_label)}", '
                    'fillcolor="#dbeafe"];'
                )

                graph.append(
                    f'{evidence_id} [label="{escape(evidence_label)}", '
                    'fillcolor="#f8fafc"];'
                )

                graph.append(
                    f'{category_id} [label="{escape(item["Category"])}", '
                    'fillcolor="#ede9fe"];'
                )

                graph.append(
                    f"{paper_id} -> {evidence_id};"
                )

                graph.append(
                    f"{evidence_id} -> {category_id};"
                )

                categories_seen.add(category_id)

            graph.append(
                'researchgap [label="Candidate Research Gap", '
                'fillcolor="#dcfce7"];'
            )

            for category_id in categories_seen:

                graph.append(
                    f"{category_id} -> researchgap;"
                )

            graph.append("}")

            st.graphviz_chart(
                "\n".join(graph),
                use_container_width=True
            )

        else:

            st.info(
                "Evidence is needed to build the knowledge graph."
            )

        st.caption(
            "Automatically extracted evidence must be verified "
            "against the original PDF."
        )

        if st.session_state.future_directions:

            st.subheader("Future Research Directions")

            for direction in st.session_state.future_directions:

                st.markdown(f"- {direction}")


# =====================================================
# PROBLEM STATEMENT
# =====================================================

elif page == "Problem Statement":

    st.title("Problem Statement")

    if not st.session_state.analysis_complete:

        st.info(
            "Run AI analysis in Upload & Analyze first."
        )

    else:

        st.write(
            st.session_state.problem_statement
        )

        st.caption(
            "AI-generated draft. Verify claims against the original papers."
        )


# =====================================================
# HISTORY
# =====================================================

elif page == "History":

    st.title("Research History")

    history = load_history()

    if not history:

        st.info("No saved research history.")

    else:

        for record in history:

            title = (
                f"{record.get('timestamp', '')} — "
                f"{record.get('topic', 'Research topic')}"
            )

            with st.expander(title):

                st.write("**Papers**")

                for paper in record.get("papers", []):

                    st.write(
                        f"- {paper.get('title')} "
                        f"({paper.get('year')})"
                    )

                st.write("**Research Gap**")

                st.write(
                    record.get("research_gap", "")
                )

                if record.get("literature"):

                    st.write("**Literature Survey**")

                    st.dataframe(
                        pd.DataFrame(record["literature"]),
                        use_container_width=True,
                        hide_index=True
                    )

    if history and st.button("Clear History"):

        HISTORY_FILE.write_text(
            "[]",
            encoding="utf-8"
        )

        st.rerun()


# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":

    st.title("Research Reports")

    if not st.session_state.analysis_complete:

        st.info(
            "Complete AI analysis before downloading your report."
        )

    else:

        st.write(
            "Download your complete research analysis "
            "as a professionally formatted PDF."
        )

        pdf = create_pdf_report()

        st.download_button(
            "Download ResearchGap AI Report (PDF)",
            data=pdf,
            file_name="ResearchGapAI_Report.pdf",
            mime="application/pdf",
            type="primary"
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ResearchGap AI | Powered by Qwen 3.5 4B through Ollama"
)