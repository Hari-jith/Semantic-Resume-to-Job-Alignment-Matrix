import nltk
import streamlit as st
import pandas as pd

from utils.text_utils import (
    clean_text,
    create_chunks,
    is_boilerplate_chunk,
)
from utils.embedding_utils import load_model, generate_embeddings
from utils.file_utils import extract_text_from_file
from utils.scoring_utils import (
    build_faiss_index,
    build_alignment_matrix,
    compute_best_alignment,
    compute_ats_metrics,
    get_strong_weak_matches,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(
    page_title="Semantic Resume-JD Alignment (ATS)",
    layout="wide"
)

st.title("Semantic Resume Alignment / ATS Scorer")
st.caption(
    "Chunk-level semantic matching between a resume and a job description, "
    "using all-mpnet-base-v2 embeddings + FAISS cosine similarity search."
)

# ------------------------------------------------------------
# Sidebar: parameters (defaults match the notebook)
# ------------------------------------------------------------
st.sidebar.header("Chunking Parameters")
chunk_size = st.sidebar.slider("Chunk size (sentences)", 2, 10, 5)
overlap = st.sidebar.slider("Chunk overlap (sentences)", 0, chunk_size - 1, 2)
top_k = st.sidebar.slider("Top-K matches per resume chunk", 1, 10, 5)

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload resume (.pdf, .docx, .txt)", key="resume")
    resume_text_input = st.text_area("...or paste resume text", height=200, key="resume_text")

with col2:
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload JD (.pdf, .docx, .txt)", key="jd")
    jd_text_input = st.text_area("...or paste job description text", height=200, key="jd_text")

run_button = st.button("Run Semantic Alignment", type="primary")

# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
if run_button:

    # --- Resolve raw text ---
    raw_resume_text = extract_text_from_file(resume_file) if resume_file else resume_text_input
    raw_jd_text = extract_text_from_file(jd_file) if jd_file else jd_text_input

    if not raw_resume_text.strip() or not raw_jd_text.strip():
        st.error("Please provide both a resume and a job description (upload or paste).")
        st.stop()

    # --- Clean text (same as notebook clean_text) ---
    with st.spinner("Cleaning text..."):
        resume_clean = clean_text(raw_resume_text)
        jd_clean = clean_text(raw_jd_text)

    # --- Chunk text (same as notebook create_chunks) ---
    with st.spinner("Creating semantic chunks..."):
        resume_chunks = create_chunks(resume_clean, chunk_size=chunk_size, overlap=overlap)
        jd_chunks = create_chunks(jd_clean, chunk_size=chunk_size, overlap=overlap)

    if len(resume_chunks) == 0 or len(jd_chunks) == 0:
        st.error("Could not generate chunks from the provided text. Please check the input.")
        st.stop()

    st.success(f"Resume split into {len(resume_chunks)} chunks. JD split into {len(jd_chunks)} chunks.")

    # --- Flag non-substantive chunks (contact/header blocks, declarations) ---
    boilerplate_flags = [is_boilerplate_chunk(chunk) for chunk in resume_chunks]
    excluded_count = sum(boilerplate_flags)

    if excluded_count:
        st.info(
            f"{excluded_count} resume chunk(s) look like header/contact info or "
            f"declaration boilerplate and will be excluded from the ATS score "
            f"(still shown below, tagged)."
        )

    # --- Load model ---
    tokenizer, model, device = load_model()

    # --- Generate embeddings (same as notebook generate_embeddings) ---
    with st.spinner("Generating embeddings..."):
        resume_embeddings = generate_embeddings(resume_chunks, tokenizer, model, device)
        jd_embeddings = generate_embeddings(jd_chunks, tokenizer, model, device)

    # --- Build FAISS index over JD chunks ---
    with st.spinner("Building FAISS index and running semantic search..."):
        faiss_index = build_faiss_index(jd_embeddings)

        alignment_df = build_alignment_matrix(
            resume_chunks, resume_embeddings, jd_chunks, faiss_index,
            top_k=top_k, boilerplate_flags=boilerplate_flags
        )
        best_alignment = compute_best_alignment(alignment_df)

        # Score only on substantive chunks; fall back to all chunks if
        # everything got flagged (e.g. a very short resume).
        substantive_alignment = best_alignment[~best_alignment["is_boilerplate"]]
        if substantive_alignment.empty:
            st.warning(
                "All resume chunks were flagged as boilerplate — scoring on the "
                "full set instead. Check that the resume text extracted correctly."
            )
            substantive_alignment = best_alignment

        ats_metrics = compute_ats_metrics(substantive_alignment)
        strong_matches, weak_matches = get_strong_weak_matches(substantive_alignment)

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------
    st.header("ATS Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ATS Score", f"{ats_metrics['ATS_Score']}")
    m2.metric("Grade", ats_metrics["Grade"])
    m3.metric("Mean Similarity", f"{ats_metrics['Mean_Similarity']:.4f}")
    m4.metric("Coverage", f"{ats_metrics['Coverage']:.2%}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Chunks Analyzed", ats_metrics["Number_of_Chunks"])
    m6.metric("High-Confidence Matches (>=0.75)", ats_metrics["High_Confidence_Matches"])
    m7.metric("Std. Dev. of Similarity", f"{ats_metrics['Std_Similarity']:.4f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Explainable Report", "Strong Matches", "Weak Matches"])

    with tab1:
        st.subheader("Best JD Match per Resume Chunk")
        st.caption("Rows marked \"boilerplate\" were excluded from the ATS score above.")
        report = best_alignment.sort_values("similarity", ascending=False).copy()
        report["status"] = report["is_boilerplate"].map(
            {True: "boilerplate (excluded)", False: "scored"}
        )
        st.dataframe(
            report[["resume_chunk", "jd_chunk", "similarity", "status"]],
            use_container_width=True,
            height=500
        )

    with tab2:
        st.subheader(f"Strong Matches (similarity >= 0.80) — {len(strong_matches)}")
        st.dataframe(
            strong_matches[["resume_chunk", "jd_chunk", "similarity"]],
            use_container_width=True
        )

    with tab3:
        st.subheader(f"Weak / Uncovered Resume Chunks (similarity < 0.60) — {len(weak_matches)}")
        st.dataframe(
            weak_matches[["resume_chunk", "jd_chunk", "similarity"]],
            use_container_width=True
        )

    with st.expander("Full Alignment Matrix (all top-K matches)"):
        st.dataframe(
            alignment_df[[
                "resume_chunk_id", "resume_chunk", "jd_chunk_id", "jd_chunk",
                "similarity", "rank", "is_boilerplate"
            ]],
            use_container_width=True
        )
