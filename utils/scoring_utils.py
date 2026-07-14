import faiss
import numpy as np
import pandas as pd

SIMILARITY_THRESHOLD = 0.75
STRONG_THRESHOLD = 0.80
WEAK_THRESHOLD = 0.60


def build_faiss_index(jd_embeddings):
    """
    Builds a FAISS Inner-Product index over JD chunk embeddings,
    same as: faiss.IndexFlatIP(embedding_dimension)
    """
    jd_embeddings = jd_embeddings.astype(np.float32)
    embedding_dimension = jd_embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    faiss_index.add(jd_embeddings)

    return faiss_index


def retrieve_top_matches(faiss_index, resume_embedding, top_k=5):

    top_k = min(top_k, faiss_index.ntotal)

    scores, indices = faiss_index.search(
        resume_embedding.reshape(1, -1).astype(np.float32), top_k
    )
    return scores[0], indices[0]


def build_alignment_matrix(resume_chunks, resume_embeddings, jd_chunks, faiss_index,
                            top_k=5, boilerplate_flags=None):
    """
    resume_chunks : list of resume chunk text strings
    jd_chunks     : list of jd chunk text strings (order must match faiss_index)
    boilerplate_flags : optional list[bool], same length/order as resume_chunks.
                         True marks a chunk as non-substantive (header/contact/
                         declaration text). If omitted, all chunks are treated
                         as substantive (is_boilerplate=False).
    """

    if boilerplate_flags is None:
        boilerplate_flags = [False] * len(resume_chunks)

    alignment_records = []

    for chunk_id, resume_chunk_text in enumerate(resume_chunks):

        scores, indices = retrieve_top_matches(
            faiss_index, resume_embeddings[chunk_id], top_k
        )

        for rank, (score, jd_idx) in enumerate(zip(scores, indices), start=1):

            alignment_records.append({
                "resume_chunk_id": chunk_id,
                "resume_chunk": resume_chunk_text,
                "jd_chunk_id": int(jd_idx),
                "jd_chunk": jd_chunks[jd_idx],
                "similarity": float(score),
                "rank": rank,
                "is_boilerplate": boilerplate_flags[chunk_id]
            })

    alignment_df = pd.DataFrame(alignment_records)
    return alignment_df


def compute_best_alignment(alignment_df):
    """
    Best (highest similarity) JD match for every resume chunk.
    """
    best_alignment = (
        alignment_df.sort_values("similarity", ascending=False)
        .groupby("resume_chunk_id")
        .first()
        .reset_index()
    )
    return best_alignment


def compute_ats_metrics(best_alignment):
    """
    Replicates notebook's ats_metrics computation, collapsed to a
    single resume (single row of metrics).
    """

    similarity = best_alignment["similarity"]

    mean_similarity = similarity.mean()
    max_similarity = similarity.max()
    min_similarity = similarity.min()
    std_similarity = similarity.std()
    number_of_chunks = similarity.count()

    high_confidence_matches = int((similarity >= SIMILARITY_THRESHOLD).sum())

    coverage = high_confidence_matches / number_of_chunks if number_of_chunks else 0

    ats_score = (0.70 * mean_similarity + 0.30 * coverage) * 100
    ats_score = round(ats_score, 2)

    metrics = {
        "Mean_Similarity": mean_similarity,
        "Max_Similarity": max_similarity,
        "Min_Similarity": min_similarity,
        "Std_Similarity": std_similarity,
        "Number_of_Chunks": number_of_chunks,
        "High_Confidence_Matches": high_confidence_matches,
        "Coverage": coverage,
        "ATS_Score": ats_score,
        "Grade": assign_grade(ats_score)
    }

    return metrics


def assign_grade(score):

    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Average"
    else:
        return "Needs Improvement"


def get_strong_weak_matches(best_alignment):

    strong_matches = best_alignment[best_alignment["similarity"] >= STRONG_THRESHOLD]
    weak_matches = best_alignment[best_alignment["similarity"] < WEAK_THRESHOLD]

    return strong_matches, weak_matches
