import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@st.cache_resource(show_spinner="Loading MPNet model (first run only)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9)


def generate_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=256):

    embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pooling(outputs, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)

        embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings)

    return embeddings
