# senseqnet/inference.py

import os
import torch
import numpy as np
from Bio import SeqIO
import esm

from senseqnet.models import ImprovedLSTMClassifier

MODEL_FILENAME = "SenSeqNet_model.pth"
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"

def load_pretrained_model(device="cuda"):
    """
    Initializes the ImprovedLSTMClassifier with your chosen hyperparams,
    loads the 'SenSeqNet_model.pth' checkpoint, and returns the model in eval mode.
    """

    checkpoint_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

    input_dim = 1280        # for ESM2_t33_650M_UR50D
    hidden_dim = 181
    num_layers = 4
    dropout_rate = 0.4397133138964481
    num_classes = 2

    model = ImprovedLSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, dropout_rate)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_esm_features(sequences, device="cuda"):
    """
    Returns an (N, 1280) mean-pooled ESM2 embedding array.
    """
    model_loader = getattr(esm.pretrained, ESM_MODEL_NAME)
    esm_model, alphabet = model_loader()
    esm_model = esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    repr_layer = esm_model.num_layers

    data = [("seq"+str(i), seq) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[repr_layer])
    token_reps = results["representations"][repr_layer]
    
    embeddings = token_reps[:, 1:-1].mean(dim=1).cpu().numpy()  # (N, 1280)
    return embeddings

def predict_senescence(fasta_path, device="cuda"):
    """
    1. Reads sequences from a FASTA file
    2. Extracts ESM2 embeddings
    3. Loads 'SenSeqNet_model.pth' model in the same folder
    4. Predicts senescence label (0 or 1)
    5. Returns a list of dicts
    """
    # Read sequences from FASTA
    seq_records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [rec.id for rec in seq_records]
    seq_strs = [str(rec.seq) for rec in seq_records]

    # Extract embeddings
    embeddings = extract_esm_features(seq_strs, device=device)

    # Reshape for LSTM: (N, seq_len=1, 1280)
    embeddings = embeddings.reshape(-1, 1, embeddings.shape[1])
    X_torch = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Load your pretrained LSTM-CNN from SenSeqNet_model.pth
    model = load_pretrained_model(device=device)

    # Forward pass
    with torch.no_grad():
        logits = model(X_torch)  # (N, 2)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 2)
        preds = np.argmax(probs, axis=1)  # 0 or 1

    # Format results
    results = []
    for sid, p, pr0, pr1 in zip(seq_ids, preds, probs[:, 0], probs[:, 1]):
        results.append({
            "sequence_id": sid,
            "prediction_label": int(p),
            "probability_negative": float(pr0),
            "probability_positive": float(pr1),
        })
    return results
