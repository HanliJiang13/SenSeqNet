# senseqnet/inference.py

import os
import torch
import numpy as np
from Bio import SeqIO
import esm

from senseqnet.models import ImprovedLSTMClassifier

def load_pretrained_model(device="cuda"):
    """
    Initializes the ImprovedLSTMClassifier with your chosen hyperparams,
    loads the 'senseqnet.pth' checkpoint, and returns the model in eval mode.
    """

    checkpoint_path = os.path.join(os.path.dirname(__file__), "senseqnet.pth")

    input_dim = 480         # for ESM2_t12_35M
    hidden_dim = 181
    num_layers = 4
    dropout_rate = 0.4397
    num_classes = 2

    model = ImprovedLSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, dropout_rate)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_esm_features(sequences, device="cuda"):
    """
    Returns an (N, 480) mean-pooled ESM2 embedding array.
    """
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()

    data = [("seq"+str(i), seq) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[12])
    token_reps = results["representations"][12]
    
    embeddings = token_reps.mean(dim=1).cpu().numpy()  # (N, 480)
    return embeddings

def predict_senescence(fasta_path, device="cuda"):
    """
    1. Reads sequences from a FASTA file
    2. Extracts ESM2 embeddings
    3. Loads 'senseqnet.pth' model in the same folder
    4. Predicts senescence label (0 or 1)
    5. Returns a list of dicts
    """
    # Read sequences from FASTA
    seq_records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [rec.id for rec in seq_records]
    seq_strs = [str(rec.seq) for rec in seq_records]

    # Extract embeddings
    embeddings = extract_esm_features(seq_strs, device=device)

    # Reshape for LSTM: (N, seq_len=1, 768)
    embeddings = embeddings.reshape(-1, 1, embeddings.shape[1])
    X_torch = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Load your pretrained LSTM-CNN from senseqnet.pth
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
