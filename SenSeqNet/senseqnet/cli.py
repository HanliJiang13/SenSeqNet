# senseqnet/cli.py

import click
from senseqnet.inference import predict_senescence

@click.command()
@click.option("--fasta", required=True, help="Path to the FASTA file.")
@click.option("--device", default="cuda", help="Device to run on: 'cuda' or 'cpu'.")
@click.option(
    "--max-sequences-per-batch",
    default=8,
    show_default=True,
    type=int,
    help="Max sequences per ESM2 batch.",
)
@click.option(
    "--max-tokens-per-batch",
    default=2000,
    show_default=True,
    type=int,
    help="Approximate token budget per ESM2 batch (includes BOS/EOS).",
)
@click.option(
    "--max-residues-per-chunk",
    default=1000,
    show_default=True,
    type=int,
    help="Split long sequences into chunks of this many residues.",
)
def main(
    fasta,
    device,
    max_sequences_per_batch,
    max_tokens_per_batch,
    max_residues_per_chunk,
):
    """
    Simple CLI to run senescence detection on a FASTA file.
    The model path is now fixed in senseqnet.inference (SenSeqNet_model.pth).
    """
    results = predict_senescence(
        fasta_path=fasta,
        device=device,
        max_sequences_per_batch=max_sequences_per_batch,
        max_tokens_per_batch=max_tokens_per_batch,
        max_residues_per_chunk=max_residues_per_chunk,
    )
    click.echo("\nSenescence Prediction Results:\n")
    for r in results:
        click.echo(
            f"SeqID: {r['sequence_id']} => Label={r['prediction_label']}  "
            f"Prob[neg]={r['probability_negative']:.4f}, "
            f"Prob[pos]={r['probability_positive']:.4f}"
        )

if __name__ == "__main__":
    main()
