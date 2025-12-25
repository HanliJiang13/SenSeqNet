# senseqnet/cli.py

import click
from senseqnet.inference import predict_senescence

@click.command()
@click.option("--fasta", required=True, help="Path to the FASTA file.")
@click.option("--device", default="cuda", help="Device to run on: 'cuda' or 'cpu'.")
def main(fasta, device):
    """
    Simple CLI to run senescence detection on a FASTA file.
    The model path is now fixed in senseqnet.inference (SenSeqNet_model.pth).
    """
    results = predict_senescence(fasta_path=fasta, device=device)
    click.echo("\nSenescence Prediction Results:\n")
    for r in results:
        click.echo(
            f"SeqID: {r['sequence_id']} => Label={r['prediction_label']}  "
            f"Prob[neg]={r['probability_negative']:.4f}, "
            f"Prob[pos]={r['probability_positive']:.4f}"
        )

if __name__ == "__main__":
    main()
