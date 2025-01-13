# SenSeqNet: A Deep Learning Framework for Cellular Senescence Detection from Protein Sequences

This repository includes source code, datasets, and models for the study:

**Hanli Jiang, Li Lin, Dongliang Deng, Jianyu Ren, Xin Yang, Lubin Liu (2024) (in submission)**

---

## Overview

**SenSeqNet** is a deep learning framework designed to detect cellular senescence from protein sequences. By leveraging advanced protein language models (e.g., ESM2) for embedding extraction and a specialized LSTM+CNN architecture, SenSeqNet aims to provide a **fast** and **accurate** way to predict whether a given protein sequence is associated with senescence.

## Features

- **ESM-based Embeddings**: Automatically extract high-quality embeddings using ESM2.
- **BiLSTM+CNN Architecture**: Combines the strengths of recurrent and convolutional layers for effective feature learning.
- **Command-Line Interface (CLI)**: Quickly run predictions on your own FASTA files using a single command.
- **PyPI Package**: Easily install the package and integrate it into your workflow.

---

## Installation

SenSeqNet is **available on PyPI**. You can install it with:

```bash
pip install senseqnet

---

## Usage

Once installed, the CLI tool senseqnet-predict is available.
Below is a simple example using CUDA:

senseqnet-predict --fasta test_sequences.fasta --device cuda

Command-Line Arguments
--fasta: Path to your input FASTA file (required).
--device: Inference device (e.g., "cpu" or "cuda"). Defaults to "cuda" if available.


Contact
For inquiries, please contact Hanli Jiang at: hhanli.jiang@mail.utoronto.ca
