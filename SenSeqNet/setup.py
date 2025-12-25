# setup.py

from pathlib import Path
import setuptools

here = Path(__file__).resolve().parent
readme_path = here / "README.md"
if not readme_path.exists():
    readme_path = here.parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setuptools.setup(
    name="senseqnet",
    version="1.1.0",
    author="Hanli Jiang",
    author_email="hhanlijiang@mail.utoronto.ca",
    description="A Deep Learning Framework for Cellular Senescence Detection from Protein Sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HanliJiang13/SenSeqNet_Package",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"senseqnet": ["SenSeqNet_model.pth", "data/*.fasta"]},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "numpy",
        "pandas",
        "biopython",
        "click",
        "fair-esm",      # For ESM2 embeddings (pip install fair-seq)
    ],
    entry_points={
        "console_scripts": [
            "senseqnet-predict = senseqnet.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
