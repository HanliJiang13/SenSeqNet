# setup.py

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="senseqnet",
    version="1.0.0",
    author="Hanli Jiang",
    author_email="hhanlijiang@mail.utoronto.ca",
    description="A Deep Learning Framework for Cellular Senescence Detection from Protein Sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HanliJiang13/SenSeqNet_Package",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"senseqnet": ["senseqnet.pth"]},
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
