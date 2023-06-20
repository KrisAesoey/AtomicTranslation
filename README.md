# Rule Learning as Machine Translation using the Atomic Knowledge Bank

This is the offical repository for my master's thesis exploring the possiblities of using the Transformer architecture as a method of performing neural machine learning translation from natural language to logical formulas. The experiments are performed on datasets from the literature (marked as DKET), and the Atomic knowledge bank.

## Overview

```
project
|   README.md
|   evaluation.py
|   DketEperiments.ipynb
|   AtomicPyTorchSmallDatasets.ipynb
|   AtomicPyTorchFullDataset.ipynb
|
|___generation
|   |   atomic_generator.py
|   |   atomic_logifier.py
|   |   atomic_preprocessor.py
|   |   dket_generator.py
|   |   filehandler.py
|
|___dket_data
|   |   The original DKET datasets.
| 
|___dket_datasets
|   |   The DKET dataset in correct format for the Transformer.
|
|___atomic_data
|   |   The original Atomic data.
|
|___atomic_datasets
    |   The completed datasets made from Atomic.
```

## How to run

To create the datasets for yourself you run the `atomic_generator.py` file in your terminal/shell.
`python ./generation/atomic_generator.py`

If you wish to run the experiments, you can run the Jupyter Notebooks from end-to-end and it will perform the construction of vocabulary, data, training and evaluation. A variable is used in the notebooks that you can change to alter which dataset you want to run specifically.

## Generation

The generation folder consists of all the necessary code that converts the DKET and Atomic data into usable datasets for our Transformer models. The `dket_preprocessor.py` file replaces the indices found in the ontologies with the word they correspond to in the original sentence.

The `atomic_preprocessor.py`splits, corrects and POS-tags the data from Atomic into a format that makes it possible for the `atomic_logifier.py` file to run its algorithm that creates rules. They are combined in the `atomic_generator.py` file that performs the entire end-to-end process.

## Evalulation

The evaluation criterias are located in the `evaluation.py` file that consist of the three methods we measure accuracy of the translations. Formula accuracy (completely correct translations), Edit Distance (Levenshtein) and Token accuracy.
