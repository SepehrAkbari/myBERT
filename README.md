# BERT

This is an implementation of BERT (Bidirectional Encoder Representations from Transformers) based on the [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

## Overview

The model is built using NumPy exclusively, and all code is vectorized. The implementation is done in six phases:

1. [WordPiece Tokenization](./src/phase1_tokenizer.py)
2. [Segment Embeddings](./src/phase2_embeddings.py)
3. [Masked Language Modeling (MLM)](./src/phase3_mlm.py)
4. [Next Sentence Prediction (NSP)](./src/phase4_nsp.py)
5. [BERT Pooler](./src/phase5_pooler.py)
6. [Encoder](./src/phase6_encoder.py)

Every phase can be run independently. The complete BERT model is assembled in [bert.py](./src/bert.py), which showcases the pre-training and sentiment analysis tasks using [bert-tiny weights](https://huggingface.co/prajjwal1/bert-tiny) from Praj Bhargava. The vocabulary corpus used is [BERT Base Uncased](https://huggingface.co/google-bert/bert-base-uncased) from Google Research.

## Usage

To run a phase independently:

```bash
python phase<number>_<name>.py
```

To run the complete BERT model for pre-training and sentiment analysis:

```bash
python bert.py [-h] 
               --task {pretrain,sentiment} 
               --text "First text." 
               [--text2 "Second text."] 
               [--encoder {tiny,random}]
```

The `random` encoder initializes weights randomly, while the `tiny` encoder uses pre-trained weights from the bert-tiny model which can be downloaded using:

```bash
cd config
python _weights.py
```

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](./LICENSE).