'''
BERT.
'''

import numpy as np
import argparse

from phase1_tokenizer import WordPieceTokenizer
from phase2_embeddings import BertEmbeddings
from phase3_mlm import apply_mlm_mask, MLMHead
from phase4_nsp import create_nsp_examples, NSPHead
from phase5_pooler import BertPooler
from phase6_encoder import MockBertEncoder, TinyBertEncoder 


class BERT:
    def __init__(self, vocab, weights='random'):
        self.max_len = 128
        self.num_layers = 2
        self.weights_mode = weights
        
        if weights == 'random':
            self.hidden_size = 768
            self.encoder = MockBertEncoder(self.hidden_size, self.num_layers)
        elif weights == 'tiny':
            self.hidden_size = 128
            self.encoder = TinyBertEncoder() 
        
        self.tokenizer = WordPieceTokenizer(vocab)
        self.embeddings = BertEmbeddings(len(vocab), self.max_len, self.hidden_size)
        self.pooler = BertPooler(self.hidden_size)
        self.mlm_head = MLMHead(self.hidden_size, len(vocab))
        self.nsp_head = NSPHead(self.hidden_size)
        
        self.sentiment_head = np.random.randn(self.hidden_size, 2) * 0.02 

        if weights == 'tiny':
            self._load_weights()

    def _load_weights(self):
        """
        Loads weights.
        """
        loaded = np.load("../weights/TinyBert_weights.npz")
        
        vocab_size = len(self.tokenizer.vocab)
        self.embeddings.token_embeddings = loaded["embeddings.word_embeddings"][:vocab_size]
        self.embeddings.position_embeddings = loaded["embeddings.position_embeddings"][:self.max_len]
        self.embeddings.segment_embeddings = loaded["embeddings.token_type_embeddings"]

        self.pooler.W = loaded["pooler.dense.weight"].T 
        self.pooler.b = loaded["pooler.dense.bias"]
        
        self.mlm_head.b = loaded["cls.predictions.bias"][:vocab_size]
        self.mlm_head.W = self.embeddings.token_embeddings.T

    def train(self, VOCAB, text, label_id, learning_rate=0.01):
        """
        One step of gradient descent.
        """
        tokens_a = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        token_ids = [VOCAB.get(t, VOCAB["[UNK]"]) for t in tokens]
        
        input_ids = np.array([token_ids])
        seg_ids = np.zeros_like(input_ids)
        
        embedded = self.embeddings.forward(input_ids, seg_ids)
        encoded = self.encoder.forward(embedded)
        pooled = self.pooler.forward(encoded)
        
        logits = pooled.dot(self.sentiment_head)
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        d_logits = probs.copy()
        d_logits[0, label_id] -= 1
        
        grad_W = pooled.T.dot(d_logits)
        
        self.sentiment_head -= learning_rate * grad_W
        
        return probs[0, label_id]

    def pipeline(self, VOCAB, text_a, text_b=None, task="sentiment"):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            
        token_ids = [VOCAB.get(t, VOCAB["[UNK]"]) for t in tokens]
        print(f"(i) Token IDs: {token_ids}")
        
        input_ids = np.array([token_ids])
        seg_ids = np.array([segment_ids])
        
        embedded = self.embeddings.forward(input_ids, seg_ids)
        print(f"(ii) Embeddings Shape: {embedded.shape}")
        
        encoded = self.encoder.forward(embedded)
        print(f"(iii) Encoder Output: {encoded.shape}")

        if task == "pretrain":
            self._pretraining(encoded, input_ids)
        elif task == "sentiment":
            self._sentiment(encoded)

    def _pretraining(self, encoded, input_ids):
        mlm_logits = self.mlm_head.forward(encoded)
        pred_id = np.argmax(mlm_logits, axis=-1)
        print(f"(iv) Predicted token IDs shape: {pred_id.shape}")
        
        vocab_inv = {v: k for k, v in self.tokenizer.vocab.items()}
        print("     Decoded tokens:", [vocab_inv.get(i, "[UNK]") for i in pred_id[0]])
        
        pooled = self.pooler.forward(encoded)
        nsp_probs = self.nsp_head.forward(pooled)
        print(f"(v) NSP's P(IsNext) = {nsp_probs[0][0]:.4f}")

    def _sentiment(self, encoded):
        pooled = self.pooler.forward(encoded)
        logits = pooled.dot(self.sentiment_head)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        print(f"\nSentiment Classification:")
        print(f"    Negative: {probs[0][0]:.4f}")
        print(f"    Positive: {probs[0][1]:.4f}")


if __name__ == "__main__":
    with open("../weights/vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    
    VOCAB = {word: idx for idx, word in enumerate(vocab_list)}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["pretrain", "sentiment"], required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--text2", type=str, default="jumps over the dog")
    parser.add_argument("--encoder", type=str, default="random", choices=["random", "tiny"])
    
    args = parser.parse_args()
    
    bert = BERT(vocab=VOCAB, weights=args.encoder)
    
    if args.task == "sentiment":
        train_data = [
            ("I am so happy", 1),
            ("This is awesome", 1),
            ("I love this", 1),
            ("I am very sad", 0),
            ("This is terrible", 0),
            ("I hate this", 0)
        ]
        
        print("\nTraining Classifier...")
        for epoch in range(1000):
            loss = 0
            for text, label in train_data:
                prob_correct = bert.train(VOCAB, text, label, learning_rate=0.1)
                loss += (1 - prob_correct)
            
            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}. Avg Error: {loss/len(train_data):.3f}")
        print()
        
    bert.pipeline(VOCAB, args.text, args.text2 if args.task == "pretrain" else None, task=args.task)