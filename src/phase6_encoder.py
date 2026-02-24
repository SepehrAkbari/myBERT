'''
Phase 6: Fine-Tuning Architecture.

Adapts the pre-trained model for specific downstream tasks such as text classification or named entity recognition. It includes task-specific heads and allows for freezing certain layers of the BERT model during training to prevent overfitting and retain learned representations.
'''

import numpy as np
from typing import List


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TinyBertLayer:
    """A single layer of BERT using loaded weights."""
    def __init__(self, weights, layer_idx=0):
        prefix = f"layer{layer_idx}."
        
        # Attention Weights
        self.W_q = weights[prefix + "attention.query.weight"].T
        self.b_q = weights[prefix + "attention.query.bias"]
        
        self.W_k = weights[prefix + "attention.key.weight"].T
        self.b_k = weights[prefix + "attention.key.bias"]
        
        self.W_v = weights[prefix + "attention.value.weight"].T
        self.b_v = weights[prefix + "attention.value.bias"]
        
        # Attention Output
        self.W_o = weights[prefix + "attention.output.weight"].T
        self.b_o = weights[prefix + "attention.output.bias"]
        
        # Feed Forward Network
        self.W_int = weights[prefix + "intermediate.weight"].T
        self.b_int = weights[prefix + "intermediate.bias"]
        
        self.W_out = weights[prefix + "output.weight"].T
        self.b_out = weights[prefix + "output.bias"]
        
        self.head_dim = 128 // 2 # 2 heads, hidden size 128
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        
        # self-attention (single head)
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v
        
        # scaled dot-product attention
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(Q.shape[-1])
        attn_probs = softmax(scores)
        attn_output = np.matmul(attn_probs, V)
        
        # projection
        attn_output = attn_output @ self.W_o + self.b_o
        x = x + attn_output
        
        # feed forward
        intermediate = np.maximum(0, x @ self.W_int + self.b_int) 
        output = intermediate @ self.W_out + self.b_out
        
        return x + output
    
class TinyBertEncoder:
    """Uses real loaded weights."""
    def __init__(self, weights_path="../weights/TinyBert_weights.npz"):
        self.weights = np.load(weights_path)
        
        self.layers = [
            TinyBertLayer(self.weights, layer_idx=0),
            TinyBertLayer(self.weights, layer_idx=1)
        ]
        
    def forward(self, embeddings):
        x = embeddings
        for _, layer in enumerate(self.layers):
            x = layer.forward(x)
        return x
    
class MockBertEncoder:
    """Simulated BERT encoder with 12 layers."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Each layer just adds a small transformation
        self.layers = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers
    
    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers (no gradient updates)."""
        # setting layers to True to freeze
        for idx in layer_indices:
            self.layer_frozen[idx] = True
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        # setting all to False to unfreeze
        self.layer_frozen = [False] * self.num_layers
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        x = embeddings
        for i, layer in enumerate(self.layers):
            if not self.layer_frozen[i]:
                x = x @ layer + x  # Simplified residual
            else:
                # Frozen: still compute but mark as no-grad
                x = x @ layer + x
        return x

class BertForSequenceClassification:
    """BERT with classification head."""
    
    def __init__(self, hidden_size: int, num_labels: int, freeze_bert: bool = False):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.freeze_bert = freeze_bert
        self.bias = np.zeros(num_labels)
        
        if freeze_bert:
            self.encoder.freeze_layers(list(range(12)))
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for classification.`
        """
        # passing through BERT encoder
        encoded = self.encoder.forward(embeddings)
        # extracting first token
        cls_token = encoded[:, 0, :]
        # projection
        logits = cls_token @ self.classifier + self.bias
        
        return logits

class BertForTokenClassification:
    """BERT with token-level classification (NER, POS tagging)."""
    
    def __init__(self, hidden_size: int, num_labels: int):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.bias = np.zeros(num_labels)
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for token classification.
        """
        # passing through BERT encoder
        encoded = self.encoder.forward(embeddings)
        # projecting each token
        logits = encoded @ self.classifier + self.bias
        
        return logits
    
    
if __name__ == "__main__":
    batch_size = 2
    seq_length = 5
    hidden_size = 768
    
    embeddings = np.random.randn(batch_size, seq_length, hidden_size)
    
    # Sequence Classification
    num_classes = 2
    seq_model = BertForSequenceClassification(hidden_size, num_classes, freeze_bert=True)
    seq_logits = seq_model.forward(embeddings)
    
    print("Sequence Classification")
    print(f"Input shape: {embeddings.shape}")
    print(f"Logits shape: {seq_logits.shape}\n") 
    
    # Token Classification
    num_tags = 9
    token_model = BertForTokenClassification(hidden_size, num_tags)
    token_logits = token_model.forward(embeddings)
    
    print("Token Classification")
    print(f"Input shape: {embeddings.shape}")
    print(f"Logits shape: {token_logits.shape}")