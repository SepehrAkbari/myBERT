'''
Phase 5: BERT Pooler.

The pooler grabs the first token ([CLS]), which acts as the summary of the sequence. It passes it through a dense layer followed by a tanh activation to stabilize the representation.
'''

import numpy as np


def tanh(x):
    return np.tanh(x)

def apply_dropout(x: np.ndarray, prob: float, training: bool = True) -> np.ndarray:
    # no dropout
    if not training or prob == 0.0:
        return x
    
    # dropout mask scaled (by 1/1-p) to maintain expected value
    mask = np.random.binomial(1, 1 - prob, size=x.shape)
    return x * mask / (1 - prob)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Pool the [CLS] token representation.
        """
        # get first token ([CLS])
        cls_token = hidden_states[:, 0]
        # dense layer
        pooled_output = np.dot(cls_token, self.W) + self.b
        # tanh activation
        pooled_output = tanh(pooled_output)
        
        return pooled_output
    
class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier_W = np.random.randn(hidden_size, num_classes) * 0.02
        self.classifier_b = np.zeros(num_classes)
    
    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Classify sequences.
        """
        # pooling [CLS]
        pooled_output = self.pooler.forward(hidden_states)
        # dropout
        pooled_output = apply_dropout(pooled_output, self.dropout_prob, training)
        # classifier
        logits = np.dot(pooled_output, self.classifier_W) + self.classifier_b
        
        return logits


if __name__ == "__main__":
    batch_size = 3
    seq_length = 10
    hidden_size = 768
    num_classes = 2
    
    hidden_states = np.random.randn(batch_size, seq_length, hidden_size)
    classifier = SequenceClassifier(hidden_size, num_classes)
    logits = classifier.forward(hidden_states, training=True)
    
    print(f"Hidden States Shape: {hidden_states.shape}")
    print(f"Logits Shape: {logits.shape}")
    print(f"Logits: \n{logits}")
    