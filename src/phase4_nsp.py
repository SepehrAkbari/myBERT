'''
Phase 4: Next Sentence Prediction (NSP).

Creates pairs of sentences and trains the model to predict if the second sentence follows the first in the original text. This helps the model grasp inter-sentence relationships.
'''

import numpy as np
from typing import List, Tuple
import random


def create_nsp_examples(
    documents: List[List[str]], 
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Create NSP training examples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    all_docs = [doc for doc in documents if len(doc) > 1]
    examples = []
    
    for _ in range(num_examples):
        # random doc and sentence/point
        doc_idx = random.randint(0, len(all_docs) - 1)
        doc = all_docs[doc_idx]
        
        # pivot point to split sentences
        split_idx = random.randint(0, len(doc) - 2) # n-1 so B is not empty
        
        sntnc_a = doc[split_idx]
        
        # IsNext or NotNext (50% chance)
        is_next = random.random() < 0.5
        if is_next:
            # pick actual next sentence
            sntnc_b = doc[split_idx + 1]
            label = 0 # IsNext
        else:
            # pick random sentence from different doc
            rand_doc_idx = random.randint(0, len(all_docs) - 1)
            while rand_doc_idx == doc_idx:
                rand_doc_idx = random.randint(0, len(all_docs) - 1)
                
            rand_doc = all_docs[rand_doc_idx]
            rand_split = random.randint(0, len(rand_doc) - 1)
            sntnc_b = rand_doc[rand_split]
            label = 1 # NotNext
            
        examples.append((sntnc_a, sntnc_b, label))
    return examples

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext probability.
        """
        # linear projection: Wx + b
        logits = np.dot(cls_hidden, self.W) + self.b
        # into probabilities
        probs = softmax(logits)
        return probs

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    documents = [
        ["The dog sat.", "It was happy.", "Then it slept."],
        ["I love ML.", "Python is great.", "AI is future."],
        ["Pizza is good.", "Cheese is melted.", "Crust is crispy."]
    ]
    batch_size = 4
    hidden_size = 768

    examples = create_nsp_examples(documents, num_examples=4, seed=49)
    
    cls_vectors = np.random.randn(batch_size, hidden_size)
    
    nsp_head = NSPHead(hidden_size)
    probs = nsp_head.forward(cls_vectors)
    
    for sntnc_a, sntnc_b, label in examples:
        label_str = "IsNext" if label == 0 else "NotNext"
        print(f"Sentence A: {sntnc_a}\nSentence B: {sntnc_b}\nLabel: {label_str}\n")
    
    print(f"Predicted Probabilities (IsNext, NotNext):\n{probs}")
    