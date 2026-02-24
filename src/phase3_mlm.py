'''
Phase 3: Masked Language Modeling (MLM).

Applies a mask to input tokens and forces the model to predict the masked tokens using surrounding context. This teaches the model to understand the relationship between words deeply.
'''

import numpy as np
from typing import Tuple


def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply BERT's MLM masking strategy.
    """
    if seed is not None:
        np.random.seed(seed)
    
    masked_input = np.copy(token_ids)
    labels = np.full(token_ids.shape, -100) # -100 to ignore
    
    # selecting tokens to mask
    prob_matrix = np.random.rand(*token_ids.shape)
    masked_idx = prob_matrix < mask_prob
    # saving the originals
    labels[masked_idx] = token_ids[masked_idx]
    
    idx_to_mask = np.where(masked_idx)
    roll = np.random.rand(len(idx_to_mask[0]))

    # 80% replace with [MASK]
    replace_mask = roll < 0.8
    masked_input[idx_to_mask[0][replace_mask], idx_to_mask[1][replace_mask]] = mask_token_id
    
    # 10% replace with random token
    random_mask = (roll >= 0.8) & (roll < 0.9)
    random_tokens = np.random.randint(0, vocab_size, size=np.sum(random_mask))
    masked_input[idx_to_mask[0][random_mask], idx_to_mask[1][random_mask]] = random_tokens
    
    # 10% keep original tokens
    
    return masked_input, labels, masked_idx
    

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token probabilities.
        """
        # linear projection: Wx + b
        logits = np.dot(hidden_states, self.W) + self.b
        
        # turn into probabilities with softmax
        e_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = e_logits / e_logits.sum(axis=-1, keepdims=True)
        
        return probs
    

if __name__ == "__main__":
    token_ids = np.array([
        [101, 2054, 2003, 2023, 102],
        [101, 100,  200,  300,  102]
    ])
    vocab_size = 30000
    mask_token_id = 103
    hidden_size = 768
    
    masked_input, labels, masked_idx = apply_mlm_mask(token_ids, vocab_size, mask_token_id, mask_prob=0.5, seed=49)
    
    hidden_states = np.random.randn(2, 5, hidden_size)
    
    mlm_head = MLMHead(hidden_size, vocab_size)
    preds = mlm_head.forward(hidden_states)
    
    print(f"Original Tokens: {token_ids}")
    print(f"Masked Input: {masked_input}")
    print(f"Labels: {labels}")
    print(f"Predictions shape: {preds.shape}")