'''
Phase 2: Segment Embeddings.

Turns token into vectors, and adds the token (what the word is), position (where it sits), and segment (which sentence it belongs to), into one rich representation.
'''

import numpy as np


class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings.
        """
        # position ids
        batch_size, seq_length = token_ids.shape
        position_ids = np.arange(seq_length, dtype=int)
        
        # getting embeddings
        token_embeds = self.token_embeddings[token_ids] # (batch, seq, hidden)
        segment_embeds = self.segment_embeddings[segment_ids] # (batch, seq, hidden)
        position_embeds = self.position_embeddings[position_ids] # (seq, hidden) -> broadcast
        
        # BERT embedding is the sum
        return token_embeds + position_embeds + segment_embeds
        


if __name__ == "__main__":
    embedr = BertEmbeddings(vocab_size = 30522, 
                            max_position = 512, 
                            hidden_size = 768)
    
    token_ids_batch = np.array([ 
        [101, 1996, 4937, 102, 2878, 102, 653, 368, 102, 108],
        [101, 100,  100,  102, 100,  102, 100, 100, 102, 108]
    ]) # (2, 10)
    
    segment_ids_batch = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]) # (2, 10)
    
    embeds = embedr.forward(token_ids_batch, segment_ids_batch)
    print(f"Tensor shape: {embeds.shape}")  # (2, 10, 768)