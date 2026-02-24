'''
Phase 1: WordPiece Tokenization.

Tokenizes input text by breaking words into subword units based on a predefined vocabulary (e.g. the word "learning" gets tokenized into "learn" and "##ing"). It helps the model handle rare words and maintain a manageable vocabulary size.
'''

from typing import List, Dict


class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        # split by whitespace
        for word in text.lower().split():
            # tokenize each word
            word_tokens = self._tokenize_word(word)
            # append together
            tokens.extend(word_tokens)
            
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # length check
        if len(word) > self.max_word_len:
            return [self.unk_token]
        
        subtokens = []
        start = 0
        
        # process word
        while start < len(word):
            end = len(word)
            current = None
            
            # searching for longest matching substring
            while start < end:
                sub = word[start:end]
                # non-initial subword
                if start > 0:
                    sub = "##" + sub
                # some subpart matches
                if sub in self.vocab:
                    current = sub
                    break
                end -= 1
                
            # no matching
            if current is None:
                return [self.unk_token]
            
            subtokens.append(current)
            start = end # move to next part
            
        return subtokens
                

if __name__ == "__main__":
    vocab = {"[UNK]": 0, 
             "the": 1, "cat": 2, 
             "un": 3, "##believ": 4, "##able": 5}
    
    tokenizer = WordPieceTokenizer(vocab)
    
    text1 = "the cat"
    text2 = "unbelievable"
    text3 = "xyz"
    
    print(tokenizer.tokenize(text1))  # ['the', 'cat']
    print(tokenizer.tokenize(text2))  # ['un', '##believ', '##able'
    print(tokenizer.tokenize(text3))  # ['[UNK]']