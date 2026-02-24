'''
TinyBert Weights from Praj Bhargava.
'''

import numpy as np
import requests
import torch
import io


def get_weights(saveas = "weights.npz"):
    """
    TinyBert Weights from HuggingFace.
    """
    url = "https://huggingface.co/prajjwal1/bert-tiny/resolve/main/pytorch_model.bin"
    response = requests.get(url)
    response.raise_for_status()
    
    buffer = io.BytesIO(response.content)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    weights = {}
    
    weights["embeddings.word_embeddings"] = state_dict["bert.embeddings.word_embeddings.weight"].numpy()
    weights["embeddings.position_embeddings"] = state_dict["bert.embeddings.position_embeddings.weight"].numpy()
    weights["embeddings.token_type_embeddings"] = state_dict["bert.embeddings.token_type_embeddings.weight"].numpy()
    
    for i in range(2):
        print(f"Processing layer {i}...")
        hf_prefix = f"bert.encoder.layer.{i}."
        my_prefix = f"layer{i}."
        
        weights[f"{my_prefix}attention.query.weight"] = state_dict[f"{hf_prefix}attention.self.query.weight"].numpy()
        weights[f"{my_prefix}attention.query.bias"] = state_dict[f"{hf_prefix}attention.self.query.bias"].numpy()
        weights[f"{my_prefix}attention.key.weight"] = state_dict[f"{hf_prefix}attention.self.key.weight"].numpy()
        weights[f"{my_prefix}attention.key.bias"] = state_dict[f"{hf_prefix}attention.self.key.bias"].numpy()
        weights[f"{my_prefix}attention.value.weight"] = state_dict[f"{hf_prefix}attention.self.value.weight"].numpy()
        weights[f"{my_prefix}attention.value.bias"] = state_dict[f"{hf_prefix}attention.self.value.bias"].numpy()
        weights[f"{my_prefix}attention.output.weight"] = state_dict[f"{hf_prefix}attention.output.dense.weight"].numpy()
        weights[f"{my_prefix}attention.output.bias"] = state_dict[f"{hf_prefix}attention.output.dense.bias"].numpy()
        
        weights[f"{my_prefix}intermediate.weight"] = state_dict[f"{hf_prefix}intermediate.dense.weight"].numpy()
        weights[f"{my_prefix}intermediate.bias"] = state_dict[f"{hf_prefix}intermediate.dense.bias"].numpy()
        weights[f"{my_prefix}output.weight"] = state_dict[f"{hf_prefix}output.dense.weight"].numpy()
        weights[f"{my_prefix}output.bias"] = state_dict[f"{hf_prefix}output.dense.bias"].numpy()

    weights["pooler.dense.weight"] = state_dict["bert.pooler.dense.weight"].numpy()
    weights["pooler.dense.bias"] = state_dict["bert.pooler.dense.bias"].numpy()
    
    weights["cls.predictions.bias"] = state_dict["cls.predictions.bias"].numpy()
    weights["cls.transform.dense.weight"] = state_dict["cls.predictions.transform.dense.weight"].numpy()
    weights["cls.transform.dense.bias"] = state_dict["cls.predictions.transform.dense.bias"].numpy()
    
    np.savez(saveas, **weights)
    print(f"Weights saved as {saveas}.")

if __name__ == "__main__":
    get_weights(saveas = "TinyBert_weights.npz")