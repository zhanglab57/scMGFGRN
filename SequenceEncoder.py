import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path


class SequenceEncoder(nn.Module):

    def __init__(
        self,
        output_dim=1000,
        model_name="zhihan1996/DNABERT-2-117M",
        max_length=1024,         
        freeze_backbone=True,   
        device=None
    ):
        super().__init__()

        self.output_dim = output_dim
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.dnabert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.projection = nn.Linear(768, output_dim)

        self.dnabert = self.dnabert.to(self.device)
        self.projection = self.projection.to(self.device)

        if freeze_backbone:
            for param in self.dnabert.parameters():
                param.requires_grad = False

    def forward(self, sequence):

        
        cleaned_sequence = [
            seq.upper().replace(" ", "").replace("\n", "")
            for seq in sequence
        ]

        segments = []
        for i in range(0, 2000, 1000):
            segment = seq[i:i+1000]
            segments.append(segment)
        
        
        # Tokenization
        inputs = self.tokenizer(
            segments,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # 可选冻结
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.dnabert(**inputs)
        else:
            outputs = self.dnabert(**inputs)

        hidden_states = outputs.last_hidden_state

        # Masked Mean Pooling
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).float()
        hidden_states = hidden_states * mask

        sum_embeddings = hidden_states.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)

        dna_embedding = sum_embeddings / sum_mask

        
        #enhancer_embedding = self.projection(dna_embedding)

        return enhancer_embedding
