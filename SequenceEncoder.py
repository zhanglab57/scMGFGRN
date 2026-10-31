#pip install torch transformers pandas numpy tqdm
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# =========================
# 1. 参数配置
# =========================
INPUT_CSV = "mESC_gene_sequence.csv"  # 输入序列文件
OUTPUT_CSV = "gene_embeddings.csv"
OUTPUT_NPY = "gene_embeddings.npy"
MODEL_NAME = "zhihan1996/DNABERT-2-117M"

BATCH_SIZE = 8
MAX_LENGTH = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. 读取基因序列
# =========================
df = pd.read_csv(INPUT_CSV)

# 默认第一列为 DNA 序列
sequences = df.iloc[:, 0].fillna("").astype(str)
gene_names = df["gene_name"].astype(str).tolist()

# 清理序列，仅保留合法 DNA 字符
valid_bases = set("ACGTN")
sequences = sequences.str.upper().str.replace(r"\s+", "", regex=True)

# 删除空序列，并记录对应基因名
valid_mask = sequences.str.len() > 0
valid_idx  = valid_mask[valid_mask].index        # 保留原始行索引

sequences  = sequences.loc[valid_idx].reset_index(drop=True)
gene_names = df.loc[valid_idx, "gene_name"].astype(str).tolist() 

print(f"Number of genes: {len(sequences)}")

# =========================
# 3. 加载 DNABERT-2
# =========================

MODEL_NAME = "./DNABERT-2-117M"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)
model.eval()

print("Model device:", next(model.parameters()).device)

# =========================
# 4. 批量提取序列嵌入
# =========================
all_embeddings = []

with torch.no_grad():
    for start in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch_sequences = sequences.iloc[
            start:start + BATCH_SIZE
        ].tolist()

        # 替换非法字符，避免 tokenizer 遇到非标准碱基
        batch_sequences = [
            "".join(base if base in valid_bases else "N"
                    for base in seq)
            for seq in batch_sequences
        ]

        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)
        hidden_states = outputs[0]
        
        # 根据 attention_mask 对有效 token 做平均池化
        mask = inputs["attention_mask"].unsqueeze(-1).to(
            hidden_states.dtype
        )

        # 排除特殊 token（如 CLS、EOS 等）
        special_tokens_mask = torch.tensor(
            [
                tokenizer.get_special_tokens_mask(
                    ids.tolist(),
                    already_has_special_tokens=True
                )
                for ids in inputs["input_ids"]
            ],
            device=device,
            dtype=mask.dtype
        ).unsqueeze(-1)

        mask = mask * (1 - special_tokens_mask)

        embedding = (hidden_states * mask).sum(dim=1) / \
                    mask.sum(dim=1).clamp(min=1)

        all_embeddings.append(embedding.cpu().numpy())

# =========================
# 5. 保存结果
# =========================
embeddings = np.concatenate(all_embeddings, axis=0)

embedding_df = pd.DataFrame(
    embeddings,
    index=gene_names,
    columns=[f"DNABERT2_{i}" for i in range(embeddings.shape[1])]
)

embedding_df.index.name = "gene"
embedding_df.to_csv(OUTPUT_CSV)
np.save(OUTPUT_NPY, embeddings)

print("Embedding shape:", embeddings.shape)
print("Saved:", OUTPUT_CSV)
print("Saved:", OUTPUT_NPY)
