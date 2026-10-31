
import os
import random
import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBO_PATH = "go.obo"
TERM_PATH = "terms.tsv"

BP_PATH = (
    "mESC_BP_GO_mapped.csv"
)

PAIR_PATH = (
    "/home/featurize/work/cs-main/DeepTGI/dataset/scRNA-Seq/STRING Dataset/mESC/TFs+500/AllPair.csv"
)

HIDDEN_DIM = 128
GO_EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
N_SPLITS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. Parse GO ontology
#    Original graph direction: child -> parent
# ============================================================

def parse_go_obo(file_path):
    go_graph = nx.DiGraph()
    go_name_dict = {}
    go_def_dict = {}

    current_id = None
    current_name = ""
    current_def = ""
    parents = []
    is_obsolete = False

    def save_term():
        if current_id and not is_obsolete:
            go_graph.add_node(current_id)

            if current_name:
                go_name_dict[current_id] = current_name

            if current_def:
                go_def_dict[current_id] = current_def

            for parent in parents:
                go_graph.add_edge(current_id, parent)

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line == "[Term]":
                save_term()

                current_id = None
                current_name = ""
                current_def = ""
                parents = []
                is_obsolete = False

            elif line.startswith("["):
                save_term()

                current_id = None
                current_name = ""
                current_def = ""
                parents = []
                is_obsolete = False

            elif line.startswith("id: GO:"):
                current_id = line.split("id: ", 1)[1]

            elif line.startswith("name:") and current_id:
                current_name = line.split("name:", 1)[1].strip()

            elif line.startswith("def:") and current_id:
                current_def = line.split("def:", 1)[1].strip()

            elif line.startswith("is_obsolete: true"):
                is_obsolete = True

            elif line.startswith("is_a: GO:") and current_id:
                parent_id = line.split("is_a:", 1)[1].split()[0]
                parents.append(parent_id)

        save_term()

    # Remove edges whose parent terms were not retained
    valid_nodes = set(go_graph.nodes())
    go_graph.remove_edges_from([
        (child, parent)
        for child, parent in go_graph.edges()
        if parent not in valid_nodes
    ])

    if not nx.is_directed_acyclic_graph(go_graph):
        raise ValueError("GO ontology graph is not a DAG.")

    return go_graph, go_name_dict, go_def_dict


# ============================================================
# 3. Build GO hierarchy features
#    One multi-hot vector per GO node:
#    the node itself + all ancestors
# ============================================================

def get_go_ancestor_vector(go_graph, all_go_terms):
    go2idx = {
        go: i for i, go in enumerate(all_go_terms)
    }

    n_go = len(all_go_terms)

    features = torch.zeros(
        (n_go, n_go), dtype=torch.float32
    )

    for go in all_go_terms:
        i = go2idx[go]
        features[i, i] = 1.0

        ancestors = nx.ancestors(go_graph, go)

        for ancestor in ancestors:
            if ancestor in go2idx:
                features[i, go2idx[ancestor]] = 1.0

    return features


# ============================================================
# 4. GOTreeEncoder
#    GO hierarchy features -> GAT -> GO node embeddings
# ============================================================

class GOTreeEncoder(nn.Module):

    def __init__(
        self,
        obo_path=OBO_PATH,
        hidden_dim=HIDDEN_DIM,
        out_dim=GO_EMBED_DIM
    ):
        super().__init__()

        (
            self.go_graph,
            self.go_name_dict,
            self.go_def_dict
        ) = parse_go_obo(obo_path)

        # One consistent node ordering for features, edges and lookup
        self.all_terms = list(self.go_graph.nodes())

        self.go2idx = {
            go: i for i, go in enumerate(self.all_terms)
        }

        self.idx2go = {
            i: go for go, i in self.go2idx.items()
        }

        hierarchy_features = get_go_ancestor_vector(
            self.go_graph, self.all_terms
        )

        self.register_buffer(
            "hierarchy_features", hierarchy_features
        )

        # Reverse child -> parent into parent -> child.
        # This propagates information from ancestors to descendants.
        edges = [
            (
                self.go2idx[parent],
                self.go2idx[child]
            )
            for child, parent in self.go_graph.edges()
        ]

        if len(edges) > 0:
            edge_index = torch.tensor(
                edges, dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )

        self.register_buffer("edge_index", edge_index)

        self.struct_proj = nn.Linear(
            len(self.all_terms), hidden_dim
        )

        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            add_self_loops=True
        )

        self.gat2 = GATConv(
            hidden_dim,
            out_dim,
            heads=4,
            concat=False,
            add_self_loops=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self):
        x = self.struct_proj(self.hierarchy_features)

        x = self.gat1(x, self.edge_index)
        x = self.norm1(F.gelu(x))
        x = self.dropout(x)

        x = self.gat2(x, self.edge_index)
        x = self.norm2(x)

        # [number of GO nodes, GO embedding dimension]
        return x

    def get_go_embeddings(self):
        return self.forward()


# ============================================================
# 5. Map genes to GO node indices
#
# Assumed BP matrix:
#   rows    = GO IDs
#   columns = gene symbols
#   values  = 0/1 or nonnegative annotation values
# ============================================================

def build_gene_go_mapping(bp_path, go2idx):

    bp = pd.read_csv(bp_path, index_col=0)

    # Normalize row identifiers
    bp.index = bp.index.astype(str).str.strip()
    bp.columns = bp.columns.astype(str).str.strip()

    gene2go_idx = {}

    for gene in bp.columns:

        values = pd.to_numeric(
            bp[gene], errors="coerce"
        ).fillna(0).to_numpy()

        go_indices = []

        for go_id, value in zip(bp.index, values):

            if value > 0 and go_id in go2idx:
                go_indices.append(go2idx[go_id])

        gene2go_idx[gene] = sorted(set(go_indices))

    return gene2go_idx


# ============================================================
# 6. Dataset
# ============================================================

class TGIGODataset(Dataset):

    def __init__(self, pairs, labels, gene2go_idx):
        self.pairs = list(pairs)
        self.labels = np.asarray(labels)
        self.gene2go_idx = gene2go_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        tf, gene = self.pairs[idx]

        tf_indices = self.gene2go_idx.get(tf, [])
        gene_indices = self.gene2go_idx.get(gene, [])

        return (
            tf_indices,
            gene_indices,
            float(self.labels[idx])
        )


# ============================================================
# 7. Collate variable-length GO lists
#    -1 denotes padding
# ============================================================

def go_collate_fn(batch):

    tf_lists, gene_lists, labels = zip(*batch)

    def pad_indices(index_lists):

        max_len = max(
            1, max(len(indices) for indices in index_lists)
        )

        padded = torch.full(
            (len(index_lists), max_len),
            -1,
            dtype=torch.long
        )

        for i, indices in enumerate(index_lists):

            if len(indices) > 0:
                padded[i, :len(indices)] = torch.tensor(
                    indices, dtype=torch.long
                )

        return padded

    return (
        pad_indices(tf_lists),
        pad_indices(gene_lists),
        torch.tensor(labels, dtype=torch.float32)
    )


# ============================================================
# 8. Transformer-based TF-gene interaction predictor
#
# TF GO embeddings   -> mean pooling -> token 1
# Gene GO embeddings -> mean pooling -> token 2
# Two tokens         -> Transformer
# Transformer output -> concatenation -> classifier
# ============================================================

class GOTransformer(nn.Module):

    def __init__(
        self,
        go_encoder,
        go_dim=GO_EMBED_DIM,
        n_heads=NUM_HEADS,
        n_layers=NUM_LAYERS,
        dropout=DROPOUT
    ):
        super().__init__()

        self.go_encoder = go_encoder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=go_dim,
            nhead=n_heads,
            dim_feedforward=go_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(go_dim * 2, go_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(go_dim, 1)
        )

    @staticmethod
    def mean_pool(go_embeddings, indices):

        # indices: [batch_size, max_num_go_terms]
        valid_mask = indices >= 0

        # Avoid invalid indexing for padding
        safe_indices = indices.clamp(min=0)

        selected = go_embeddings[safe_indices]

        mask = valid_mask.unsqueeze(-1).to(
            selected.dtype
        )

        summed = (selected * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1.0)

        pooled = summed / counts

        # Genes with no GO annotations receive zero vectors
        has_go = valid_mask.any(
            dim=1, keepdim=True
        )

        pooled = pooled * has_go.to(pooled.dtype)

        return pooled

    def forward(self, tf_go_idx, gene_go_idx):

        # Compute GO embeddings once per forward pass
        go_embeddings = self.go_encoder()

        tf_embedding = self.mean_pool(
            go_embeddings, tf_go_idx
        )

        gene_embedding = self.mean_pool(
            go_embeddings, gene_go_idx
        )

        # [B, 2, D]
        pair_tokens = torch.stack(
            [tf_embedding, gene_embedding],
            dim=1
        )

        encoded = self.transformer(pair_tokens)

        # [B, 2 * D]
        pair_embedding = encoded.reshape(
            encoded.size(0), -1
        )

        # [B], logits for BCEWithLogitsLoss
        logits = self.classifier(
            pair_embedding
        ).squeeze(-1)

        return logits


# ============================================================
# 9. Train and evaluate one fold
# ============================================================

def train_one_fold(
    model,
    train_loader,
    test_loader,
    device,
    epochs=EPOCHS
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0
        total_samples = 0

        for tf_idx, gene_idx, labels in train_loader:

            tf_idx = tf_idx.to(device)
            gene_idx = gene_idx.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(tf_idx, gene_idx)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if (epoch + 1) % 10 == 0 or epoch == 0:

            avg_loss = total_loss / max(
                total_samples, 1
            )

            print(
                f"Epoch {epoch + 1:03d}/{epochs}, "
                f"Loss={avg_loss:.4f}"
            )

    # Evaluation
    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():

        for tf_idx, gene_idx, labels in test_loader:

            logits = model(
                tf_idx.to(device),
                gene_idx.to(device)
            )

            probabilities = torch.sigmoid(logits)

            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(
                probabilities.cpu().numpy().tolist()
            )

    all_labels = np.asarray(all_labels, dtype=int)
    all_scores = np.asarray(all_scores)

    auc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(
        all_labels, all_scores
    )

    # A fixed 0.5 threshold for classification metrics
    predictions = (all_scores >= 0.5).astype(int)

    acc = accuracy_score(all_labels, predictions)
    precision = precision_score(
        all_labels, predictions, zero_division=0
    )
    recall = recall_score(
        all_labels, predictions, zero_division=0
    )
    f1 = f1_score(
        all_labels, predictions, zero_division=0
    )

    return {
        "AUROC": auc,
        "AUPR": aupr,
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# 10. training and testing for five-fold cross-validation
# ============================================================

def Cross_validation():

    print("Device:", DEVICE)

    print("Loading GO hierarchy...")

    # Parse once to build the gene annotation mapping
    go_encoder = GOTreeEncoder(
        obo_path=OBO_PATH,
        hidden_dim=HIDDEN_DIM,
        out_dim=GO_EMBED_DIM
    )

    print("Number of GO nodes:", len(go_encoder.all_terms))
    print("Number of GO edges:", go_encoder.edge_index.size(1))

    gene2go_idx = build_gene_go_mapping(
        BP_PATH,
        go_encoder.go2idx
    )

    annotated_genes = sum(
        len(indices) > 0
        for indices in gene2go_idx.values()
    )

    print("Genes in GO matrix:", len(gene2go_idx))
    print("Genes with GO annotations:", annotated_genes)

    print("Loading interaction pairs...")

    pairs_df = pd.read_csv(PAIR_PATH)

    # Support either already-named columns or three unnamed columns
    if not {"tf", "gene", "label"}.issubset(
        set(pairs_df.columns)
    ):
        if pairs_df.shape[1] != 3:
            raise ValueError(
                "AllPair.csv must contain exactly three columns: "
                "tf, gene, label."
            )

        pairs_df.columns = ["tf", "gene", "label"]

    pairs_df["tf"] = pairs_df["tf"].astype(str).str.strip()
    pairs_df["gene"] = pairs_df["gene"].astype(str).str.strip()

    pairs_df["label"] = pd.to_numeric(
        pairs_df["label"], errors="raise"
    ).astype(int)

    if not set(pairs_df["label"].unique()).issubset({0, 1}):
        raise ValueError("Interaction labels must be binary 0/1.")

    pairs = list(zip(pairs_df["tf"], pairs_df["gene"]))
    labels = pairs_df["label"].to_numpy()

    # Check coverage before training
    missing_tf = sum(
        tf not in gene2go_idx for tf, gene in pairs
    )
    missing_gene = sum(
        gene not in gene2go_idx for tf, gene in pairs
    )

    print("Total interaction pairs:", len(pairs))
    print("Positive pairs:", int(labels.sum()))
    print("Pairs with TF absent from GO matrix:", missing_tf)
    print("Pairs with gene absent from GO matrix:", missing_gene)

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Both positive and negative pairs are required."
        )

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1
    ):

        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        train_dataset = TGIGODataset(
            [pairs[i] for i in train_idx],
            labels[train_idx],
            gene2go_idx
        )

        test_dataset = TGIGODataset(
            [pairs[i] for i in test_idx],
            labels[test_idx],
            gene2go_idx
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=go_collate_fn,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=go_collate_fn,
            num_workers=0
        )

        # Fresh GO encoder and predictor for each fold
        fold_encoder = GOTreeEncoder(
            obo_path=OBO_PATH,
            hidden_dim=HIDDEN_DIM,
            out_dim=GO_EMBED_DIM
        )

        model = GOTransformer(
            go_encoder=fold_encoder,
            go_dim=GO_EMBED_DIM,
            n_heads=NUM_HEADS,
            n_layers=NUM_LAYERS,
            dropout=DROPOUT
        )

        result = train_one_fold(
            model,
            train_loader,
            test_loader,
            DEVICE,
            epochs=EPOCHS
        )

        result["Fold"] = fold
        fold_results.append(result)

        print("Fold results:", result)

    # Summary
    results_df = pd.DataFrame(fold_results)

    metric_columns = [
        "AUROC", "AUPR", "ACC",
        "Precision", "Recall", "F1"
    ]

    print("\n========== Cross-validation results ==========")

    for metric in metric_columns:
        values = results_df[metric]

        print(
            f"{metric}: {values.mean():.4f} "
            f"+/- {values.std(ddof=1):.4f}"
        )

    results_df.to_csv(
        "GO_hierarchy_transformer_results.csv",
        index=False
    )

    print(
        "\nSaved results to "
        "GO_hierarchy_transformer_results.csv"
    )


#Training with all data and save GO TF-gene embeddings
def main():

    print("Device:", DEVICE)

    # ==========================================
    # 1. 初始化 GO-GAT 编码器
    # ==========================================
    print("Loading GO hierarchy...")

    go_encoder = GOTreeEncoder(
        obo_path=OBO_PATH,
        hidden_dim=HIDDEN_DIM,
        out_dim=GO_EMBED_DIM
    )

    print("Number of GO nodes:", len(go_encoder.all_terms))

    # ==========================================
    # 2. 构建 gene -> GO 映射
    # ==========================================
    gene2go_idx = build_gene_go_mapping(
        BP_PATH,
        go_encoder.go2idx
    )

    print("Genes in GO matrix:", len(gene2go_idx))

    # ==========================================
    # 3. 读取全部 TF-gene pairs
    # ==========================================
    print("Loading interaction pairs...")

    pairs_df = pd.read_csv(PAIR_PATH)

    if not {"tf", "gene", "label"}.issubset(pairs_df.columns):
        if pairs_df.shape[1] != 3:
            raise ValueError(
                "AllPair.csv must contain three columns: tf, gene, label"
            )
        pairs_df.columns = ["tf", "gene", "label"]

    pairs_df["tf"] = pairs_df["tf"].astype(str).str.strip()
    pairs_df["gene"] = pairs_df["gene"].astype(str).str.strip()
    pairs_df["label"] = pd.to_numeric(
        pairs_df["label"], errors="raise"
    ).astype(int)

    pairs = list(zip(pairs_df["tf"], pairs_df["gene"]))
    labels = pairs_df["label"].to_numpy(dtype=np.float32)

    print("Total pairs:", len(pairs))

    # ==========================================
    # 4. 全部 pair 训练一次，不进行交叉验证
    # ==========================================
    dataset = TGIGODataset(
        pairs,
        labels,
        gene2go_idx
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=go_collate_fn,
        num_workers=0
    )

    model = GOTransformer(
        go_encoder=go_encoder,
        go_dim=GO_EMBED_DIM,
        n_heads=NUM_HEADS,
        n_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    print("Training model on all pairs...")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0
        total_samples = 0

        for tf_idx, gene_idx, batch_labels in train_loader:

            tf_idx = tf_idx.to(DEVICE)
            gene_idx = gene_idx.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(tf_idx, gene_idx)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS}, "
                f"Loss={total_loss / max(total_samples, 1):.4f}"
            )

    # 保存训练后的模型
    torch.save(model.state_dict(), "GO_GAT_model.pth")

    # ==========================================
    # 5. 提取所有 pair 的 GAT embedding
    # ==========================================
    print("Extracting GAT pair embeddings...")

    model.eval()
    pair_embeddings = []

    with torch.no_grad():

        # 对完整 GO 图运行 GAT，得到所有 GO 节点的嵌入
        go_embeddings = model.go_encoder()

        # 保持 AllPair.csv 原始顺序
        extract_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=go_collate_fn,
            num_workers=0
        )

        for tf_idx, gene_idx, _ in extract_loader:

            tf_idx = tf_idx.to(DEVICE)
            gene_idx = gene_idx.to(DEVICE)

            # 分别聚合 TF 和 target gene 的 GO-GAT 嵌入
            tf_embedding = model.mean_pool(
                go_embeddings, tf_idx
            )

            gene_embedding = model.mean_pool(
                go_embeddings, gene_idx
            )

            # 拼接：[TF embedding | gene embedding]
            pair_embedding = torch.cat(
                [tf_embedding, gene_embedding],
                dim=1
            )

            pair_embeddings.append(
                pair_embedding.cpu().numpy()
            )

    pair_embeddings = np.concatenate(
        pair_embeddings, axis=0
    ).astype(np.float32)

    # 保存特征矩阵
    np.save("GO_GAT_pair_embeddings.npy", pair_embeddings)

    # 保存 pair 对应信息，确保能找到每行对应的 TF 和 gene
    pairs_df[["tf", "gene", "label"]].to_csv(
        "GO_GAT_pair_info.csv",
        index=False
    )

    print("Embedding shape:", pair_embeddings.shape)
    print("Saved: GO_GAT_pair_embeddings.npy")
    print("Saved: GO_GAT_pair_info.csv")
    print("Saved: GO_GAT_model.pth")


if __name__ == "__main__":
    main()

