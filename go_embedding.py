import torch
import torch.nn as nn
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

# ========= 解析 OBO =========
def parse_go_obo(file_path):
    go_graph = nx.DiGraph()
    go_name_dict = {}
    go_def_dict = {}
    with open(file_path, 'r') as f:
        current_id, current_name, current_def = None, "", ""
        parents = []
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id, current_name, current_def = None, "", ""
                parents = []
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
                go_graph.add_node(current_id)
            elif line.startswith("name:") and current_id:
                current_name = line.split("name: ")[1]
                go_name_dict[current_id] = current_name
            elif line.startswith("def:") and current_id:
                current_def = line.split("def: ")[1].strip('"')
                go_def_dict[current_id] = current_def
            elif line.startswith("is_a: GO:"):
                parent_id = line.split("is_a: ")[1].split()[0]
                parents.append(parent_id)
            elif line == "" and current_id:
                for p in parents:
                    go_graph.add_edge(current_id, p)
    return go_graph, go_name_dict, go_def_dict

# ========= 加载 GO term 列表函数 =========
def load_go_id_list_from_compact_file(tsv_path, target_namespace="bp"):
    target_key = f"##go-terms {target_namespace.lower()}"
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].strip().lower().startswith(target_key):
                if i + 1 < len(lines):
                    go_ids_line = lines[i + 1].strip()
                    go_id_list = [go.strip() for go in go_ids_line.split(',') if go.strip()]
                    return go_id_list
    raise ValueError(f"GO term list for namespace '{target_namespace}' not found in {tsv_path}")

# ========= Multi-hot 构造 ==========
def get_go_ancestor_vector(go_graph, all_go_terms, embed_terms):
    go2idx = {go: i for i, go in enumerate(all_go_terms)}
    vectors = []
    for go in embed_terms:
        vec = [0] * len(all_go_terms)
        if go in go_graph:
            ancestors = nx.ancestors(go_graph, go)
            for anc in ancestors:
                if anc in go2idx:
                    vec[go2idx[anc]] = 1
            if go in go2idx:
                vec[go2idx[go]] = 1
        vectors.append(vec)
    return torch.tensor(vectors, dtype=torch.float32)

class GOEmbedProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=False)
    def forward(self, x, edge_index):
        return self.gat_conv(x, edge_index)

class GOGlobalEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, out_dim, heads=4, concat=False)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = self.norm(x)
        return x



class GOTreeEncoder(nn.Module):
    def __init__(self, obo_path="./dataset/go.obo",
                 term_path="./dataset/terms.tsv",
                 namespace="bp", device="cuda:0"):
        super().__init__()
        self.device = device
        self.namespace = namespace
        self.cached_embedding = None

        self.go_graph, self.go_name_dict, self.go_def_dict = parse_go_obo(obo_path)
        self.go_id_list = load_go_id_list_from_compact_file(term_path, namespace)
        self.all_terms = list(self.go_graph.nodes())
        self.go2idx = {go: i for i, go in enumerate(self.go_id_list)}

        self.multi_hot = get_go_ancestor_vector(self.go_graph, self.all_terms, self.go_id_list)
        self.struct_proj = GOEmbedProjector(self.multi_hot.shape[1], 128)
                     
        self.go_texts = [f"{self.go_name_dict.get(go, '')}. {self.go_def_dict.get(go, '')}" for go in self.go_id_list]
        self.bert_proj = nn.Linear(768, 128)

        sub_graph = self.go_graph.subgraph(self.go_id_list).copy()
        edge_index = [
            [self.go2idx[src], self.go2idx[tgt]]
            for src, tgt in sub_graph.edges()
            if src in self.go2idx and tgt in self.go2idx
        ]
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        self.gat1 = GATLayer(256, 128)
        self.gat2 = GOGlobalEmbedder(in_dim=128, hidden_dim=256, out_dim=512)
        self.norm = nn.LayerNorm(512)
        self.pool = global_mean_pool

        self.to(self.device)

    def forward(self):
        if self.cached_embedding is not None:
            return self.cached_embedding

        struct_feat = self.struct_proj(self.multi_hot.to(self.device))
        fused_feat = struct_feat
        x = self.gat1(fused_feat, self.edge_index.to(self.device))
        x = self.gat2(x, self.edge_index.to(self.device))
        x = self.norm(x)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        pooled = self.pool(x, batch)

        return pooled

    def get_go_embeddings(self):
        return self.forward()
