import torch
import torch.nn as nn
import dgl.nn as dglnn


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, feature, drug_idx, disease_idx):
        drug_feat = self.dropout(feature["drug"][drug_idx])
        disease_feat = self.dropout(feature["disease"][disease_idx])
        pair_feat = torch.cat([drug_feat, disease_feat], dim=-1)
        score = self.mlp(pair_feat).squeeze(-1)
        return score, drug_feat, disease_feat


class NodeEmbedding(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, rel_names):
        super().__init__()
        convs = {}
        for rel in rel_names:
            conv = dglnn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True)
            nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            convs[rel] = conv

        self.embedding = dglnn.HeteroGraphConv(convs, aggregate="sum")
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_feats)
        self.prelu = nn.PReLU()

    def forward(self, graph, inputs, bn=False, dp=False):
        h = self.embedding(graph, inputs)
        out = {}
        for ntype, feat in h.items():
            if bn and dp:
                out[ntype] = self.prelu(self.dropout(self.bn_layer(feat)))
            elif dp:
                out[ntype] = self.prelu(self.dropout(feat))
            elif bn:
                out[ntype] = self.prelu(self.bn_layer(feat))
            else:
                out[ntype] = self.prelu(feat)
        return out


class SemanticAttention(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.last_beta = None

    def forward(self, z):
        weight = self.project(z).mean(0)
        beta = torch.softmax(weight, dim=0)
        self.last_beta = beta.detach()
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class SubnetworkEncoder(nn.Module):
    def __init__(self, ntypes, in_feats, out_feats, dropout):
        super().__init__()
        self.ntypes = ntypes
        self.drug_disease = NodeEmbedding(
            in_feats, out_feats, dropout, ["drug_drug", "drug_disease", "disease_disease"]
        )
        self.CPM_disease = NodeEmbedding(
            in_feats, out_feats, dropout, ["CPM_CPM", "CPM_disease", "disease_disease"]
        )
        self.CPM_CHP = NodeEmbedding(
            in_feats, out_feats, dropout, ["CPM_CPM", "CPM_CHP", "CHP_CHP"]
        )
        self.CHP_drug = NodeEmbedding(
            in_feats, out_feats, dropout, ["CHP_CHP", "CHP_drug", "drug_drug"]
        )
        self.drug_gene = NodeEmbedding(
            in_feats, out_feats, dropout, ["drug_drug", "drug_gene", "gene_gene"]
        )
        self.gene_disease = NodeEmbedding(
            in_feats, out_feats, dropout, ["gene_gene", "gene_disease", "disease_disease"]
        )

    def forward(self, g, h, bn=False, dp=False):
        new_h = {ntype: [] for ntype in self.ntypes}

        g_sub = g.edge_type_subgraph(["drug_drug", "drug_disease", "disease_disease"])
        h_sub = self.drug_disease(g_sub, {"drug": h["drug"], "disease": h["disease"]}, bn, dp)
        new_h["drug"].append(h_sub["drug"])
        new_h["disease"].append(h_sub["disease"])

        g_sub = g.edge_type_subgraph(["CPM_CPM", "CPM_disease", "disease_disease"])
        h_sub = self.CPM_disease(g_sub, {"CPM": h["CPM"], "disease": h["disease"]}, bn, dp)
        new_h["CPM"].append(h_sub["CPM"])
        new_h["disease"].append(h_sub["disease"])

        g_sub = g.edge_type_subgraph(["CPM_CPM", "CPM_CHP", "CHP_CHP"])
        h_sub = self.CPM_CHP(g_sub, {"CPM": h["CPM"], "CHP": h["CHP"]}, bn, dp)
        new_h["CPM"].append(h_sub["CPM"])
        new_h["CHP"].append(h_sub["CHP"])

        g_sub = g.edge_type_subgraph(["CHP_CHP", "CHP_drug", "drug_drug"])
        h_sub = self.CHP_drug(g_sub, {"CHP": h["CHP"], "drug": h["drug"]}, bn, dp)
        new_h["CHP"].append(h_sub["CHP"])
        new_h["drug"].append(h_sub["drug"])

        g_sub = g.edge_type_subgraph(["drug_drug", "drug_gene", "gene_gene"])
        h_sub = self.drug_gene(g_sub, {"drug": h["drug"], "gene": h["gene"]}, bn, dp)
        new_h["drug"].append(h_sub["drug"])
        new_h["gene"].append(h_sub["gene"])

        g_sub = g.edge_type_subgraph(["gene_gene", "gene_disease", "disease_disease"])
        h_sub = self.gene_disease(g_sub, {"gene": h["gene"], "disease": h["disease"]}, bn, dp)
        new_h["gene"].append(h_sub["gene"])
        new_h["disease"].append(h_sub["disease"])

        out = dict(h)
        for ntype in self.ntypes:
            if new_h[ntype]:
                out[ntype] = new_h[ntype][0]
        return out


class Model(nn.Module):
    def __init__(self, etypes, ntypes, in_feats, hidden_feats, num_heads, dropout):
        super().__init__()
        self.ntypes = ntypes
        self.num_heads = num_heads

        for ntype in ntypes:
            linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            setattr(self, f"{ntype}_linear", linear)

        self.hetero_layer1 = NodeEmbedding(hidden_feats, hidden_feats, dropout, etypes)
        self.hetero_layer2 = NodeEmbedding(hidden_feats, hidden_feats, dropout, etypes)
        self.subnet_layer = SubnetworkEncoder(ntypes, hidden_feats, hidden_feats, dropout)
        self.layer_attention_drug = SemanticAttention(hidden_feats, hidden_feats)
        self.layer_attention_disease = SemanticAttention(hidden_feats, hidden_feats)
        self.predict = MLPDecoder(hidden_feats, dropout)
        self.last_explain = {}

    def _project_inputs(self, x):
        h = {}
        for ntype in self.ntypes:
            linear = getattr(self, f"{ntype}_linear")
            h[ntype] = linear(x[ntype])
        return h

    def forward(self, g, x, mdrug, mdis, pair_drug_idx, pair_disease_idx):
        drug_emb_list = [mdrug]
        disease_emb_list = [mdis]
        drug_sources = ["m2v"]
        disease_sources = ["m2v"]

        h = self._project_inputs(x)
        drug_emb_list.append(h["drug"])
        disease_emb_list.append(h["disease"])
        drug_sources.append("input_projection")
        disease_sources.append("input_projection")

        h = self.hetero_layer1(g, h, bn=True, dp=True)
        h = self.hetero_layer2(g, h, bn=True, dp=True)
        drug_emb_list.append(h["drug"])
        disease_emb_list.append(h["disease"])
        drug_sources.append("hetero")
        disease_sources.append("hetero")

        h = self.subnet_layer(g, h, bn=False, dp=True)
        drug_emb_list.append(h["drug"])
        disease_emb_list.append(h["disease"])
        drug_sources.append("subnet")
        disease_sources.append("subnet")

        h["drug"] = self.layer_attention_drug(torch.stack(drug_emb_list, dim=1))
        h["disease"] = self.layer_attention_disease(torch.stack(disease_emb_list, dim=1))

        self.last_explain = {
            "drug_layer_beta": self.layer_attention_drug.last_beta,
            "disease_layer_beta": self.layer_attention_disease.last_beta,
            "drug_sources": drug_sources,
            "disease_sources": disease_sources,
        }
        return self.predict(h, pair_drug_idx, pair_disease_idx)
