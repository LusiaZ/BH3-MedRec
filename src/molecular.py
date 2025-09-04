import torch
import torch.nn as nn
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
import numpy as np

class HyperGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(HyperGraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, hypergraph_adj):
        x = torch.mm(hypergraph_adj, x)
        x = self.linear(x)
        return x


class DrugEmbeddingPretrainer(nn.Module):
    def __init__(self, num_functional_groups, embedding_dim, num_drugs):
        super(DrugEmbeddingPretrainer, self).__init__()
        self.group_embedding = nn.Embedding(num_functional_groups, embedding_dim)
        self.hypergcn1 = HyperGraphConvolution(embedding_dim, embedding_dim)
        self.hypergcn2 = HyperGraphConvolution(embedding_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(num_drugs, 10))

    def forward(self, hypergraph_adj, group_to_index, drug_to_groups):
        drug_embeddings = {}
        all_drug_embeddings = []

        for drug, groups in drug_to_groups.items():
            group_indices = [group_to_index[group] for group in groups if group in group_to_index]
            if len(group_indices) == 0:
                continue
            group_embeddings = self.group_embedding(
                torch.LongTensor(group_indices).to(self.group_embedding.weight.device))

            aggregated_group_embedding = group_embeddings.mean(dim=0, keepdim=True)

            all_drug_embeddings.append(aggregated_group_embedding)

        drug_embeddings_matrix = torch.cat(all_drug_embeddings, dim=0)  # [num_drugs, embedding_dim]

        drug_embeddings_matrix = drug_embeddings_matrix.T  # [embedding_dim, num_drugs]

        # print(f"drug_embeddings_matrix shape: {drug_embeddings_matrix.shape}")
        # print(f"self.W.T shape: {self.W.T.shape}")

        mapped_drug_embeddings = torch.mm(drug_embeddings_matrix, self.W)  # [embedding_dim, num_nodes]
        print(f"mapped_drug_embeddings shape: {mapped_drug_embeddings.shape}")

        # hypergraph convolution
        embedding = self.hypergcn1(mapped_drug_embeddings.T, hypergraph_adj)
        embedding = torch.relu(embedding)
        embedding = self.hypergcn2(embedding, hypergraph_adj)


        for idx, (drug, _) in enumerate(drug_to_groups.items()):
            drug_index = idx
            if drug_index < embedding.shape[0]:
                drug_embeddings[drug] = embedding[drug_index, :]
            else:
                print(f"Warning: Drug {drug} has an out-of-bounds index!")

        return drug_embeddings


def detect_functional_groups_with_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return []  # Return empty list for invalid SMILES

    # Define functional groups using SMARTS patterns
    FUNCTIONAL_GROUP_PATTERNS = {
        "hydroxyl": "[OH]",
        "carboxyl": "C(=O)O",
        "amine": "[NH2]",
        "sulfhydryl": "[SH]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "ketone": "[CX3](=O)[#6]",
        "aromatic": "c1ccccc1",
        "ether": "[CX4][OX2][CX4]",
        "phenol": "c[OH]",
        "ester": "C(=O)O[C;!H]",
    }

    functional_groups = []
    for fg_name, smarts in FUNCTIONAL_GROUP_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            functional_groups.append(fg_name)

    return functional_groups


def build_hypergraph(drug_smiles_file):
    drug_smiles = pd.read_csv(drug_smiles_file)
    hyperedges = defaultdict(list)
    functional_group_map = {}

    for _, row in drug_smiles.iterrows():
        drug_name = row['ndc']
        smiles = row['moldb_smiles']
        functional_groups = detect_functional_groups_with_rdkit(smiles)
        if functional_groups:
            functional_group_map[drug_name] = functional_groups
            hyperedges[drug_name] = functional_groups

    return hyperedges, functional_group_map


def save_pretrained_drug_embeddings(pretrained_embeddings, save_path):
    torch.save(pretrained_embeddings, save_path)
    print(f"Pretrained drug embeddings saved to {save_path}")


def main():
    drug_smiles_file = '../data/molecular_info/ndc2smiles.csv'  # Replace with your file path
    embedding_dim = 64
    save_path = '../data/input/drug_embeddings.pt'
    hyperedges, functional_group_map = build_hypergraph(drug_smiles_file)
    all_functional_groups = set(func for groups in functional_group_map.values() for func in groups)
    group_to_index = {fg: i for i, fg in enumerate(all_functional_groups)}
    hypergraph_adj = torch.eye(len(all_functional_groups))
    num_drugs = len(functional_group_map)
    pretrainer = DrugEmbeddingPretrainer(len(group_to_index), embedding_dim, num_drugs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrainer.to(device)
    hypergraph_adj = hypergraph_adj.to(device)
    pretrained_embeddings = pretrainer(hypergraph_adj, group_to_index, functional_group_map)
    save_pretrained_drug_embeddings(pretrained_embeddings, save_path)


if __name__ == '__main__':
    main()
