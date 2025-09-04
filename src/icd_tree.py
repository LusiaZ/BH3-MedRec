import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import os
from tqdm import tqdm

# Construct icd hierarchy
def build_icd_hierarchy(icd_df):
    icd_hierarchy = defaultdict(lambda: defaultdict(list))
    for icd_code in icd_df['icd_code']:
        first_layer = icd_code[:3]  # The first three nums define the first layer
        for i in range(4, len(icd_code) + 1):
            current_layer = icd_code[:i]
            icd_hierarchy[first_layer][len(current_layer)].append(current_layer)
    return icd_hierarchy

def build_combined_icd_tree(icd_df):
    return build_icd_hierarchy(icd_df)

# Intra attention
class IntraLayerAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(IntraLayerAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(1)


# Inter attention
class InterLayerAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(InterLayerAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)

    def forward(self, lower_layer, upper_layer):
        lower_layer = lower_layer.unsqueeze(1)
        upper_layer = upper_layer.unsqueeze(1)
        combined = torch.cat([lower_layer, upper_layer], dim=0)
        attn_output, _ = self.attention(combined, combined, combined)
        return attn_output.squeeze(1)

def save_icd_embeddings_with_codes(icd_codes, icd_embeddings, filepath):
    icd_embeddings_dict = {icd_codes[idx]: icd_embeddings[idx].cpu() for idx in range(min(len(icd_codes), icd_embeddings.size(0)))}
    torch.save(icd_embeddings_dict, filepath)
    print(f"file save to {filepath}")

def initialize_and_save_icd_embeddings_with_codes(icd_tree, icd_codes, filepath='pretrained_icd_embeddings_with_codes.pt'):
    model = ICDHierarchyEmbedding(icd_tree, emb_dim=64, num_heads=4)
    icd_embeddings = model.icd_embeddings.weight
    print(f"ICD Embeddings initialized with size: {icd_embeddings.size(0)}")
    save_icd_embeddings_with_codes(icd_codes, icd_embeddings, filepath)
    return model

class ICDHierarchyEmbedding(nn.Module):
    def __init__(self, icd_tree, emb_dim=64, num_heads=4):
        super(ICDHierarchyEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.icd_tree = icd_tree
        self.flattened_icd_codes = self.flatten_icd_tree(icd_tree)
        print(f"Flattened ICD codes count in model: {len(self.flattened_icd_codes)}")
        self.icd_embeddings = nn.Embedding(len(self.flattened_icd_codes), emb_dim)
        print(f"ICD Embeddings initialized with size: {self.icd_embeddings.weight.size(0)}")
        self.intra_attention = IntraLayerAttention(emb_dim, num_heads)
        self.inter_attention = InterLayerAttention(emb_dim, num_heads)

    def flatten_icd_tree(self, icd_tree):
        all_codes = []
        for first_layer, layers in icd_tree.items():
            for _, codes in layers.items():
                all_codes.extend(codes)
        unique_codes = list(set(all_codes))
        print(f"Total unique ICD codes in tree: {len(unique_codes)}")
        return unique_codes

    def forward(self):
        updated_icd_tree = defaultdict(lambda: defaultdict(list))
        for first_layer, layers in tqdm(self.icd_tree.items(), desc="Processing ICD tree"):
            for layer, codes in list(layers.items()):
                code_indices = torch.LongTensor(
                    [self.flattened_icd_codes.index(code) for code in codes if code in self.flattened_icd_codes]
                )
                embeddings = self.icd_embeddings(code_indices)
                layer_embedding = self.intra_attention(embeddings)

                if layer > 1:
                    prev_codes = self.icd_tree[first_layer][layer - 1]
                    prev_code_indices = torch.LongTensor(
                        [self.flattened_icd_codes.index(code) for code in prev_codes if code in self.flattened_icd_codes]
                    )
                    prev_layer_embedding = self.icd_embeddings(prev_code_indices)

                    layer_embedding = self.inter_attention(prev_layer_embedding, layer_embedding)

                updated_icd_tree[first_layer][layer] = layer_embedding

        self.icd_tree = updated_icd_tree

        top_layer_embeddings = [self.icd_tree[first_layer][max(self.icd_tree[first_layer].keys())] for first_layer in self.icd_tree]
        return torch.cat(top_layer_embeddings, dim=0)

# Read diagnoses and procedures files
diagnoses_file = os.path.join('..', 'data', 'input', 'd_icd_diagnoses.csv')
procedures_file = os.path.join('..', 'data', 'input', 'd_icd_procedures.csv')

# Load diagnoses and procedures data
diagnoses_df = pd.read_csv(diagnoses_file, skiprows=1, names=['icd_code', 'icd_version'])
procedures_df = pd.read_csv(procedures_file, skiprows=1, names=['icd_code', 'icd_version'])

# Construct ICD tree and combine ICD-9 and ICD-10
diagnosis_icd_tree = build_combined_icd_tree(diagnoses_df)
procedure_icd_tree = build_combined_icd_tree(procedures_df)

# Filter all ICD codes of diagnoses and procedures
diagnoses_codes = diagnoses_df['icd_code'].tolist()
procedures_codes = procedures_df['icd_code'].tolist()

initialize_and_save_icd_embeddings_with_codes(diagnosis_icd_tree, diagnoses_codes, filepath='../data/input/diagnosis_embeddings.pt')

initialize_and_save_icd_embeddings_with_codes(procedure_icd_tree, procedures_codes, filepath='../data/input/procedure_embeddings.pt')


