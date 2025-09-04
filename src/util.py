import itertools
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter, defaultdict
from rdkit import Chem
import torch
import pickle
import re
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

def flatten(lst):
    if any(isinstance(i, list) for i in lst):
        return list(itertools.chain.from_iterable(lst))
    return lst

def flatten_recursive(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_recursive(item))
        else:
            flat_list.append(item)
    return flat_list

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        processed_patient = []
        for adm in patient:
            if len(adm) >= 3:
                diag_codes, pro_codes, med_codes = adm[:3]
                diag_codes = diag_codes if isinstance(diag_codes, list) else [diag_codes]
                pro_codes = pro_codes if isinstance(pro_codes, list) else [pro_codes]
                med_codes = med_codes if isinstance(med_codes, list) else [med_codes]
                diag_codes = flatten_recursive(diag_codes)
                pro_codes = flatten_recursive(pro_codes)
                med_codes = flatten_recursive(med_codes)
                try:
                    diag_codes = [int(code) for code in diag_codes]
                    pro_codes = [int(code) for code in pro_codes]
                    med_codes = [int(code) for code in med_codes]
                except (ValueError, TypeError) as e:
                    print(f"Skipping adm due to non-integer codes: {adm}")
                    continue
                if any(isinstance(code, list) for code in diag_codes + pro_codes + med_codes):
                    print(f"Skipping adm due to nested list: {adm}")
                    continue
                if not (all(isinstance(code, int) for code in diag_codes) and
                        all(isinstance(code, int) for code in pro_codes) and
                        all(isinstance(code, int) for code in med_codes)):
                    print(f"Skipping adm due to non-integer elements after flattening: {adm}")
                    continue
                processed_patient.append([diag_codes, pro_codes, med_codes])
            else:
                print(f"Skipping adm due to insufficient length: {adm}")
                continue
        if len(processed_patient) == 0:
            processed_patient.append([[], [], []])
        return processed_patient


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def multi_label_metric(y_gt, y_pred, y_pred_prob):
    ja = jaccard_score(y_gt, y_pred, average='samples')
    prauc = average_precision_score(y_gt, y_pred_prob, average='samples')
    avg_p = np.mean(np.sum(np.logical_and(y_gt, y_pred), axis=1) / np.sum(y_pred, axis=1))
    avg_r = np.mean(np.sum(np.logical_and(y_gt, y_pred), axis=1) / np.sum(y_gt, axis=1))
    avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r)
    return ja, prauc, avg_p, avg_r, avg_f1


def ddi_rate_score(records, path):
    with open(path, 'rb') as f:
        ddi_A = pickle.load(f)
    all_cnt, dd_cnt = 0, 0
    for patient in records:
        for adm in patient:
            med_code_set = np.array(list(itertools.combinations(adm, 2)))
            if len(med_code_set) == 0:
                continue
            all_cnt += len(med_code_set)
            dd_cnt += sum(ddi_A[med_code_set[:, 0], med_code_set[:, 1]])
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def get_n_params(model):
    return sum(p.nelement() for p in model.parameters())


def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [
        x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)
    ]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score_val = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score_val)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            try:
                all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
            except ValueError:
                all_micro.append(0)
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            try:
                all_micro.append(
                    average_precision_score(y_gt[b], y_prob[b], average="macro")
                )
            except ValueError:
                all_micro.append(0)
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1_val = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def ddi_rate_score(record, path="../data/output/ddi_A_final.pkl"):
    ddi_A = dill.load(open(path, "rb"))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    for index, atc3 in med_voc.items():

        smilesList = list(molecule[atc3])
        counter = 0
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
                )
                adjacency = Chem.GetAdjacencyMatrix(mol)
                if fingerprints.shape[0] < adjacency.shape[0]:
                    fingerprints = np.pad(fingerprints, (0, adjacency.shape[0] - fingerprints.shape[0]), 'constant')
                elif fingerprints.shape[0] > adjacency.shape[0]:
                    fingerprints = fingerprints[:adjacency.shape[0]]

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except:
                continue

        average_index.append(counter)

    N_fingerprint = len(fingerprint_dict)
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        if item > 0:
            average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)


def separate_symptoms_and_diseases(icd_codes):
    symptoms = []
    diseases = []

    symptom_pattern = re.compile(r'^(78[0-9]|79[0-9]|R[0-9]{2})')

    for code in icd_codes:
        code = str(code)
        if symptom_pattern.match(code):
            symptoms.append(code)
        else:
            diseases.append(code)

    return symptoms, diseases


def pad_batch(inputs, desired_batch_size, pad_adm=[[], [], []]):
    while len(inputs) < desired_batch_size:
        inputs.append(pad_adm)
    return inputs


def custom_collate_fn(batch, desired_batch_size=10, device="cpu", drug_fingerprints_dict=None):
    if drug_fingerprints_dict is None:
        raise ValueError("drug_fingerprints_dict cannot be None")

    processed_inputs = []
    for patient in batch:
        if isinstance(patient, list):
            valid_adms = [adm for adm in patient if isinstance(adm, list) and len(adm) == 3]
            if len(valid_adms) > 0:
                processed_inputs.append(valid_adms)
            else:
                processed_inputs.append([[], [], []])
        else:
            processed_inputs.append([[], [], []])

    batch_size = len(processed_inputs)

    if batch_size < desired_batch_size:
        pad_size = desired_batch_size - batch_size
        for _ in range(pad_size):
            dummy_input = [[], [], []]
            processed_inputs.append(dummy_input)

    max_adm_len = max(len(patient) for patient in processed_inputs)
    padded_inputs = []
    smiles_fingerprints = []
    for patient in processed_inputs:
        padded_patient = patient.copy()
        patient_fingerprints = []
        while len(padded_patient) < max_adm_len:
            padded_patient.append([[], [], []])
        for adm in padded_patient:
            med_codes = adm[2]
            adm_fingerprints = [
                drug_fingerprints_dict.get(ndc, np.zeros(2048, dtype=np.float32))
                for ndc in med_codes
            ]
            patient_fingerprints.append(adm_fingerprints)
        padded_inputs.append(padded_patient)
        smiles_fingerprints.append(patient_fingerprints)

    smiles_fingerprints_tensor = torch.zeros(
        (desired_batch_size, max_adm_len, 2048), dtype=torch.float32
    ).to(device)
    for i, patient_fingerprints in enumerate(smiles_fingerprints):
        for j, adm_fingerprints in enumerate(patient_fingerprints):
            if adm_fingerprints:
                smiles_fingerprints_tensor[i, j, :] = torch.tensor(adm_fingerprints).mean(dim=0)

    return padded_inputs, smiles_fingerprints_tensor

