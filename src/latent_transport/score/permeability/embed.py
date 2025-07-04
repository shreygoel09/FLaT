import torch
import numpy as np
from typing import List
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs


rdBase.DisableLog('rdApp.error')

def fingerprints_from_smiles(smiles: List, size=2048):
    fps = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)
    return np.concatenate(fps, axis=0)


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


def getMolDescriptors(mol, missingVal=0):
    values = []
    for _, fn in Descriptors._descList:
        try: values.append(fn(mol))
        except: values.append(missingVal)
    for fn in [
        rdMolDescriptors.CalcNumLipinskiHBD,
        rdMolDescriptors.CalcNumLipinskiHBA,
        rdMolDescriptors.CalcNumRotatableBonds
    ]:
        try: values.append(fn(mol))
        except: values.append(missingVal)
    return np.array(values)


def get_pep_dps(smi_list: List[str]):
    dps = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        descs = getMolDescriptors(mol) if mol else np.zeros((213,))
        dps.append(descs)
    return np.array(dps)

@torch.no_grad()
def embed_smiles(batch, model) -> np.ndarray:
    output = model(batch['input_ids'], batch['attention_mask'])
    embedding = output.last_hidden_state
    return embedding