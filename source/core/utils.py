import numpy as np
import pandas as pd
import re
import os
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

import project
project.setup()


def get_random_direction():
    direction = np.random.uniform(-0.5, 0.5, 3)
    while np.linalg.norm(direction) == 0:
        direction = np.random.uniform(-0.5, 0.5, 3)
    return direction / np.linalg.norm(direction)


def rotation_matrix(alpha, beta, gamma):
    """
    Creates rotation matrix.
    @param alpha: rotation angle about x axis.
    @param beta: rotation angle about y axis.
    @param gamma: rotation angle about z axis.
    @return: 3x3 rotation matrix.
    @rtype: numpy.ndarray
    """
    rx = np.array([
        [1,             0,              0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)],
    ])
    ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [            0, 1,            0],
        [-np.sin(beta), 0, np.cos(beta)],
    ])
    rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [            0,              0, 1],
    ])
    return rx.dot(ry).dot(rz)


def str_to_set(s):
    '''
    Convert a string to a set of characters.
    '''
    return {c for c in s}


def str_to_list(s):
    '''
    Convert a string to a list of characters.
    '''
    return [c for c in s]


def map_chains(b_1, b_2, b_r, b_l):
    '''
    Finds an order of receptor and ligand

    @param b_1: chains from structure 1
    @param b_2: chains from structure 2
    @param b_r: chains from receptor
    @param b_l: chains from ligand

    @returns: dict with key 'r' and 'l'
    '''

    b_1 = str_to_set(b_1)
    b_2 = str_to_set(b_2)
    b_r = str_to_set(b_r)
    b_l = str_to_set(b_l)

    if b_1 <= b_r and b_2 <= b_l:
        return {'r': 1, 'l': 2}
    if b_1 <= b_l and b_2 <= b_r:
        return {'r': 2, 'l': 1}
    return None


strange_complexes = [
    '1K4C_AB:C', '2VIS_AB:C', '1FC2_C:D',
    '1QFW_HL:AB', '1OYV_B:I', '3P57_CD:P', '3P57_IJ:P', '3AAD_A:D',
    '3CPH_G:A', '1FQ1_A:B',
]


def read_benchmark_table(table_path, drop_strange=False):
    table_bm5 = pd.read_excel(table_path)
    regex_difficulty = re.compile(r'(?P<name>.+) \((?P<count>\d+)\)')
    regex_complex = re.compile(r'(?P<complex>(?P<name>[0-9A-Z]+)_(?P<chains1>[A-Z]+):(?P<chains2>[A-Z]+))')
    regex_pdb_id = re.compile(r'(?P<name>[0-9A-Z]+)_(?P<chains>[A-Z]*)')
    table_bm5.columns = table_bm5.iloc[1]
    table_bm5.drop([0, 1], axis=0, inplace=True)
    table_bm5['Difficulty'] = None
    table_bm5['Complex ID'] = None
    table_bm5['Chains Bound 1'] = None
    table_bm5['Chains Bound 2'] = None
    table_bm5['Chains Unbound 1'] = None
    table_bm5['Chains Unbound 2'] = None
    table_bm5.reset_index(inplace=True, drop=True)
    difficulty = None
    drop_ids = []
    for row in table_bm5.iterrows():
        idx, value = row
        m_difficulty = regex_difficulty.match(value.Complex)
        m_complex = regex_complex.match(value.Complex)
        m_pdb_id_1 = regex_pdb_id.match(str(value['PDB ID 1']))
        m_pdb_id_2 = regex_pdb_id.match(str(value['PDB ID 2']))
        table_bm5.loc[idx, 'Difficulty'] = difficulty
        if m_difficulty:
            difficulty = m_difficulty.group('name')
            drop_ids.append(idx)
        if m_complex:
            # remove asterisks at the end
            table_bm5.loc[idx, 'Complex'] = m_complex.group('complex')
            # add new values
            table_bm5.loc[idx, 'Complex ID'] = m_complex.group('name')
            table_bm5.loc[idx, 'Chains Bound 1'] = m_complex.group('chains1')
            table_bm5.loc[idx, 'Chains Bound 2'] = m_complex.group('chains2')
        if m_pdb_id_1:
            table_bm5.loc[idx, 'Chains Unbound 1'] = m_pdb_id_1.group('chains')
        if m_pdb_id_2:
            table_bm5.loc[idx, 'Chains Unbound 2'] = m_pdb_id_2.group('chains')
    table_bm5.drop(drop_ids, axis=0, inplace=True)
    table_bm5.set_index('Complex', inplace=True, drop=True)
    if drop_strange:
        table_bm5.drop(strange_complexes, axis=0, inplace=True)
    return table_bm5


def process_benchmark_sequence(record, structure_id):
    pattern_id = re.compile(r'.*:(?P<chain>[A-Z]*)')
    m_id = pattern_id.match(record.id)
    if not m_id:
        raise ValueError('Record ID does not match a pattern.')
    chain_id = m_id.group('chain')
    new_record = SeqIO.SeqRecord(
        record.seq,
        id=f'{structure_id}_{chain_id}',
        name=f'{structure_id}_{chain_id}',
        description=f'Chain {chain_id} from {structure_id}.',
        annotations={'Structure ID': structure_id, 'Chain ID': chain_id},
    )
    return new_record


def extract_sequences(pdb_file, structure_id=None):
    sequence_list = list(SeqIO.parse(pdb_file, "pdb-atom"))
    if not structure_id:
        structure_id = os.path.splitext(Path(pdb_file).parts[-1])[0]
    func = lambda x, s=structure_id: process_benchmark_sequence(x, s)
    sequence_list = list(map(func, sequence_list))
    return sequence_list


def get_structure_paths(idx, table, structure_path):
    structure_path = Path(structure_path)
    value = table.loc[idx]
    paths = []
    for a in ['r', 'l']:
        for b in ['u', 'b']:
            paths.append(str(structure_path / f'{value["Complex ID"]}_{a}_{b}.pdb'))
    return paths