import numpy as np
import re
import pandas as pd
from pathlib import Path
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


def read_benchmark_table(table_path):
    table_bm5 = pd.read_excel(table_path)
    regex_difficulty = re.compile(r'(?P<name>.+) \((?P<count>\d+)\)')
    regex_complex = re.compile(r'(?P<name>[0-9A-Z]+)_(?P<chainsR>[A-Z]+):(?P<chainsL>[A-Z]+)')
    table_bm5.columns = table_bm5.iloc[1]
    table_bm5.drop([0, 1], axis=0, inplace=True)
    table_bm5['Difficulty'] = None
    table_bm5['Complex ID'] = None
    table_bm5['Chains R'] = None
    table_bm5['Chains L'] = None
    table_bm5.reset_index(inplace=True, drop=True)
    difficulty = None
    drop_ids = []
    for row in table_bm5.iterrows():
        idx, value = row
        m_difficulty = regex_difficulty.match(value.Complex)
        m_complex = regex_complex.match(value.Complex)
        table_bm5.loc[idx, 'Difficulty'] = difficulty
        if m_difficulty:
            difficulty = m_difficulty.group('name')
            drop_ids.append(idx)
        if m_complex:
            table_bm5.loc[idx, 'Complex ID'] = m_complex.group('name')
            table_bm5.loc[idx, 'Chains R'] = m_complex.group('chainsR')
            table_bm5.loc[idx, 'Chains L'] = m_complex.group('chainsL')
    table_bm5.drop(drop_ids, axis=0, inplace=True)
    table_bm5.set_index('Complex', inplace=True, drop=True)
    return table_bm5


def get_structure_paths(idx, table, structure_path):
    structure_path = Path(structure_path)
    value = table.loc[idx]
    paths = []
    for a in ['r', 'l']:
        for b in ['u', 'b']:
            paths.append(str(structure_path / f'{value["Complex ID"]}_{a}_{b}.pdb'))
    return paths