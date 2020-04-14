'''
We try to investigate whether protein-protein benchmark is read correctly.

The script does the following:
1. Reads all files from the protein-protein benchmark v5.
2. Matches bound and unbound structures.
3. Performes pairwise alignment of all chains from the matched structures.
4. Outputs complex name, number of residues in each chain, tables with alignment socres.
   Also saves all the alignments as pickle dump of a dict.
'''

import project
project.setup()
from pathlib import Path
from core.utils import read_benchmark_table, get_structure_paths, map_chains
import prody as pdy
import numpy as np


pp_benchmark_dir = Path('/home/semyon/mipt/GPCR-TEAM/pp_benchmark_v5')
pp_benchmark_structures = pp_benchmark_dir / 'benchmark5' / 'structures'
# pp_benchmark_structures_out_bound = project.data_path / 'benchmark' / 'pp5_bound'
# pp_benchmark_structures_out_unbound = project.data_path / 'benchmark' / 'pp5_unbound'
pp_benchmark_table_path = pp_benchmark_dir / 'Table_BM5.xlsx'


if __name__ == '__main__':
    benchmark_table = read_benchmark_table(pp_benchmark_table_path, drop_strange=True)
    counter = 0
    for idx, row in benchmark_table.loc[benchmark_table.index[counter:1000]].iterrows():
        counter += 1
        print('#' * 20, 'COMPLEX:', idx, '#' * 20)
        print('ITERATION:', counter)
        table_chains_12 = {
            1: {'b': row['Chains Bound 1'], 'u': row['Chains Unbound 1']},
            2: {'b': row['Chains Bound 2'], 'u': row['Chains Unbound 2']},
        }
        path_r_u, path_r_b, path_l_u, path_l_b =\
            get_structure_paths(idx, benchmark_table, pp_benchmark_structures)
        structures_rl = {
            'r': {'b': pdy.parsePDB(path_r_b), 'u': pdy.parsePDB(path_r_u)},
            'l': {'b': pdy.parsePDB(path_l_b), 'u': pdy.parsePDB(path_l_u)},
        }
        structure_chains_rl = {
            k1: {
                k2: ''.join(np.unique(v2.getChids()))
                for k2, v2 in v1.items()
            }
            for k1, v1 in structures_rl.items()
        }
        mapping = map_chains(
            table_chains_12[1]['b'],
            table_chains_12[2]['b'],
            structure_chains_rl['r']['b'],
            structure_chains_rl['l']['b'],
        )
        structures_12 = {
            mapping[k1]: {
                k2: v2
                for k2, v2 in v1.items()
            }
            for k1, v1 in structures_rl.items()
        }
        structure_chains_12 = {
            mapping[k1]: {
                k2: v2
                for k2, v2 in v1.items()
            }
            for k1, v1 in structure_chains_rl.items()
        }
        # check that chains match for unbound
        check_mapping_12 = map_chains(
            table_chains_12[1]['u'],
            table_chains_12[2]['u'],
            structure_chains_12[1]['u'],
            structure_chains_12[2]['u'],
        )
        if not check_mapping_12 or check_mapping_12['r'] != 1:
            print('Can not map chains for unbound structures')
            print(
                table_chains_12[1]['u'],
                table_chains_12[2]['u'],
                structure_chains_12[1]['u'],
                structure_chains_12[2]['u'],
            )
            assert check_mapping_12 and check_mapping_12['r'] == 1
        
        # TODO alignment and output
