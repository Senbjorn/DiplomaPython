'''
    This script helps to investigate protein-protein benchmark v5.
    Primal purpose is to find differences in number of atoms in bound and unbound conformations
'''

import pymol #pylint: disable=import-error
import pymol.cmd as cmd #pylint: disable=import-error

from pathlib import Path
import pandas as pd
from core.utils import read_benchmark_table, get_structure_paths
import project
project.setup()


pp_benchmark_dir = Path('/home/semyon/mipt/GPCR-TEAM/pp_benchmark_v5')
pp_benchmark_structures = pp_benchmark_dir / 'benchmark5' / 'structures'
# pp_benchmark_structures_out_bound = project.data_path / 'benchmark' / 'pp5_bound'
# pp_benchmark_structures_out_unbound = project.data_path / 'benchmark' / 'pp5_unbound'
pp_benchmark_table_path = pp_benchmark_dir / 'Table_BM5.xlsx'

table_bm5 = read_benchmark_table(pp_benchmark_table_path)
atom_count_df = pd.DataFrame(
    columns=[
        'Complex ID',
        'Right Unbound', 'Right Bound',
        'Left Unbound', 'Left Bound'
    ]
)
total = len(table_bm5.index)
pos = 0
for row in table_bm5.iterrows():
    idx, value = row
    paths = get_structure_paths(idx, table_bm5, pp_benchmark_structures)
    for pdb_file in paths:
        cmd.load(pdb_file)
    selection_names = list(map(lambda x: Path(x).parts[-1][:-4], paths))
    atom_counts = list(map(cmd.count_atoms, selection_names))
    atom_count_df.loc[atom_count_df.shape[0]] = [value['Complex ID']] + atom_counts
    # cmd.save(str(pp_benchmark_structures_out_unbound / f'{value["Complex ID"]}.pdb'),
    #         f'{selection_names[0]} or {selection_names[2]}')
    # cmd.save(str(pp_benchmark_structures_out_bound / f'{value["Complex ID"]}.pdb'),
    #         f'{selection_names[1]} or {selection_names[3]}')
    cmd.delete('all')
    pos += 1
    # print(f'Iteration {pos}/{total}', end='\r', flush=True)
atom_count_df['Right Balanced'] = atom_count_df['Right Unbound'] == atom_count_df['Right Bound']
atom_count_df['Left Balanced'] = atom_count_df['Left Unbound'] == atom_count_df['Left Bound']
atom_count_df.to_csv(project.data_path / 'benchmark' / 'pp5_atom_count.csv')
print(
    'Number of right balanced entries:', 
    atom_count_df['Right Balanced'].sum()
)
print(
    'Number of left balanced entries:', 
    atom_count_df['Left Balanced'].sum()
)
print(
    'Number of balanced entries:', 
    (atom_count_df['Right Balanced'] & atom_count_df['Left Balanced']).sum()
)
cmd.quit(0)
