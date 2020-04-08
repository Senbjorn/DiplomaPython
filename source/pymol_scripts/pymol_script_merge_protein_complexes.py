'''
    This script merges protein and ligand from protein-protein benchmark v5
    for bound and unbound conformations
'''

import pymol #pylint: disable=import-error
import pymol.cmd as cmd #pylint: disable=import-error

from pathlib import Path
from utils import read_benchmark_table, get_structure_paths
import project
project.setup()


pp_benchmark_dir = Path('/home/semyon/mipt/GPCR-TEAM/pp_benchmark_v5')
pp_benchmark_structures = pp_benchmark_dir / 'benchmark5' / 'structures'
pp_benchmark_structures_out_bound = project.data_path / 'benchmark' / 'pp5_bound'
pp_benchmark_structures_out_unbound = project.data_path / 'benchmark' / 'pp5_unbound'
pp_benchmark_table_path = pp_benchmark_dir / 'Table_BM5.xlsx'

table_bm5 = read_benchmark_table(pp_benchmark_table_path)
total = len(table_bm5.index)
pos = 0
for row in table_bm5.iterrows():
    idx, value = row
    paths = get_structure_paths(idx, table_bm5, pp_benchmark_structures)
    for pdb_file in paths:
        cmd.load(pdb_file)
    selection_names = list(map(lambda x: Path(x).parts[-1][:-4], paths))
    cmd.save(str(pp_benchmark_structures_out_unbound / f'{value["Complex ID"]}.pdb'),
             f'{selection_names[0]} or {selection_names[2]}')
    cmd.save(str(pp_benchmark_structures_out_bound / f'{value["Complex ID"]}.pdb'),
             f'{selection_names[1]} or {selection_names[3]}')
    cmd.delete('all')
    pos += 1
    print(f'Iteration {pos}/{total}', end='\r', flush=True)
cmd.quit(0)
