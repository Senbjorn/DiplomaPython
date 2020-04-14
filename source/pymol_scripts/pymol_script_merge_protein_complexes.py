'''
    This script merges protein and ligand from protein-protein benchmark v5
    for bound and unbound conformations
'''

import pymol #pylint: disable=import-error
import pymol.cmd as cmd #pylint: disable=import-error

from pathlib import Path
from core.utils import read_benchmark_table, get_structure_paths, map_chains
import project
project.setup()

# create all necessary paths
pp_benchmark_dir = Path('/home/semyon/mipt/GPCR-TEAM/pp_benchmark_v5')
pp_benchmark_structures = pp_benchmark_dir / 'benchmark5' / 'structures'
pp_benchmark_structures_out_bound = project.data_path / 'benchmark' / 'pp5_bound'
pp_benchmark_structures_out_unbound = project.data_path / 'benchmark' / 'pp5_unbound'
pp_benchmark_table_path = pp_benchmark_dir / 'Table_BM5.xlsx'

# read benchmark table
table_bm5 = read_benchmark_table(pp_benchmark_table_path, drop_strange=True)

# iteration over rows and save structures
total = len(table_bm5.index)
pos = 0
counter = 0
for row in table_bm5.loc[table_bm5.index[:1000]].iterrows():
    idx, value = row
    c_1 = {x for x in value['Chains Bound 1']}
    c_2 = {x for x in value['Chains Bound 2']}
    paths = get_structure_paths(idx, table_bm5, pp_benchmark_structures)
    for pdb_file in paths:
        cmd.load(pdb_file)
    # selections <comp_name>_r_u, <comp_name>_r_b, <comp_name>_l_u, <comp_name>_l_b
    selection_names = list(map(lambda x: Path(x).parts[-1][:-4], paths))
    
    # save bound structures
    bound_names = selection_names[1::2]
    # print('\tbound_names:', bound_names)
    # select chains
    
    # cmd.save(str(pp_benchmark_structures_out_bound / f'{value["Complex ID"]}.pdb'),
    #          f'{bound_names[0]} or {bound_names[1]}')
    # save unbound structures
    unbound_names = selection_names[::2]
    # print(unbound_names)
    # rename chains
    # select chains
    # cmd.save(str(pp_benchmark_structures_out_unbound / f'{value["Complex ID"]}.pdb'),
    #          f'{selection_names[0]} or {selection_names[2]}')
    # unbound_chains = set(cmd.get_chains(f'{unbound_names[0]} or {unbound_names[1]}'))
    bound_chains_r = set(cmd.get_chains(bound_names[0]))
    bound_chains_l = set(cmd.get_chains(bound_names[1]))
    status = c_1 == bound_chains_r and c_2 == bound_chains_l
    m = map_chains(c_1, c_2, bound_chains_r, bound_chains_l)
    if not m:
        print('complex:', idx)
        print('\tchain r:', bound_chains_r)
        print('\tchain l:', bound_chains_l)
        counter += 1
    # if not status:
    #     counter += 1
    #     print('complex:', idx)
    #     print('\tchain 1:', c_1)
    #     print('\tchain 2:', c_2)
    #     print('\tchain r:', bound_chains_r)
    #     print('\tchain l:', bound_chains_l)
    #     print('\tmapping:', map_chains(c_1, c_2, bound_chains_r, bound_chains_l))
    # print('\tstatus:', status)
    # print(idx, unbound_chains, bound_chains)
    cmd.delete('all')
    pos += 1
    # print(f'Iteration {pos}/{total}', end='\r', flush=True)
print('counter:', counter)
cmd.quit(0)
