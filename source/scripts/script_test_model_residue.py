import project
from pathlib import Path
from core.model_residue import *

# setup project variables
project.setup()


def test_1():
    input_file = str(project.data_path / "6awr_edited.pdb")
    chain_b_file = str(project.output_path / "6awr_cB.pdb")
    chain_b_seq_file = str(project.output_path / "6awr_cB.pir")
    chain_b_align_file = str(project.output_path / "6awr_cB_align.pir")
    pdb_id_b = "6awr_cB"
    pdb_data = pdy.parsePDB(input_file)
    chain_data = pdb_data.select("protein and chain B")
    pdy.writePDB(chain_b_file, chain_data)
    seq = None
    for seq in SeqIO.parse(input_file, "pdb-seqres"):
        if seq.annotations['chain'] == 'B':
            break
    else:
        print("Not found!")
        return
    seq.id = pdb_id_b
    write_sequence(chain_b_seq_file, seq, seq.id, SeqTypeMod.ANY, seq.id)
    r_set = self_alignment({8, 100, 14}, chain_b_file, chain_b_seq_file, chain_b_align_file)
    fix_residues(r_set, chain_b_align_file)


if __name__ == "__main__":
    test_1()
