from core.refine import *
from pdbfixer import PDBFixer
import os
import traceback
import project
project.setup()
forcefield = "charmm36.xml"
# forcefield = "amber14/protein.ff14SB.xml"

if __name__ == "__main__":
    source_path = project.data_path / "benchmark" / "pp5_unbound"
    output_path = project.data_path / "benchmark" / "modeled_pp5_unbound"
    print("Searching folder:", str(source_path))
    total = len(os.listdir(str(source_path)))
    for i, pdb_file_name in enumerate(os.listdir(str(source_path))):
        try:
            pdb_file_path = str(source_path / pdb_file_name)
            pdb_output_path = str(output_path / pdb_file_name)
            print(f'iter {i + 1}/{total}: {pdb_output_path}')
            fixer = PDBFixer(filename=pdb_file_path)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(True)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            app.PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_output_path, 'w'))
        except Exception as e:
            print(f'Exception::{type(e)}: {e}')
