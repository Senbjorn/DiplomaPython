from refine import *
from pdbfixer import PDBFixer
import os
import traceback
import project
project.setup()
forcefield = "charmm36.xml"
# forcefield = "amber14/protein.ff14SB.xml"

if __name__ == "__main__":
    originA_path = project.data_path / "benchmark" / "originA"
    print("Searching folder:", str(originA_path))
    for pdb_file_name in os.listdir(str(originA_path)):
        pdb_file_path = str(originA_path / pdb_file_name)
        pdb_output_path = str(project.data_path / "benchmark" / "modeled" / (pdb_file_name[:4] + "_modeled.pdb"))
        fixer = PDBFixer(filename=pdb_file_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        app.PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_output_path, 'w'))