from modeller import *
from modeller.automodel import *    # Load the automodel class
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


def create_mra(residues, pdb_file, alignment_file='../output/tmp/alignment.ali'):
    """
    Creates an alignment of sequence from .pdb file. and one where selected residues are missing.

    @param residues: list of missing residues
    @type residues: list
    @param pdb_file: input pdb file.
    @type pdb_file: str
    @param alignment_file: output file which contains alignment result.
    @type alignment_file: str
    """
    origin = SeqIO.parse("../data/6awr_edited.pdb", "pdb-seqres")
    print(type(origin))
    for r in origin:
        print(r.seq[27])


def model_residue(pdb_file, alignment_file='../output/tmp/alignment.ali'):
    """
    Models residues which are missing in the alignment file.

    @param pdb_file: pdb molecule to model.
    @type pdb_file: str
    @param alignment_file: alignment which contains original sequence and reduced sequence.
    Only missing residues will be modelled. You can get alignment_file via create_mra function
    @type alignment_file: str
    """
    pass

# log.verbose()
# env = environ()
#
# # directories for input atom files
# env.io.atom_files_directory = ['.', '../atom_files']
# a = loopmodel(env, alnfile='alignment.ali',
#               knowns='1qg8', sequence='1qg8_fill')
# a.starting_model = 1
# a.ending_model = 1
# a.loop.starting_model = 1
# a.loop.ending_model = 2
# a.loop.md_level = refine.fast
# a.make()
# model_residue()