import project
from enum import Enum, unique
from modeller import *
from modeller.automodel import *    # Load the automodel class
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import os
import re
import prody as pdy


@unique
class SeqTypeMod(Enum):
    RAY = "structureX"
    NMR = "structureN"
    MOD = "structureM"
    ANY = "structure"
    SEQ = "sequence"


def self_alignment(residues, pdb_file, sequence_file, alignment_file='../output/tmp/alignment.ali', postfix="_REFINED"):
    """
    Creates an alignment of sequence from .pdb file. to itself.

    @param residues: residues to model.
    @type residues: set
    @param pdb_file: path to structure.
    @type: str
    @param sequence_file: input sequence which contains protein sequence in pir format.
    @type sequence_file: str
    @param alignment_file: output file which contains alignment result.
    @type alignment_file: str
    @return: full set of residues for modelling.
    @rtype: set
    """
    # read sequence and structure
    target_sequence = SeqIO.read(sequence_file, "pir")
    structure = pdy.parsePDB(pdb_file)

    # initialize missing residue set as all residue indices
    missing_residues = set(range(1, len(target_sequence) + 1))
    # remove residues presented in the structure
    for r in structure.iterResidues():
        missing_residues.remove(r.getResnum())
    # add residues of interest
    full_residues = missing_residues.union(residues)
    # create sequence which contains template residues
    real_seq = target_sequence.seq.tomutable()
    for i in missing_residues:
        real_seq[i - 1] = '-'
    origin_sequence = SeqIO.SeqRecord(real_seq, target_sequence.id, target_sequence.name, target_sequence.description)
    # save as an alignment
    with open(alignment_file, "w") as output_file:
        write_sequence(output_file, origin_sequence,
                       origin_sequence.id, SeqTypeMod.ANY, pdb_file)
        write_sequence(output_file, target_sequence,
                       target_sequence.id + postfix, SeqTypeMod.SEQ, target_sequence.id + postfix)
    return full_residues


def write_sequence(handle, sequence, id, seq_type, path,
                   res1="FIRST", ch1="@", res2="LAST", ch2="@",
                   name="", source="", resolution="", r_factor=""):
    """
    Writes sequence in PIR format for modeller.

    @param handle: output handle.
    @type handle: handle type
    @param sequence: output sequence.
    @type sequence: Bio.SeqIO.SeqRecord
    @param id: sequence identifier. It should not contain some characters such as colons, semicolons etc.
    Use only alphanumerical characters and underscore.
    @type id: str
    @param seq_type: sequence type.
    @type seq_type: SeqTypeMod
    @param path: path to corresponding structure.
    @type path: str
    @param res1: first residue number.
    @type res1: str
    @param ch1: chain number.
    @type ch1: str
    @param res2: last residue number.
    @type res2: str
    @param ch2: chain number.
    @type ch2: str
    @param name: sequence name.
    @type name: str
    @param source: source from where the sequence came from.
    @type source: str
    @param resolution: x-ray resolution.
    @type resolution: str
    @param r_factor: r-factor.
    @type r_factor: str
    """
    out_seq = SeqIO.SeqRecord(sequence.seq, id, "", "{0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}:{8}:{9}".format(
            seq_type.value,
            path,
            res1,
            ch1,
            res2,
            ch2,
            name,
            source,
            resolution,
            r_factor
    ))
    SeqIO.write(out_seq, handle, format="pir")


def fix_residues(residues, alignment_file):
    """
    Adds missing residues file or refines residues containing missing atoms.

    @param residues: set of missing residues
    @type residues: set
    @param alignment_file: alignment which contains original sequence and reduced sequence.
    Only missing residues will be modelled. You can get alignment_file via create_mra function
    @type alignment_file: str
    """
    sequences = SeqIO.parse(alignment_file, "pir")

    knowns_id = next(sequences).id
    sequence_id = next(sequences).id
    log.verbose()
    env = environ()

    # directories for input atom files and output files
    env.io.atom_files_directory = [str(project.data_path.absolute()),
                                   str(project.output_path.absolute())]
    os.chdir(project.output_path)

    # choose residues to model
    class MyModel(automodel):
        def select_atoms(self):
            selection_list = [self.residues[str(i)] for i in residues]
            return selection(selection_list)

    a = MyModel(env, alnfile=alignment_file, knowns=knowns_id, sequence=sequence_id)
    a.starting_model = 1
    a.ending_model = 1
    a.md_level = refine.fast
    a.make()
    # back to project dir
    os.chdir(project.project_path)
