import numpy as np
import time

# OpenMM
import simtk.openmm as omm
import simtk.openmm.app as app
from simtk.unit import *

# ProDy
import prody as pdy


"""
Creates system as .pdb file containing appropriate configuration for DRSystem.
:param path: path to .pdb file with initial system.
:param tmp_file: is a file containing new system.
"""
def create_system(path, tmp_file="../output/tmp_system.pdb"):
    pdy_protein = pdy.parsePDB(path)
    pdy_protein = pdy_protein.select("protein")
    time0 = time.time()
    pdy.writePDB(tmp_file, pdy_protein)
    print("write PDB(prody): {0:.4f} sec".format(time.time() - time0))
    time0 = time.time()
    omm_object = app.PDBFile(tmp_file)
    print("read PDB(openmm):", time.time() - time0, "sec")
    time0 = time.time()
    modeller = app.Modeller(omm_object.getTopology(), omm_object.getPositions())
    modeller.addHydrogens()
    print("add hydrogens(openmm):", time.time() - time0, "sec")
    time0 = time.time()
    with open(tmp_file, mode='w') as inputfile:
        app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), inputfile)
    print("write PDB(openmm):", time.time() - time0, "sec")


"""
TODO: save calculated energy and force for speed up purposes.
"""
class DRSystem:
    _pdy_protein_init = None
    _omm_protein = None
    _ligand = None
    _substrate = None
    _simulation = None
    # _modes_init = None
    # _modes = None
    # _eigenvalues = None
    # _anm = None

    """
    Creates a new system for docking refinement.
    :param pdb_file: path to a file containing system from which different samples are produced.
    Chain A is a ligand to refine. Chain B is a substrate. Note that it should meet all forcefield requirements such as
    hydrogens etc.
    :param forcefiled_name: name of forcefiled which is necessary to calculate force vector and energy
    :type pdb_file: str
    :type forcefiled: str
    """
    def __init__(self, pdb_file, forcefield_name, n_modes=10):
        self._pdy_protein_init = pdy.parsePDB(pdb_file)
        self._omm_protein = app.PDBFile(pdb_file)
        self._ligand = self._pdy_protein_init.select("chain A")
        self._substrate = self._pdy_protein_init.select("chain B")

        # OpenMM System
        forcefield = app.ForceField(forcefield_name)
        integrator = omm.LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
        system = forcefield.createSystem(
            self._omm_protein.topology,
            constraints=app.HBonds,
        )
        self._simulation = app.Simulation(self._omm_protein.topology, system, integrator)
        self._simulation.context.setPositions(self._omm_protein.positions)

    """
    Returns current position.
    :returns: position of the ligand as an array of 3d coordinates.
    """
    def get_position(self):
        return self._ligand.getCoords()

    """
    Updates ligand position.
    :param new_position: new ligand position in angstrom
    """
    def set_position(self, new_position):
        self._ligand.setCoords(new_position)
        iterator = self._ligand.iterAtoms()
        i = 0
        for atom in iterator:
            self._omm_protein.positions[atom.getIndex()] = omm.Vec3(*new_position[i]) * nanometer / 10
            i += 1
        self._simulation.context.setPositions(self._omm_protein.positions)

    """
    Get ligand.
    :returns: ligand object.
    """
    def get_ligand(self):
        return self._ligand

    """
    Calculates energy of the whole system.
    :returns: energy value in kDJ/mol
    """
    def get_energy(self):
        state = self._simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    """
    Calculates force vector that acts upon ligand.
    :returns: force vector
    """
    def get_force(self):
        state = self._simulation.context.getState(getForces=True)
        return state.getForces()

    """
    Translates initial system on vector t and rotates via rotation operator r.
    :param t: translation vector t.
    :param r: rotation operator.
    """
    def set_rigid(self, t, r):
        pass

    # """
    # Selects a part of the system.
    # """
    # def select(self, selstr, **kwargs):
    #     self._ligand = self._ligand.select(selstr, **kwargs)
class NMSpaceWrapper:
    _system = None
    _modes_init = None
    _modes = None
    _eigenvalues = None
    _anm = None
    _position = None

    """
    Create new wrapper of DRSystem instance. It allows you to operate the system in NM space.
    :param drs: DRSystem instance to wrap.
    :param n_modes: number of modes. Should be less than NumberOfAtoms - 6.
    :type drs: DRSystem.
    :type n_modes: int.
    """
    def __init__(self, drs, n_modes = 10):
        self._position = np.zeros(n_modes)
        self._system = drs
        # TODO: add normal modes
        self._anm = pdy.ANM('anm')
        self._anm.buildHessian(self._system.get_ligand())
        self._anm.calcModes(n_modes=n_modes, zeros=False)
        self._modes_init = self._anm.getEigvecs()
        self._modes = self._anm.getEigvecs()
        self._eigenvalues = self._anm.getEigvals()

    """
    Returns current position in NM space.
    :returns: position of the ligand as an m-d vector from NM space.
    """
    def get_position(self):
        return self._position

    """
    Updates ligand position in NM space.
    :param new_position: new ligand position in NM space.
    """
    def set_position(self, new_position):
        real_pos = self._system.get

    """
    Calculates energy of the whole system.
    :returns: energy value in kDJ/mol.
    """
    def get_energy(self):
        return self._system.get_energy()

    """
    Calculates force vector that acts upon each mode.
    :returns: force vector in NM space.
    """
    def get_force(self):
        aw_force = self._system.get_force()
        # TODO: calculate modewise force

    """
    Translates initial system on vector t and rotates via rotation operator r.
    :param t: translation vector t.
    :param r: rotation operator.
    """
    def set_rigid(self, t, r):
        self._system.set_rigid(t, r)
        # TODO: move modes (rotate)

    """
    Get normal modes.'
    :returns: normal modes.
    """
    def get_modes(self):
        return self._modes

    """
    Get eigenvalues.
    :returns: eigenvalues.
    """
    def get_eigenvalues(self):
        return self._eigenvalues


# Unit
def Test_1():
    print("START TEST 1")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    s = 0
    for i in range(100):
        s = drs.get_energy()
    print("energy time repeat:", (-t0 + time.time()) / 100., "sec")
    t0 = time.time()
    s = 0
    for i in range(100):
        s = drs.get_force()
    print("force time:", (-t0 + time.time()) / 100., "sec")
    print("OK")
    print("END TEST 1")


def Test_2():
    print("START TEST 2")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    s = 0
    time_full = 0
    coordinates = drs.get_position()
    for i in range(100):
        offset = np.random.random(coordinates.shape) / 5
        t0 = time.time()
        drs.set_position(coordinates + offset)
        time_full += time.time() - t0
    print("set_position time:", time_full / 100., "sec")

    print("OK")
    print("END TEST 2")


def Test_3():
    print("START TEST 3")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    coordinates = drs.get_position()
    time_full = 0
    value = None
    for i in range(100):
        offset = np.random.random(coordinates.shape) / 5
        drs.set_position(coordinates + offset)
        t0 = time.time()
        value = drs.get_energy()
        time_full += time.time() - t0
    print("energy time:", time_full / 100., "sec")
    time_full = 0
    value = None
    for i in range(100):
        offset = np.random.random(coordinates.shape) / 5
        drs.set_position(coordinates + offset)
        t0 = time.time()
        value = drs.get_force()
        time_full += time.time() - t0
    print("force time:", (time_full) / 100., "sec")
    print("OK")
    print("END TEST 3")


if __name__ == "__main__":
    Test_1()