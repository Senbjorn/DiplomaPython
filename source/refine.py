import numpy as np
import time

# OpenMM
import simtk.openmm as omm
import simtk.openmm.app as app
from simtk.unit import *

# ProDy
import prody as pdy


def rotation_matrix(alpha, beta, gamma):
    """
    Creates rotation matrix.
    @param alpha: rotation angle about x axis.
    @param beta: rotation angle about y axis.
    @param gamma: rotation angle about z axis.
    @return: 3x3 rotation matrix.
    @rtype: numpy.ndarray
    """
    rx = np.array([
        [1,             0,              0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)],
    ])
    ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [            0, 1,            0],
        [-np.sin(beta), 0, np.cos(beta)],
    ])
    rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [            0,              0, 1],
    ])
    return rx.dot(ry).dot(rz)


def create_system(path, tmp_file="../output/tmp_system.pdb", forcefield_name="charmm36.xml"):
    """
    Creates system as .pdb file containing appropriate configuration for DRSystem.
    @param path:  path to .pdb file with initial system.
    @type path: str
    @param tmp_file: a file containing new system.
    @type tmp_file: str
    """
    pdy_protein = pdy.parsePDB(path)
    pdy_protein = pdy_protein.select("protein")
    time0 = time.time()
    pdy.writePDB(tmp_file, pdy_protein)
    print("write PDB(prody): {0:.4f} sec".format(time.time() - time0))
    time0 = time.time()
    omm_object = app.PDBFile(tmp_file)
    print("read PDB(openmm):", time.time() - time0, "sec")
    time0 = time.time()
    forcefield = app.ForceField(forcefield_name)
    modeller = app.Modeller(omm_object.getTopology(), omm_object.getPositions())
    modeller.addHydrogens(forcefield=forcefield)
    print("add hydrogens and extra particles(openmm):", time.time() - time0, "sec")
    time0 = time.time()
    with open(tmp_file, mode='w') as inputfile:
        app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), inputfile)
    print("write PDB(openmm):", time.time() - time0, "sec")


# TODO: save calculated energy and force for speed up purposes.
class DRSystem:
    # PDY
    _pdy_protein_init = None
    _refine_prot_init = None
    _refine_prot = None
    _static_prot_init = None
    # OMM
    _omm_protein = None
    _simulation = None
    _center = None

    def __init__(self, pdb_file, forcefield_name, refine="chain A", static="chain B"):
        """
        Creates a new system for refinement.
        @param pdb_file: path to a file containing system from which different samples are produced.
        Chain A is a protein to refine. Chain B is a substrate. Note that it should meet
        all forcefield requirements such as hydrogens etc.
        @type pdb_file: str
        @param forcefield_name: name of forcefiled which is necessary to calculate force vector and energy.
        @type forcefield_name: str
        @param refine: atom selection to refine.
        @type refine: str
        @param static: atom selection which is to be static. Should be disjoint with refine selection.
        @type static: str
        """

        self._pdy_protein_init = pdy.parsePDB(pdb_file)
        self._omm_protein = app.PDBFile(pdb_file)
        print(len(self._pdy_protein_init.select(refine)), len(self._pdy_protein_init.select(static)))
        self._refine_prot = self._pdy_protein_init.select(refine).copy()
        self._refine_prot_init = self._pdy_protein_init.select(refine)
        self._static_prot_init = self._pdy_protein_init.select(static)
        self._center = pdy.calcCenter(self._refine_prot, weights=self._refine_prot.getMasses())
        self._center_init = pdy.calcCenter(self._refine_prot_init, weights=self._refine_prot_init.getMasses())

        # OpenMM System
        forcefield = app.ForceField(forcefield_name)
        # [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(self._omm_protein.topology)

        # Set the atom types
        # for template in templates:
            # for atom in template.atoms:
            #     atom.type = ...  # set the atom types here
            # Register the template with the forcefield.
            # forcefield.registerResidueTemplate(template)

        integrator = omm.LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
        system = forcefield.createSystem(
            self._omm_protein.topology,
            constraints=app.HBonds,
        )
        self._simulation = app.Simulation(self._omm_protein.topology, system, integrator)
        self._simulation.context.setPositions(self._omm_protein.positions)

    def get_init_position(self):
        """
        Returns initial position.
        @return: initial position of the ligand as an array of 3d coordinates in angstrom.
        @rtype: numpy.ndarray
        """
        return self._refine_prot_init.getCoords()

    def get_position(self):
        """
        Returns current position.
        @return: position of the ligand as an array of 3d coordinates in angstrom.
        @rtype: numpy.ndarray
        """
        return self._refine_prot.getCoords()

    def set_position(self, new_position):
        """
        Updates ligand position.
        @param new_position: new ligand position in angstrom.
        @type new_position: numpy.ndarray
        """
        self._refine_prot.setCoords(new_position)
        iterator = self._refine_prot.iterAtoms()
        i = 0
        for atom in iterator:
            self._omm_protein.positions[atom.getIndex()] = omm.Vec3(*new_position[i]) * nanometer / 10
            i += 1
        self._simulation.context.setPositions(self._omm_protein.positions)

    def get_refine_prot(self):
        """
        Get ligand.
        @return: ligand object.
        @rtype: ProDy.AtomGroup
        """
        return self._refine_prot

    def get_energy(self):
        """
        Calculates energy of the whole system.
        @return: energy value in kDJ/mol.
        @rtype: float
        """
        state = self._simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    def get_force(self):
        """
        Calculates force vector that acts upon a protein to be refined.
        @return: force vector in kJ/mole/nm.
        @rtype: numpy.ndarray
        """
        state = self._simulation.context.getState(getForces=True)
        all_forces = np.array(state.getForces().value_in_unit(kilojoule_per_mole / nanometer))
        iterator = self._refine_prot.iterAtoms()
        i = 0
        forces = []
        for atom in iterator:
            forces.append(all_forces[atom.getIndex()])
            i += 1
        return np.array(forces)

    def set_rigid(self, t, r):
        """
        Translates initial system by vector t and rotates via rotation operator r.
        @param t: translation vector t.
        @type t: np.ndarray
        @param r: rotation operator.
        @type r: np.ndarray
        """
        pos = self.get_init_position().copy()
        self._center = self._center_init + t
        for i in range(len(pos)):
            pos[i] += t
            pos[i] = np.dot(r, pos[i] - self._center) + self._center
        self.set_position(pos)
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

    def __init__(self, drs, n_modes=10):
        """
        Create new wrapper of DRSystem instance. It allows you to operate the system in NM space.
        @param drs: DRSystem instance to wrap.
        @type drs: DRSystem
        @param n_modes: number of modes. Should be less than 3 * NumberOfAtoms - 6.
        @type n_modes: int
        """
        self._position = np.zeros(n_modes)
        self._system = drs
        self._refine_prot_init = self._system._refine_prot.copy()
        self._anm = pdy.ANM('anm')
        self._anm.buildHessian(self._system.get_refine_prot())
        self._anm.calcModes(n_modes=n_modes, zeros=False)
        self._modes_init = self._anm.getEigvecs().copy().T
        self._modes = self._anm.getEigvecs().copy().T
        self._eigenvalues = self._anm.getEigvals()

    def get_position(self):
        """
        Returns current position in NM space.
        @return: position of the protein as an m-d vector from NM space.
        @rtype: numpy.ndarray
        """
        return self._position

    def set_position(self, new_position):
        """
        Updates protein position in NM space.
        @param new_position: new protein position in NM space.
        @type new_position: numpy.ndarray
        """
        self._position = new_position
        old_atom_position = self._refine_prot_init.getCoords()
        atom_position = old_atom_position +\
                        np.dot(self._modes.T, self._position).reshape((len(old_atom_position), 3))
        self._system.set_position(atom_position)

    def get_system_position(self):
        """
        Returns current coordinates
        @return: coordinates of protein atoms as an array of 3d coordinates.
        @rtype: numpy.ndarray
        """
        return self._system.get_position()

    def get_energy(self):
        """
        Calculates energy of the whole system.
        @return: energy value in kDJ/mol.
        @rtype: float
        """
        return self._system.get_energy()

    def get_force(self):
        """
        Calculates force vector that acts upon each mode.
        @return: force vector in NM space.
        @rtype: numpy.ndarray
        """
        aw_force = self._system.get_force()
        force_1d = np.reshape(aw_force, (aw_force.shape[0] * 3,))
        return np.dot(self._modes, force_1d)

    def set_rigid(self, t, r):
        """
        Translates initial system by vector t and rotates via rotation operator r.
        @param t: translation vector t.
        @type t: np.ndarray
        @param r: rotation operator.
        @type r: np.ndarray
        """
        self._system.set_rigid(t, r)
        m = self._modes_init.shape[0]
        n = self._modes_init.shape[1]
        tmp_modes = np.reshape(self._modes_init, (m, n // 3, 3)).copy()
        for i in range(m):
            for j in range(n // 3):
                tmp_modes[i][j] = np.dot(r, tmp_modes[i][j])
        self._modes = np.reshape(tmp_modes, (m, n))

    def get_modes(self):
        """
        Get normal modes.'
        @return: normal modes (shape = (m, 3N)).
        @rtype: numpy.ndarray
        """
        return self._modes

    def get_eigenvalues(self):
        """
        Get eigenvalues.
        @return: eigenvalues.
        @rtype: numpy.ndarray
        """
        return self._eigenvalues


def init_rapid_rmsd(nmw):
    pass


def save_trajectory(states, output_path):
    pass


def confined_gradient_descent(
        nmw, fold_parameter=0.9, termination="growth",
        absolute_bound=float("inf"), relative_bound=7.0, etol=1, n_samples=10, max_iter=100, return_traj=False):
    """
    Performs gradient descent of a system with respect to a special confinement.

    @param nmw: system to optimize.
    @type nmw: NMSpaceWrapper
    @param fold_parameter: fold step when choosing optimal step.
    @type fold_parameter: float
    @param termination: termination condition.
    @type termination: str
    @param absolute_bound: maximum rmsd between inital state and any intermediate state.
    @param relative_bound: maximum rmsd between actual state and the next.
    @param etol: terminates when |E(i+1) - E(i)| < etol
    @type etol: float
    @param n_samples: maximum number of iterations while optimizing energy along a gradient.
    @type n_samples: int
    @param max_iter: maximum number of iterations
    @type max_iter: int
    @param return_traj: if true all intermediate states and energies are returned. Otherwise, only final state and energy
    @type return_traj: bool
    @return: dictionary containing all the results.
        "states" - list of all states obtained.
        "values" - list of all energies obtained.
    @rtype: dict
    """

    # number of modes
    m = len(nmw.get_eigenvalues())

    # number of atoms
    n = len(nmw.get_system_position())

    # modes
    modes = nmw.get_modes()

    # special values
    r_bound_value = n * relative_bound ** 2
    a_bound_value = n * absolute_bound ** 2

    # initial state
    iteration_count = 0
    states = []
    energies = []

    anti_gradient = None

    energy_init = nmw.get_energy()

    position_init = nmw.get_position().copy()

    energies.append(energy_init)
    states.append(nmw.get_system_position().copy())
    # main cycle
    while True:
        # it is a weighted anti-gradient!
        anti_gradient = nmw.get_force().copy()

        # find a step
        a = np.dot(anti_gradient, anti_gradient)
        upper_bound = (r_bound_value / a) ** 0.5
        lower_bound = 0
        print("bounds:", lower_bound, ";", upper_bound)

        # find optimal step
        folds = np.zeros((n_samples,))
        eng = np.ones((n_samples,)) * float("inf")
        folds[0] = 1
        nmw.set_position(folds[0] * upper_bound * anti_gradient + position_init)
        eng[0] = nmw.get_energy()
        for i in range(1, n_samples):
            folds[i] = folds[i - 1] * fold_parameter
            if folds[i] * upper_bound <= lower_bound:
                break
            nmw.set_position(folds[i] * upper_bound * anti_gradient + position_init)
            eng[i] = nmw.get_energy()
            if energy_init > eng[i] > eng[i - 1]:
                break

        # update state list
        j = np.argmin(eng)
        nmw.set_position(folds[j] * upper_bound * anti_gradient + position_init)
        states.append(nmw.get_system_position().copy())
        energies.append(eng[j])

        iteration_count += 1
        if termination == "growth":
            if energy_init < eng[j]:
                states.pop()
                energies.pop()
                break
        elif termination == "etol":
            if abs(eng[j] - energy_init) < etol:
                break
        else:
            raise ValueError(f"Wrong termination criterion: {termination}")
        if iteration_count >= max_iter:
            break

        position_init = folds[j] * upper_bound * anti_gradient + position_init
        energy_init = eng[j]

    if not return_traj:
        return {"states": states[-1:], "energies": energies[-1:]}
    else:
        return {"states": states, "energies": energies}
