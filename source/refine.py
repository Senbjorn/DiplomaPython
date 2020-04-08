import numpy as np
import quaternion
import time
import datetime as dt
import logging
from enum import Enum

# MDTraj
import mdtraj as mdt

# OpenMM
import simtk.openmm as omm
import simtk.openmm.app as app
from simtk.unit import *

# ProDy
import prody as pdy

import project
from fast_computation import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s') #%(asctime)s - %(process)s -
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_system(path, output_path='../output/tmp_system.pdb', forcefield_name='charmm36.xml'):
    """
    Creates system as .pdb file containing appropriate configuration for DRSystem.
    @param path:  path to .pdb file with initial system.
    @type path: str
    @param output_path: a file containing new system.
    @type output_path: str
    """
    pdy_protein = pdy.parsePDB(path)
    pdy_protein = pdy_protein.select('protein')
    time0 = time.time()
    pdy.writePDB(output_path, pdy_protein)
    print('write PDB(prody): {0:.4f} sec'.format(time.time() - time0))
    time0 = time.time()
    omm_object = app.PDBFile(output_path)
    print('read PDB(openmm):', time.time() - time0, 'sec')
    time0 = time.time()
    forcefield = app.ForceField(forcefield_name)
    modeller = app.Modeller(omm_object.getTopology(), omm_object.getPositions())
    modeller.addHydrogens(forcefield=forcefield, pH=6.4)
    print('add hydrogens and extra particles(openmm):', time.time() - time0, 'sec')
    time0 = time.time()
    with open(output_path, mode='w') as output_file:
        app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), output_file)
    print('write PDB(openmm):', time.time() - time0, 'sec')


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

    def __init__(self, pdb_file, forcefield_name, refine='chain A', static='chain B'):
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
        Updates ligand position
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
        Get ligand
        @return: ligand object
        @rtype: ProDy.AtomGroup
        """

        return self._refine_prot

    def get_energy(self):
        """
        Calculates energy of the whole system
        @return: energy value in kDJ/mol
        @rtype: float
        """

        state = self._simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    def get_force(self):
        """
        Calculates force vector that acts upon a protein to be refined
        @return: force vector in kJ/mole/nm
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
        Translates initial system by vector t and rotates via rotation operator r
        @param t: translation vector t
        @type t: np.ndarray
        @param r: rotation operator
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

    def __init__(self, drs, n_modes=10, cutoff=15):
        """
        Create new wrapper of DRSystem instance. It allows you to operate the system in NM space.
        @param drs: DRSystem instance to wrap.
        @type drs: DRSystem
        @param n_modes: number of modes. Should be less than 3 * NumberOfAtoms - 6.
        @type n_modes: int
        @param cutoff: interaction cutoff default is 15 A.
        @type cutoff: float
        """

        self._position = np.zeros(n_modes)
        self._system = drs
        self._refine_prot_init = self._system._refine_prot.copy()
        self._anm = pdy.ANM('anm')
        self._anm.buildHessian(self._system.get_refine_prot(), cutoff=cutoff)
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


class ProteinComplex:
    """
    A protein complex class

    """

    # protein data
    _cid = None
    _source = None
    _force_field_name = None
    _selections = None

    # omm
    _omm_protein = None
    _simulation = None

    # pdy
    _pdy_protein = None
    _compartments = None

    # cache
    _force_cached = []
    _force_cache = []

    _energy_cached = False
    _energy_cache = None

    def __init__(self, pdb_file, force_field_name, selections, cid='unknown'):
        self._cid = cid
        self._source = pdb_file
        self._force_field_name = force_field_name
        self._selections = selections
        self._compartments = []
        self._pdy_protein = pdy.parsePDB(pdb_file)
        for selection in selections:
            self._compartments.append(self._pdy_protein.select(selection))
            self._force_cache.append(None)
            self._force_cached.append(False)

        self._omm_protein = app.PDBFile(pdb_file)
        forcefield = app.ForceField(self._force_field_name)

        integrator = omm.LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
        system = forcefield.createSystem(
            self._omm_protein.topology,
            constraints=app.HBonds,
        )
        self._simulation = app.Simulation(self._omm_protein.topology, system, integrator)
        self._simulation.context.setPositions(self._omm_protein.positions)

    def get_coords(self, index):
        return self._compartments[index].getCoords()

    def set_coords(self, index, new_coords):
        self._compartments[index].setCoords(new_coords)
        iterator = self._compartments[index].iterAtoms()
        i = 0
        for atom in iterator:
            self._omm_protein.positions[atom.getIndex()] = omm.Vec3(*new_coords[i]) * nanometer / 10
            i += 1
        self._simulation.context.setPositions(self._omm_protein.positions)
        self._energy_cached = False
        for i in range(len(self)):
            self._force_cached[i] = False

    def get_force(self, index):
        if self._force_cached[index]:
            return self._force_cache[index]
        state = self._simulation.context.getState(getForces=True)
        all_forces = np.array(state.getForces().value_in_unit(kilojoule_per_mole / angstrom))
        iterator = self._compartments[index].iterAtoms()
        i = 0
        forces = []
        for atom in iterator:
            forces.append(all_forces[atom.getIndex()])
            i += 1
        self._force_cache[index] = np.array(forces)
        self._force_cached[index] = True
        return self._force_cache[index]

    def get_energy(self):
        if self._energy_cached:
            return self._energy_cache
        state = self._simulation.context.getState(getEnergy=True)
        self._energy_cache = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        self._energy_cached = True
        return self._energy_cache

    def get_compartment(self, index):
        return self._compartments[index].copy()

    def get_bond_lengths(self, length_unit=angstrom):
        lengths = []
        for i, bond in enumerate(self._omm_protein.getTopology().bonds()):
            atom_1 = bond[0]
            atom_2 = bond[1]
            pos_1 = self._omm_protein.getPositions()[atom_1.index].value_in_unit(length_unit)
            pos_2 = self._omm_protein.getPositions()[atom_2.index].value_in_unit(length_unit)
            lengths.append(np.linalg.norm(pos_2 - pos_1))
        return np.array(lengths)

    def copy(self):
        return ProteinComplex(self._source, self._force_field_name, self._selections)

    def to_pdb(self, handle):
        self._omm_protein.writeFile(positions=self._omm_protein.positions,
                                    topology=self._omm_protein.topology,
                                    file=handle)

    def __len__(self):
        return len(self._compartments)


class RestrictionWrapper:
    """
    Base interface for wrappers

    """

    _protein_complex = None

    def __init__(self, pc):
        """
        @param pc: protein complex data.
        @type pc: ProteinComplex
        """

        self._protein_complex = pc

    def set_position(self, index, new_pos):
        pass

    def get_position(self, index):
        pass

    def get_force(self, index):
        pass

    def get_energy(self):
        pass

    def get_compartment(self, index):
        pass

    def copy(self):
        pass

    def __len__(self):
        return len(self._protein_complex)


class RMRestrictionWrapper(RestrictionWrapper):
    """
    Wraps the original protein complex allowing movement only along normal mode and rigid transformation

    """
    _mode_params = None
    _positions = None
    _anms = None
    _modes = None
    _eigenvalues = None
    _init_coords = None
    _c_tensors = None
    _i_tensors = None
    _d_tensors = None
    _f_tensors = None
    _weights = None


    # cache
    _forces_cached = None
    _forces_cache = None

    def __init__(self, pc, mode_params):
        super().__init__(pc)
        self._mode_params = mode_params
        self._positions = []
        self._anms = []
        self._modes = []
        self._eigenvalues = []
        self._init_coords = []
        self._c_tensors = []
        self._i_tensors = []
        self._d_tensors = []
        self._f_tensors = []
        self._weights = []

        # init position
        self._positions = []
        for i in range(len(self._protein_complex)):
            self._positions.append([np.zeros((3,)), np.quaternion(1, 0, 0, 0), np.zeros((mode_params[i]['nmodes'],))])

        # init modes
        for i in range(len(self._protein_complex)):

            if mode_params[i]['nmodes'] == 0:
                self._anms.append(None)
                self._modes.append(None)
                self._eigenvalues.append(None)
                continue
            anm = pdy.ANM(f'{self._protein_complex._cid} {i + 1}')

            tmp_struct_path = str(project.output_path / f'{self._protein_complex._cid}_{i + 1}.pdb')
            pdy.writePDB(tmp_struct_path, self._protein_complex.get_compartment(i))
            omm_protein = app.PDBFile(tmp_struct_path)
            hessian = surface_bond_hessian(omm_protein, surface_constant=10e+0, depth=4.4,
                                           cutoff=mode_params[i]['cutoff'],
                                           unbonded_constant=1, bonded_constant=10)
            # hessian = bond_hessian_modification(hessian, omm_protein, cutoff=mode_params[i]['cutoff'])
            anm.setHessian(hessian)

            # anm.buildHessian(self._protein_complex.get_compartment(i), cutoff=mode_params[i]['cutoff'])
            anm.calcModes(n_modes=mode_params[i]['nmodes'], zeros=False)

            self._anms.append(anm)
            self._modes.append(anm.getEigvecs().copy().T)
            self._eigenvalues.append(anm.getEigvals().copy())

        # init init_state and tensors
        for i in range(len(self._protein_complex)):
            coords = self._protein_complex.get_coords(i)
            modes = self._modes[i]
            natoms = len(coords)
            weights = np.ones((natoms, ))
            self._weights.append(weights)
            self._init_coords.append(coords)
            self._c_tensors.append(build_c_tensor(coords, weights))
            self._i_tensors.append(build_i_tensor(coords, self._c_tensors[i], weights))
            if modes is None:
                self._d_tensors.append(None)
                self._f_tensors.append(None)
            else:
                self._d_tensors.append(build_d_tensor(coords, modes, weights))
                self._f_tensors.append(build_f_tensor(coords, modes, weights))

    def set_position(self, index, new_pos):
        self._positions[index] = new_pos
        position = self._positions[index]
        trans = position[0]
        quat = position[1]
        mode_pos = position[2]
        init_coords = self._init_coords[index]
        c_tensor = self._c_tensors[index]
        natoms = len(init_coords)
        coords = []
        modes = self._modes[index]
        if modes is None:
            mode_offset = np.zeros((natoms, 3))
        else:
            mode_offset = np.dot(mode_pos, modes).reshape((natoms, 3))
        for a in range(natoms):
            pos = quat * (np.quaternion(0, *(init_coords[a] - c_tensor + mode_offset[a])) * (1 / quat))
            pos = quaternion.as_float_array(pos)[1:] + trans + c_tensor
            coords.append(pos)
        self._protein_complex.set_coords(index, np.array(coords))

    def get_position(self, index):
        return [self._positions[index][0].copy(), self._positions[index][1].copy(), self._positions[index][2].copy()]

    def get_force(self, index):
        coords = self._protein_complex.get_coords(index)
        force = self._protein_complex.get_force(index)
        natoms = len(coords)
        weights = self._weights[index]
        w = np.sum(weights)
        position = self._positions[index]
        trans = position[0]
        quat = position[1]
        mode_pos = position[2]
        c_tensor = self._c_tensors[index]
        i_tensor = self._i_tensors[index]
        d_tensor = self._d_tensors[index]
        f_tensor = self._f_tensors[index]
        weights = self._weights[index]
        modes = self._modes[index]
        trans_force = np.sum(force, 0)
        torque = calc_torque(coords, force, c_tensor, weights)
        rotation_matrix = quaternion.as_rotation_matrix(quat)
        inertia_tensor = calc_inertia_tensor(rotation_matrix, mode_pos, i_tensor, d_tensor, f_tensor, weights)
        mode_force = np.dot(modes, force.reshape((force.shape[0] * 3,)))
        return [trans_force, torque, inertia_tensor, mode_force]

    def get_energy(self):
        return self._protein_complex.get_energy()

    def get_compartment(self, index):
        return self._protein_complex.get_compartment(index)

    def copy(self):
        pass


def save_trajectory(drs, states, output_file, tmp_file='../output/tmp_system.pdb', log=None):
    tmp_file = str(tmp_file)
    output_file = str(output_file)
    n_states = len(states)
    trj = None
    if log is not None:
        log.write('Trajectory info:\n' +
                  f'\tnumber of states: {n_states}\n' +
                  f'\ttemporary file: {tmp_file}\n' +
                  f'\touput file: {output_file}\n')
    for i in range(n_states):
        drs.set_position(states[i])
        with open(tmp_file, 'w') as input_file:
            drs._omm_protein.writeFile(positions=drs._omm_protein.positions,
                                       topology=drs._omm_protein.topology,
                                       file=input_file)
        if trj is not None:
            trj = trj.join(mdt.load(tmp_file))
        else:
            trj = mdt.load(tmp_file)
        if log is not None:
            if i % 5 == 0:
                log.write(f'creating trajectory: {100 * (i + 1) / n_states:.1f}%\n')
    if log is not None:
        log.write(f'Saving trajectory...\n')
    trj.save_pdb(output_file)
    if log is not None:
        log.write(f'Done!\n')


class CGDMode(Enum):
    FLEXIBLE = 1
    RIGID = 2
    BOTH = 3


def confined_gradient_descent(
        rw, decrement=0.9, relative_bounds_r=(0.01, 3), relative_bounds_s=(0.01, 0.5),
        max_iter=100, save_path=False, extended_result=False, log=False, mode=CGDMode.BOTH):
    """
    Performs gradient descent with respect to a special confinement


    @param rw: system to optimize.
    @type rw: RMRestrictionWrapper
    @param decrement: fold step when choosing optimal step
    @type decrement: float
    @param relative_bounds_r: minimum and maximum rmsd between actual intermediate state and the next one (rigid)
    @type relative_bounds_r: tuple
    @param relative_bounds_s: minimum and maximum rmsd between actual intermediate state and the next one (modes)
    @type relative_bounds_s: tuple
    @param max_iter: maximum number of iterations
    @type max_iter: int
    @param save_path: if true all intermediate states, energies and forces are returned.
        Otherwise, the function returns only final record
    @type save_path: bool
    @param extended_result: if true additional information is returned
    @type extended_result: bool
    @param log: if true a log will be printed to the console
    @type log: bool
    @param mode: there are three modes RIGID, FLEXIBLE and BOTH
    @type mode: CGDMode
    @return: dictionary containing all the results.
        "states" - list of all states along optimization path
        "energies" - list of all energies along optimization path
        "forces" - list of all forces along optimization path
        If return_traj is false returns only last record
    @rtype: dict
    """
    if log:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info('INITIALIZATION SATGE')
    optimization_result = {
        'positions': [],
        'energies': [],
        'forces': [],
        'coords': [],
        'success': True,
    }

    optimization_result['positions'].append(rw.get_position(0))
    optimization_result['energies'].append(rw.get_energy())
    optimization_result['forces'].append(rw.get_force(0))
    optimization_result['coords'].append(rw._protein_complex.get_coords(0))

    if extended_result:
        optimization_result['translation_diff'] = []
        optimization_result['rotation_diff'] = []
        optimization_result['mode_diff'] = []
        optimization_result['rigid_rmsd'] = []
        optimization_result['flexible_rmsd'] = []

    k = 0
    while k < max_iter:
        logger.info(f'ITERATION {k} START')
        position = optimization_result['positions'][-1]
        energy = optimization_result['energies'][-1]
        force = optimization_result['forces'][-1]
        coords = optimization_result['coords'][-1]

        f_trans = force[0]
        torque = force[1]
        inertia_inv = np.linalg.inv(force[2])
        f_modes = force[3]
        w = np.sum(rw._weights[0])

        iinv_t = inertia_inv.dot(torque)
        iinv_t24 = iinv_t.dot(iinv_t) / 4
        ft24w = f_trans.dot(f_trans) / 4 / w
        tit = torque.dot(iinv_t)
        wrmsd20 = w * relative_bounds_r[0] ** 2
        wrmsd21 = w * relative_bounds_r[1] ** 2
        a = ft24w * iinv_t24
        b0 = ft24w + tit - wrmsd20 * iinv_t24
        c0 = -wrmsd20
        b1 = ft24w + tit - wrmsd21 * iinv_t24
        c1 = -wrmsd21
        roots0 = np.roots([a, b0, c0])
        roots1 = np.roots([a, b1, c1])
        tau0 = np.max(roots0) ** 0.25
        tau1 = np.max(roots1) ** 0.25
        logger.info(f'tau: {tau0}, {tau1}')

        mcoeff = (4 * w / f_modes.dot(f_modes)) ** 0.25
        mtau0 = relative_bounds_s[0] ** 0.5 * mcoeff
        mtau1 = relative_bounds_s[1] ** 0.5 * mcoeff
        logger.info(f'mtau: {mtau0}, {mtau1}')

        logger.info('LINEAR SEARCH')
        while (tau1 > tau0 or mode == CGDMode.FLEXIBLE) and (mtau1 > mtau0 or mode == CGDMode.RIGID):
            tdiff = tau1 ** 2 / w / 2 * f_trans
            qdiff = np.quaternion(1, *(tau1 ** 2 / 2 * iinv_t))
            qdiff /= qdiff.norm()
            mdiff = mtau1 ** 2 / 2 * f_modes
            trans = position[0]
            quat = position[1]
            modes = position[2]
            if mode == CGDMode.BOTH:
                trans = tdiff + trans
                quat = qdiff * quat
                modes = mdiff + modes
            elif mode == CGDMode.RIGID:
                trans = tdiff + trans
                quat = qdiff * quat
            elif mode == CGDMode.FLEXIBLE:
                modes = mdiff + modes
            new_pos = [trans, quat, modes]

            # rmsd
            if extended_result:
                logger.info(f'EXT::tdiff: {tdiff}')
                logger.info(f'EXT::qdiff: {qdiff}')
                logger.info(f'EXT::mdiff: {mdiff}')
                coords1 = coords
                test_pos = [trans, quat, position[2]]
                rw.set_position(0, test_pos)
                coords2 = rw._protein_complex.get_coords(0)
                weights = rw._weights[0]
                rrmsd = rmsd(coords1, coords2, weights)
                logger.info(f'EXT::RRMSD: {rrmsd}')
                frmsd = ((mdiff.dot(mdiff)) / np.sum(weights)) ** 0.5
                logger.info(f'EXT::FRMSD: {frmsd}')

            rw.set_position(0, new_pos)
            new_energy = rw.get_energy()
            tau1 *= decrement * decrement
            mtau1 *= decrement * decrement
            logger.info(f'NEW ENERGY: {new_energy}')
            if new_energy < energy:
                break
        if extended_result:
            optimization_result['translation_diff'].append(tdiff)
            optimization_result['rotation_diff'].append(qdiff)
            optimization_result['mode_diff'].append(mdiff)
            optimization_result['rigid_rmsd'].append(rrmsd)
            optimization_result['flexible_rmsd'].append(frmsd)

        logger.info(f'ITERATION {k} END')
        optimization_result['energies'].append(rw.get_energy())
        if optimization_result['energies'][-2] < optimization_result['energies'][-1]:
            optimization_result['energies'].pop()
            optimization_result['success'] = False
            break
        optimization_result['positions'].append(rw.get_position(0))
        optimization_result['forces'].append(rw.get_force(0))
        optimization_result['coords'].append(rw._protein_complex.get_coords(0))
        if not save_path:
            optimization_result['positions'].pop(0)
            optimization_result['forces'].pop(0)
            optimization_result['coords'].pop(0)
            optimization_result['energies'].pop(0)
        k += 1
    return optimization_result


class MCGOptimizationResult:
    def __init__(self, extended=False, save_history=False):
        self._optimization_result = {}
        self._extended = extended
        self._save_history = save_history
        self._length = 0
        self.init_optimization_result_main()
        if self._extended:
            self.init_optimization_result_extended()

    def init_optimization_result_main(self):
        self._optimization_result['forces'] = []
        self._optimization_result['energies'] = []
        self._optimization_result['coords'] = []
        self._optimization_result['positions'] = []
        self._optimization_result['success'] = True

    def init_optimization_result_extended(self):
        self._optimization_result['translation_diff'] = []
        self._optimization_result['rotation_diff'] = []
        self._optimization_result['mode_diff'] = []
        self._optimization_result['rigid_rmsd'] = []
        self._optimization_result['flexible_rmsd'] = []

    def update_main(self, rw):
        positions = [rw.get_position(i) for i in range(len(rw))]
        energy = rw.get_energy()
        forces = [rw.get_force(i) for i in range(len(rw))]
        coords = [rw._protein_complex.get_coords(i) for i in range(len(rw))]
        self._optimization_result['positions'].append(positions)
        self._optimization_result['energies'].append(energy)
        self._optimization_result['forces'].append(forces)
        self._optimization_result['coords'].append(coords)
        self._length += 1

    def get_main(self, index, pos=-1):
        position = self._optimization_result['positions'][pos][index]
        force = self._optimization_result['forces'][pos][index]
        coords = self._optimization_result['coords'][pos][index]
        energy = self._optimization_result['energies'][pos]
        return {'energy': energy, 'position': position, 'force': force, 'coords': coords}

    def get_energy(self, pos=-1):
        energy = self._optimization_result['energies'][pos]
        return energy

    def update_translation_diff(self, td):
        self._optimization_result['translation_diff'].append(td)

    def update_mode_diff(self, md):
        self._optimization_result['mode_diff'].append(md)

    def update_rotation_diff(self, rd):
        self._optimization_result['rotation_diff'].append(rd)

    def update_rigid_rmsd(self, rr):
        self._optimization_result['rigid_rmsd'].append(rr)

    def update_flexible_rmsd(self, fr):
        self._optimization_result['flexible_rmsd'].append(fr)

    def get_extended(self, index, pos=-1):
        translation_diff = self._optimization_result['translation_diff'][pos][index]
        mode_diff = self._optimization_result['mode_diff'][pos][index]
        rotation_diff = self._optimization_result['rotation_diff'][pos][index]
        rigid_rmsd = self._optimization_result['rigid_rmsd'][pos][index]
        flexible_rmsd = self._optimization_result['flexible_rmsd'][pos][index]
        return {'translation_diff': translation_diff, 'mode_diff': mode_diff,
                'rotation_diff': rotation_diff, 'rigid_rmsd': rigid_rmsd,
                'flexible_rmsd': flexible_rmsd}

    def get_status(self):
        return self._optimization_result['success']

    def set_status(self, status):
        self._optimization_result['success'] = status

    def __len__(self):
        return self._length


def multimolecule_confined_gradient_descent(
        rw, decrement=0.9, relative_bounds_r=(0.01, 3), relative_bounds_s=(0.01, 0.5),
        max_iter=100, save_history=False, extended_result=False, log=False, mode=CGDMode.BOTH):
    """
    Performs gradient descent with respect to a special confinement


    @param rw: system to optimize.
    @type rw: RMRestrictionWrapper
    @param decrement: fold step when choosing optimal step
    @type decrement: float
    @param relative_bounds_r: minimum and maximum rmsd between actual intermediate state and the next one (rigid)
    @type relative_bounds_r: tuple
    @param relative_bounds_s: minimum and maximum rmsd between actual intermediate state and the next one (modes)
    @type relative_bounds_s: tuple
    @param max_iter: maximum number of iterations
    @type max_iter: int
    @param save_history: if true all intermediate states, energies and forces are returned.
        Otherwise, the function returns only final record
    @type save_history: bool
    @param extended_result: if true additional information is returned
    @type extended_result: bool
    @param log: if true a log will be printed to the console
    @type log: bool
    @param mode: there are three modes RIGID, FLEXIBLE and BOTH
    @type mode: CGDMode
    @return: dictionary containing all the results.
        'states' - list of all states along optimization path
        'energies' - list of all energies along optimization path
        'forces' - list of all forces along optimization path
        If return_traj is false returns only last record
    @rtype: dict
    """
    if log:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info('INITIALIZATION SATGE')
    optimization_result = MCGOptimizationResult(extended=extended_result)

    optimization_result.update_main(rw)

    k = 0
    M = len(rw)
    while k < max_iter:
        logger.info(f'ITERATION {k} START')

        tau_list = []
        mtau_list = []
        iinv_t_list = []
        w_list = []
        logger.info('Advancement region computation'.upper())
        for i in range(M):
            logger.info(f'SELECTION:{i}')
            record = optimization_result.get_main(i)
            position = record['position']
            force = record['force']
            energy = record['energy']
            coord = record['coords']

            f_trans = force[0]
            torque = force[1]
            inertia_inv = np.linalg.inv(force[2])
            f_modes = force[3]
            w = np.sum(rw._weights[0])
            w_list.append(w)
            iinv_t = inertia_inv.dot(torque)
            iinv_t_list.append(iinv_t)
            iinv_t24 = iinv_t.dot(iinv_t) / 4
            ft24w = f_trans.dot(f_trans) / 4 / w
            tit = torque.dot(iinv_t)
            wrmsd20 = w * relative_bounds_r[0] ** 2
            wrmsd21 = w * relative_bounds_r[1] ** 2
            a = ft24w * iinv_t24
            b0 = ft24w + tit - wrmsd20 * iinv_t24
            c0 = -wrmsd20
            b1 = ft24w + tit - wrmsd21 * iinv_t24
            c1 = -wrmsd21
            roots0 = np.roots([a, b0, c0])
            roots1 = np.roots([a, b1, c1])
            tau0 = np.max(roots0) ** 0.25
            tau1 = np.max(roots1) ** 0.25
            tau_list.append((tau0, tau1))
            logger.info(f'tau: {tau0}, {tau1}')

            mcoeff = (4 * w / f_modes.dot(f_modes)) ** 0.25
            mtau0 = relative_bounds_s[0] ** 0.5 * mcoeff
            mtau1 = relative_bounds_s[1] ** 0.5 * mcoeff
            mtau_list.append((mtau0, mtau1))
            logger.info(f'mtau: {mtau0}, {mtau1}')
        # get minimal interval
        logger.info('MINIMAL ADVANCEMENT REGION')
        min_tau0 = np.min([t0 for t0, t1 in tau_list])
        min_tau1 = np.min([t1 for t0, t1 in tau_list])
        min_mtau0 = np.min([mt0 for mt0, mt1 in mtau_list])
        min_mtau1 = np.min([mt1 for mt0, mt1 in mtau_list])
        logger.info(f'TAU: {min_tau0}, {min_mtau1}')
        logger.info(f'MTAU: {min_mtau0}, {min_mtau1}')
        # for all selections
        logger.info('LINEAR SEARCH')

        # extended results
        tdiff_list = []
        rrmsd_list = []
        frmsd_list = []
        qdiff_list = []
        mdiff_list = []



        logger.info('SYSTEM EVOLUTION')
        for i in range(M):
            logger.info(f'SELECTION:{i}')
            record = optimization_result.get_main(i)
            iinv_t = iinv_t_list[i]
            w = w_list[i]
            position = record['position']
            force = record['force']
            energy = record['energy']
            coords = record['coords']

            f_trans = force[0]
            torque = force[1]
            f_modes = force[3]

            tau0 = min_tau0
            tau1 = min_tau1
            mtau0 = min_mtau0
            mtau1 = min_mtau1

            while (tau1 > tau0 or mode == CGDMode.FLEXIBLE) and (mtau1 > mtau0 or mode == CGDMode.RIGID):
                tdiff = tau1 ** 2 / w / 2 * f_trans
                qdiff = np.quaternion(1, *(tau1 ** 2 / 2 * iinv_t))
                qdiff /= qdiff.norm()
                mdiff = mtau1 ** 2 / 2 * f_modes
                trans = position[0]
                quat = position[1]
                modes = position[2]
                if mode == CGDMode.BOTH:
                    trans = tdiff + trans
                    quat = qdiff * quat
                    modes = mdiff + modes
                elif mode == CGDMode.RIGID:
                    trans = tdiff + trans
                    quat = qdiff * quat
                elif mode == CGDMode.FLEXIBLE:
                    modes = mdiff + modes
                new_pos = [trans, quat, modes]

                # rmsd
                if extended_result:
                    tdiff_list.append(tdiff)
                    qdiff_list.append(qdiff)
                    mdiff_list.append(mdiff)
                    logger.info(f'EXT::tdiff: {tdiff}')
                    logger.info(f'EXT::qdiff: {qdiff}')
                    logger.info(f'EXT::mdiff: {mdiff}')
                    coords1 = coords
                    test_pos = [trans, quat, position[2]]
                    rw.set_position(i, test_pos)
                    coords2 = rw._protein_complex.get_coords(0)
                    weights = rw._weights[i]
                    rrmsd = rmsd(coords1, coords2, weights)
                    rrmsd_list.append(rrmsd)
                    logger.info(f'EXT::RRMSD: {rrmsd}')
                    frmsd = ((mdiff.dot(mdiff)) / np.sum(weights)) ** 0.5
                    frmsd_list.append(frmsd)
                    logger.info(f'EXT::FRMSD: {frmsd}')

                rw.set_position(0, new_pos)
                new_energy = rw.get_energy()
                tau1 *= decrement * decrement
                mtau1 *= decrement * decrement
                logger.info(f'NEW ENERGY: {new_energy}')
                if new_energy < energy:
                    break

        logger.info(f'ITERATION {k} END')
        current_energy = rw.get_energy()
        previous_energy = optimization_result.get_energy()
        if previous_energy < current_energy:
            optimization_result.set_status(False)
            break
        optimization_result.update_main(rw)
        if extended_result:
            optimization_result.update_translation_diff(tdiff_list)
            optimization_result.update_rotation_diff(qdiff_list)
            optimization_result.update_mode_diff(mdiff_list)
            optimization_result.update_rigid_rmsd(rrmsd_list)
            optimization_result.update_flexible_rmsd(frmsd_list)
        k += 1
    return optimization_result


def bond_hessian(omm_protein, bound_constant=10, unbound_constant=1, cutoff=7.5):
    """
    Calculates hessian for anisotropic network model where bound interactions are stiffer the unbound

    @param protein: protein
    @type protein: prody selection
    @param bound_constant: bound interaction constant
    @type bound_constant: float
    @param unbound_constant: unbound interaction constant
    @type unbound_constant: float
    @return: hessian matrix
    """

    n = omm_protein.topology.getNumAtoms()
    hessian_matrix = np.zeros((3 * n, 3 * n))
    bonded_to = {a.index: set() for a in omm_protein.topology.atoms()}
    positions = omm_protein.getPositions().value_in_unit(angstrom)
    for bond in omm_protein.topology.bonds():
        index1 = bond.atom1.index
        index2 = bond.atom2.index
        bonded_to[index1].add(index2)
        bonded_to[index2].add(index1)
    for i, atom1 in enumerate(omm_protein.topology.atoms()):
        for j, atom2 in enumerate(omm_protein.topology.atoms()):
            if i == j:
                continue
            pos1 = positions[atom1.index]
            pos2 = positions[atom2.index]
            rij = np.linalg.norm(pos2 - pos1)
            if rij >= cutoff:
                continue
            if atom2.index in bonded_to[atom1.index]:
                k = -bound_constant / rij ** 2
            else:
                k = -unbound_constant / rij ** 2
            for a in range(3):
                for b in range(3):
                    hessian_matrix[i * 3 + a, j * 3 + b] = k * (pos2[a] - pos1[a]) * (pos2[b] - pos1[b])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hessian_matrix[(i * 3):(i * 3 + 3), (i * 3):(i * 3 + 3)] -= hessian_matrix[(i * 3):(i * 3 + 3), (j * 3):(j * 3 + 3)]
    return hessian_matrix


def hydrogen_hessian(omm_protein, hydrogen_constant=10, heavy_constant=1, cutoff=7.5):
    n = omm_protein.topology.getNumAtoms()
    hessian_matrix = np.zeros((3 * n, 3 * n))
    positions = omm_protein.getPositions().value_in_unit(angstrom)
    for i, atom1 in enumerate(omm_protein.topology.atoms()):
        for j, atom2 in enumerate(omm_protein.topology.atoms()):
            if i == j:
                continue
            pos1 = positions[atom1.index]
            pos2 = positions[atom2.index]
            rij = np.linalg.norm(pos2 - pos1)
            if rij >= cutoff:
                continue
            if atom2.element == app.element.hydrogen or atom1.element == app.element.hydrogen:
                print('cool!')
                k = -hydrogen_constant / rij ** 2
            else:
                k = -heavy_constant / rij ** 2
            for a in range(3):
                for b in range(3):
                    hessian_matrix[i * 3 + a, j * 3 + b] = k * (pos2[a] - pos1[a]) * (pos2[b] - pos1[b])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hessian_matrix[(i * 3):(i * 3 + 3), (i * 3):(i * 3 + 3)] -= hessian_matrix[(i * 3):(i * 3 + 3), (j * 3):(j * 3 + 3)]
    return hessian_matrix


def surface_hessian(omm_protein, surface_constant=10.0, internal_constant=1, cutoff=7.5, depth=4.5):
    time_start = time.time()
    print(f'Surface hessian: surface_constant={surface_constant} internal_constant={internal_constant} cutoff={cutoff} depth={depth}')
    n = omm_protein.topology.getNumAtoms()
    hessian_matrix = np.zeros((3 * n, 3 * n))
    surface_factors = np.zeros((n,))
    positions = omm_protein.getPositions().value_in_unit(angstrom)
    ps, _ = find_surface(omm_protein, grid_step=1, depth=depth)
    ps = set(ps)
    for i, atom1 in enumerate(omm_protein.topology.atoms()):
        for j, atom2 in enumerate(omm_protein.topology.atoms()):
            if i == j:
                continue
            pos1 = positions[atom1.index]
            pos2 = positions[atom2.index]
            rij = np.linalg.norm(pos2 - pos1)
            if rij >= cutoff:
                continue
            if atom1 in ps and atom2 in ps:
                k = -surface_constant / rij ** 2
            else:
                k = -internal_constant / rij ** 2
            for a in range(3):
                for b in range(3):
                    hessian_matrix[i * 3 + a, j * 3 + b] = k * (pos2[a] - pos1[a]) * (pos2[b] - pos1[b])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hessian_matrix[(i * 3):(i * 3 + 3), (i * 3):(i * 3 + 3)] -= hessian_matrix[(i * 3):(i * 3 + 3), (j * 3):(j * 3 + 3)]
    print(f'Surface hessian has been computed in {dt.timedelta(seconds=time.time() - time_start)}')
    return hessian_matrix


def bond_hessian_modification(hessian, omm_protein, unbonded_constant=1, bonded_constant=10, cutoff=7.5):
    """
    Modifies hessian for anisotropic network model where bound interactions are stiffer the unbound

    @param protein: protein
    @type protein: prody selection
    @param bonded_constant: bound interaction constant
    @type bonded_constant: float
    @param unbonded_constant: unbound interaction constant
    @type unbonded_constant: float
    @param cutoff: interaction cutoff
    @type cutoff: float
    @return: hessian matrix
    """
    time_start = time.time()
    print(f'Bond hessian: unbonded_constant={unbonded_constant} bonded_constant={bonded_constant} cutoff={cutoff}')
    n = omm_protein.topology.getNumAtoms()
    hessian_matrix = hessian.copy()
    bonded_to = {a.index: set() for a in omm_protein.topology.atoms()}
    positions = omm_protein.getPositions().value_in_unit(angstrom)
    for bond in omm_protein.topology.bonds():
        index1 = bond.atom1.index
        index2 = bond.atom2.index
        bonded_to[index1].add(index2)
        bonded_to[index2].add(index1)
    for i, atom1 in enumerate(omm_protein.topology.atoms()):
        for j, atom2 in enumerate(omm_protein.topology.atoms()):
            if i == j:
                continue
            pos1 = positions[atom1.index]
            pos2 = positions[atom2.index]
            rij = np.linalg.norm(pos2 - pos1)
            if rij >= cutoff:
                continue
            if atom2.index in bonded_to[atom1.index]:
                k = bonded_constant
            else:
                k = unbonded_constant / rij ** 2
            for a in range(3):
                for b in range(3):
                    hessian_matrix[i * 3 + a, j * 3 + b] *= k

    for i in range(n):
        hessian_matrix[(i * 3):(i * 3 + 3), (i * 3):(i * 3 + 3)] = np.zeros((3, 3))
        for j in range(n):
            if i == j:
                continue
            hessian_matrix[(i * 3):(i * 3 + 3), (i * 3):(i * 3 + 3)] -= hessian_matrix[(i * 3):(i * 3 + 3), (j * 3):(j * 3 + 3)]
    print(f'Bond modification hessian has been computed in {dt.timedelta(seconds=time.time() - time_start)}')
    return hessian_matrix


def surface_bond_hessian(omm_protein, surface_constant=10.0, internal_constant=1, cutoff=7.5, depth=4.5,
                           unbonded_constant=1, bonded_constant=10):
    """
    @param omm_protein: omm protein structure
    @param bonded_constant: bound interaction constant
    @type bonded_constant: float
    @param unbonded_constant: unbound interaction constant
    @type unbonded_constant: float
    @param cutoff: interaction cutoff
    @type cutoff: float
    @return: hessian matrix
    """
    time_start = time.time()
    print(
        f'Surface-bond hessian:\n' +
        f'\tsurface_constant={surface_constant} internal_constant={internal_constant} cutoff={cutoff} depth={depth}\n' +
        f'\tunbonded_constant={unbonded_constant} bonded_constant={bonded_constant}'
    )
    n = omm_protein.topology.getNumAtoms()
    hessian_matrix = np.zeros((3 * n, 3 * n))
    positions = omm_protein.getPositions().value_in_unit(angstrom)
    ps, _ = find_surface(omm_protein, grid_step=1, depth=depth)
    ps = set(ps)
    bonded_to = {a.index: set() for a in omm_protein.topology.atoms()}
    for bond in omm_protein.topology.bonds():
        index1 = bond.atom1.index
        index2 = bond.atom2.index
        bonded_to[index1].add(index2)
        bonded_to[index2].add(index1)
    for i, atom1 in enumerate(omm_protein.topology.atoms()):
        for j, atom2 in enumerate(omm_protein.topology.atoms()):
            if i == j:
                continue
            pos1 = positions[atom1.index]
            pos2 = positions[atom2.index]
            rij = np.linalg.norm(pos2 - pos1)
            if rij >= cutoff:
                continue
            if atom1 in ps and atom2 in ps:
                k = -surface_constant / rij ** 2
            else:
                k = -internal_constant / rij ** 2
            # The key point is that surface and bond constants are applied independently
            if atom2.index in bonded_to[atom1.index]:
                k *= bonded_constant
            else:
                k *= unbonded_constant / rij ** 2
            for a in range(3):
                for b in range(3):
                    hessian_matrix[i * 3 + a, j * 3 + b] = k * (pos2[a] - pos1[a]) * (pos2[b] - pos1[b])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hessian_matrix[(i * 3):(i * 3 + 3), (i * 3):(i * 3 + 3)] -= hessian_matrix[(i * 3):(i * 3 + 3),
                                                                        (j * 3):(j * 3 + 3)]
    print(f'Surface-bond hessian has been computed in {dt.timedelta(seconds=time.time() - time_start)}')
    return hessian_matrix


def get_points(bounds, step):
    v = int(bounds[0])
    points = []
    while v < bounds[1]:
        if v > bounds[0]:
            points.append(v)
        v += step
    return points


def get_neighbours(point, step):
    neighbours = []
    for i in range(3):
        neighbour = list(point)
        neighbour[i] += step
        neighbours.append(tuple(neighbour))
    for i in range(3):
        neighbour = list(point)
        neighbour[i] -= step
        neighbours.append(tuple(neighbour))
    return neighbours


def find_surface(omm_protein, grid_step=2, depth=5.5):
    internal_grid = set()
    atoms = list(omm_protein.topology.atoms())
    positions = omm_protein.getPositions().value_in_unit(angstrom)
    for pos in positions:
        offset = grid_step * 3
        x_bounds = (pos[0] - offset, pos[0] + offset)
        y_bounds = (pos[1] - offset, pos[1] + offset)
        z_bounds = (pos[2] - offset, pos[2] + offset)
        x_points = get_points(x_bounds, grid_step)
        y_points = get_points(y_bounds, grid_step)
        z_points = get_points(z_bounds, grid_step)
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    internal_grid.add((x, y, z))
    surface_grid = set()
    for ip in internal_grid:
        for neighbour in get_neighbours(ip, grid_step):
            if neighbour not in internal_grid:
                surface_grid.add(neighbour)

    protein_surface = []
    protein_internal = []
    for i, a in enumerate(atoms):
        pos = positions[a.index]
        offset = depth
        x_bounds = (pos[0] - offset, pos[0] + offset)
        y_bounds = (pos[1] - offset, pos[1] + offset)
        z_bounds = (pos[2] - offset, pos[2] + offset)
        x_points = get_points(x_bounds, grid_step)
        y_points = get_points(y_bounds, grid_step)
        z_points = get_points(z_bounds, grid_step)
        is_surface = False
        for x in x_points:
            if is_surface:
                break
            for y in y_points:
                if is_surface:
                    break
                for z in z_points:
                    dist = ((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (z - pos[2]) ** 2) ** 0.5
                    if (x, y, z) in surface_grid and dist < depth:
                        protein_surface.append(a)
                        is_surface = True
                        break
        if not is_surface:
            protein_internal.append(a)
    return protein_surface, protein_internal