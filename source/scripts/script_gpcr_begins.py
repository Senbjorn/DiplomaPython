# choose force-field
# minimize initial state
# get normal modes
# calculate gradient using force from force-field
# make the first step

import simtk.openmm as mm
import simtk.openmm.app as app
from simtk.unit import *
import prody as pdy
import numpy as np
import time
import logging
from pathlib import Path
import pylab


def create_quad_structure(filename, force_field, tmp_file='../output/input.pdb'):
    pdy_protein = pdy.parsePDB('../data/2cds.pdb')
    pdy_protein = pdy_protein.select('protein')
    time0 = time.time()
    pdy.writePDB(tmp_file, pdy_protein)
    print('write PDB(prody): {0:.4f} sec'.format(time.time() - time0))
    time0 = time.time()
    omm_object = app.PDBFile(tmp_file)
    print('read PDB(openmm):', time.time() - time0, 'sec')
    time0 = time.time()
    modeller = app.Modeller(omm_object.getTopology(), omm_object.getPositions())
    modeller.addHydrogens()
    print('add hydrogens(openmm):', time.time() - time0, 'sec')
    time0 = time.time()
    with open(tmp_file, mode='w') as inputfile:
        app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), inputfile)
    print('write PDB(openmm):', time.time() - time0, 'sec')
    omm_object_h = modeller
    pdy_object_h = pdy.parsePDB(tmp_file)
    pdy_object = pdy_object_h.select('noh')
    return QuadStructure(omm_object_h, omm_object, pdy_object_h, pdy_object, force_field)


def create_pdbAdapter(pdy_protein, forceField, tmp_file='../output/input.pdb'):
    pdy.writePDB(tmp_file, pdy_protein)
    mm_protein = app.PDBFile(tmp_file)
    modeller = app.Modeller(mm_protein.topology, mm_protein.positions)
    modeller.addHydrogens(forceField)
    # modeller.addSolvent(forceField, boxSize=mm.Vec3(5.0, 3.5, 3.5)*nanometers)
    with open(tmp_file, mode='w') as inputfile:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, inputfile)
    return pdbAdapter(tmp_file)


class QuadStructure:
    def __init__(self, omm_object_h, omm_object, pdy_object_h, pdy_object, force_field):
        self.omm_object_h = omm_object_h
        self.omm_object = omm_object
        self.pdy_object_h = pdy_object_h
        self.pdy_object = pdy_object
        self.force_field = force_field
        self.numatoms_h = pdy_object_h.numAtoms()
        self.numatoms = pdy_object.numAtoms()

    def get_data(self, format='omm', type='H'):
        if format == 'omm':
            if type == 'H':
                return self.omm_object_h
            elif type == 'NH':
                return self.omm_object
            else:
                raise AttributeError("wrong type!")
        elif format == 'pdy':
            if type == 'H':
                return self.pdy_object_h
            elif type == 'NH':
                return self.pdy_object
            else:
                raise AttributeError("wrong type!")
        else:
            raise AttributeError("wrong format!")

    def update_data(self, data, format='omm', type='H'):
        if format == 'omm':
            if type == 'H':
                positions = data.getPositions()
                coordinates = []
                for i in range(len(positions)):
                    coordinates.append(np.array([c.value_in_unit(angstrom) for c in positions[i]]))
                self.pdy_object_h.setCoords(np.array(coordinates))
                self.update_data(self.pdy_object_h.select('noh'), format='pdy', type='NH')
            elif type == 'NH':
                self.omm_object.positions = data.getPositions()
                modeller = app.Modeller(data.getTopology(), data.getPositions())
                modeller.addHydrogens(self.force_field)
                print(modeller.getPositions()[0:15])
                print("##########")
                print(self.omm_object.getPositions()[0:15])
                self.omm_object_h = modeller
                positions = self.omm_object_h.getPositions()
                coordinates = []
                for i in range(len(positions)):
                    coordinates.append(np.array([c.value_in_unit(angstrom) for c in positions[i]]))
                self.pdy_object_h.setCoords(np.array(coordinates))
                self.pdy_object = self.pdy_object_h.select('noh')
            else:
                raise AttributeError("wrong type!")
        elif format == 'pdy':
            if type == 'H':
                self.update_data(data.select('noh'), format='pdy', type='NH')
            elif type == 'NH':
                coordinates = data.getCoords()
                positions = []
                for i in range(len(coordinates)):
                    positions.append(mm.Vec3(*coordinates[i]) * nanometer / 10)
                self.omm_object.positions = positions
                self.update_data(self.omm_object, format='omm', type='NH')
            else:
                raise AttributeError("wrong type!")
        else:
            raise AttributeError("wrong format!")

class pdbAdapter:
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], str):
                self.pdy_protein = pdy.parsePDB(args[0])
                self.omm_protein = app.PDBFile(args[0])
            elif isinstance(args[0], pdbAdapter):
                self.pdy_protein = args[0].pdy_protein
                self.omm_protein = args[0].omm_protein
            else:
                raise AttributeError("Error!")
        else:
            raise AttributeError("Error!")

    def update_omm_pos(self, omm_object):
        positions = omm_object.getPositions()
        self.omm_protein.positions = positions
        coords = []
        for i in range(len(self.omm_protein.positions)):
            coords.append(np.array([c.value_in_unit(angstrom) for c in positions[i]]))
        self.pdy_protein.setCoords(np.array(coords))

    def update_pdy_pos(self, pdy_object):
        #TODO code iters
        source = pdy_object.iterAtoms()
        target = self.pdy_protein.iterAtoms()
        coords = self.pdy_protein.getCoords()
        current_atom = next(source)
        for atom in target:
            if atom.getIndex() == current_atom.getIndex():
                coords[atom.getIndex()] = current_atom.getCoords()
                current_atom = next(source)
        self.pdy_protein.setCoords(coords)

        positions = []
        coords = self.pdy_protein.getCoords()
        for i in range(len(coords)):
            positions.append(mm.Vec3(*coords[i]) * nanometer / 10)
        self.omm_protein.positions = positions

    def get_omm_pos(self):
        return self.omm_protein.positions

    def get_pdy_pos(self):
        return self.pdy_protein.getCoords()


class simulationAdapter(pdbAdapter):
    def __init__(self, pdbad, simulation):
        super().__init__(pdbad)
        self.simulation = simulation

    def update(self):
        state = self.simulation.context.getState(getPositions=True)
        self.update_omm_pos(state)

    def getForce(self):
        state = self.simulation.context.getState(getForces=True)
        return state.getForces()

    def getEnergy(self):
        state = self.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()

    #TODO test
    def update_pdy_pos(self, pdy_object):
        super().update_pdy_pos(pdy_object)
        self.simulation.context.setPositions(self.omm_protein.positions)

    #TODO test
    def update_omm_pos(self, omm_object):
        super().update_omm_pos(omm_object)
        self.simulation.context.setPositions(self.omm_protein.positions)


# normal modes for heavy atoms
# energy with hydrogens
class nmoptimizer():
    # simulation contains dimer
    # hydrogens are added on each step (different methods available)
    # pdy and omm are chain A sturctures
    # also we have pdy without hydrogens
    def __init__(self, simadapter):
        pass


#choose forceFiled
force_field = app.ForceField('charmm36.xml')

# number of modes
m = 10
max_rmsd = 2

# create quad
quad = create_quad_structure('../data/2cds.pdb', force_field)
# permutation
omm_object_h = quad.get_data()
omm_object = quad.get_data(type='NH')
pdy_object_h = quad.get_data(format='pdy')
positions = omm_object.getPositions()
for i, pos in enumerate(positions[:15]):
    positions[i] = pos + mm.Vec3(*np.random.random(3)) * nanometer
omm_object.positions = positions
# print(*pdy_object_h.getCoords()[0:15], sep='\n')
quad.update_data(omm_object, format='omm', type='NH')
omm_object_h = quad.get_data()
positions = omm_object_h.getPositions()
omm_object = quad.get_data(type='NH')
pdy_object = quad.get_data(format='pdy', type='NH')
pdy_object_h = quad.get_data(format='pdy', type='H')
coordinates = pdy_object_h.getCoords()
positions1 = omm_object.getPositions()
index = 0
source = pdy_object.iterAtoms()
atom = next(source)
print(pdy_object.numAtoms())
for i in range(0, 15):
    if atom.getIndex() == i:
        array = atom.getCoords() - np.array([c.value_in_unit(angstrom) for c in positions[i]])
        array1  = atom.getCoords() - coordinates[i]
        vector = positions[i] - positions1[index]
        print(i, array, vector, array1)
        atom = next(source)
        index += 1
    else:
        print(i)

'''
#create correct pdb file
selection = pdy.parsePDB('../data/2cds.pdb').select('chain A and protein')
protein = create_pdbAdapter(selection, forceField)
initial = protein.pdy_protein.copy()

# myselect = initial.select('noh')
# print(myselect.numAtoms())
# for a in myselect:
#     print(a.getIndex(), end=' ')
#     if a.getIndex() > 40:
#         break
#
# iter1 = initial.iterAtoms()
# atom = next(iter1)
# print(atom)
# for a in initial.iterAtoms():
#     print(a.getIndex(), end=' ')
#     if a.getIndex() > 20:
#         break

integrator = mm.LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
system = forceField.createSystem(
        protein.omm_protein.topology,
        constraints=app.HBonds,
)
simulation = app.Simulation(protein.omm_protein.topology, system, integrator)
simulation.context.setPositions(protein.omm_protein.positions)
asim = simulationAdapter(protein, simulation)

#minimize
pe0 = asim.getEnergy()
print('energy0:', pe0)
# simulation.minimizeEnergy()
asim.update()

#show result
refined = asim.pdy_protein.copy()
pe1 = asim.getEnergy()
print('energy1:', pe1)
print('dE:', pe0 - pe1)
print('RMSD:', pdy.calcRMSD(initial, refined))

#normal modes with hydrogens
start = asim.pdy_protein.copy().select('noh')
anm = pdy.ANM('modes')
anm.buildHessian(start)
anm.calcModes(n_modes=m, zeros=False)

modes = anm.getEigvecs()
cmodes = modes
cinvmodes = cmodes.T
cmatrix = np.dot(cmodes, cinvmodes)

# do step
# my_force = np.concatenate(asim.getForce().value_in_unit(kilojoule / mole / nanometer))
n = start.numAtoms()
alpha0 = np.zeros(m)
alpha0[1] = 6.
alpha0[3] = 2.
alpha0[8] = -4.5
alpha0 = alpha0 * 4
alpha1 = np.random.random(size=m)
RMSD = 4.
p = np.array([np.dot(alpha1, alpha1), 2 * np.dot(alpha0, alpha1), np.dot(alpha0, alpha0) - RMSD ** 2 * n])
roots = np.roots(p)
# direction = np.dot(cmatrix, my_force)*1.4
direction = np.dot(cmodes, alpha0)
print('init norm:', np.linalg.norm(direction))
stop = start.copy()
pdy.writePDB('../output/sturct1.pdb', start)
stop.setCoords(stop.getCoords() + direction.reshape(len(direction) // 3, 3))
pdy.writePDB('../output/sturct2.pdb', stop)
direction = np.dot(cmodes, alpha1 * roots[1])
stop.setCoords(stop.getCoords() + direction.reshape(len(direction) // 3, 3))
print('RMSD:', pdy.calcRMSD(start, stop))
print('len:', len(direction) // 3)
'''