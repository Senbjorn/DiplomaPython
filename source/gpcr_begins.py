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
from pathlib import Path
import pylab


def create_pdbAdapter(pdy_protein, forceField, tmp_file='../output/input.pdb'):
    pdy.writePDB(tmp_file, pdy_protein)
    mm_protein = app.PDBFile(tmp_file)
    modeller = app.Modeller(mm_protein.topology, mm_protein.positions)
    modeller.addHydrogens(forceField)
    # modeller.addSolvent(forceField, boxSize=mm.Vec3(5.0, 3.5, 3.5)*nanometers)
    with open(tmp_file, mode='w') as inputfile:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, inputfile)
    return pdbAdapter(tmp_file)


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

    def update_omm_pos(self, pos):
        self.omm_protein.positions = pos
        coords = []
        for i in range(len(pos)):
            coords.append(np.array([c.value_in_unit(angstrom) for c in pos[i]]))
        self.pdy_protein.setCoords(np.array(coords))

    def update_pdy_pos(self, pos):
        self.pdy_protein.setCoords(pos)
        positions = []
        coords = self.pdy_protein.getCoords()
        for i in range(len(pos)):
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
        self.update_omm_pos(state.getPositions())

    def getForce(self):
        state = self.simulation.context.getState(getForces=True)
        return state.getForces()

    def getEnergy(self):
        state = self.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()

    # test
    def update_pdy_pos(self, pos):
        super().update_pdy_pos(pos)
        self.simulation.context.setPositions(self.omm_protein.positions)

    # test
    def update_omm_pos(self, pos):
        super().update_omm_pos(pos)
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
forceField = app.ForceField('amber10.xml')
#number of modes
m = 10
max_rmsd = 2 #angstorm

#create correct pdb file
selection = pdy.parsePDB('../data/2cds.pdb').select('chain A and protein')
protein = create_pdbAdapter(selection, forceField)
initial = protein.pdy_protein.copy()

print(initial.getDataLabels())
myselect = initial.select('noh')
print(myselect.numAtoms())
for a in myselect:
    print(a.getIndex(), end=' ')
    if a.getIndex() > 40:
        break

iter1 = initial.iterAtoms()
for a in initial.iterAtoms():
    print(a.getIndex(), end=' ')
    if a.getIndex() > 20:
        break

time_point = time.time()
time_prev = time_point
integrator = mm.LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
system = forceField.createSystem(
        protein.omm_protein.topology,
        constraints=app.HBonds,
)
simulation = app.Simulation(protein.omm_protein.topology, system, integrator)
simulation.context.setPositions(protein.omm_protein.positions)
asim = simulationAdapter(protein, simulation)
time_prev = time_point
time_point = time.time()
print('preparation time:', time_point - time_prev)

#minimize
time_point = time.time()
time_prev = time_point
pe0 = asim.getEnergy()
print('energy0:', pe0)
simulation.minimizeEnergy()
asim.update()
time_prev = time_point
time_point = time.time()
print('minimization time:', time_point - time_prev)

#show result
refined = asim.pdy_protein.copy()
pe1 = asim.getEnergy()
print('energy1:', pe1)
print('dE:', pe0 - pe1)
print('RMSD:', pdy.calcRMSD(initial, refined))

#normal modes with hydrogens
anm = pdy.ANM('modes')
anm.buildHessian(asim.pdy_protein)
anm.calcModes(n_modes=m, zeros=False)

modes = anm.getEigvecs()
cmodes = modes
cinvmodes = cmodes.T
cmatrix = np.dot(cmodes, cinvmodes)

# do step
# my_force = np.concatenate(asim.getForce().value_in_unit(kilojoule / mole / nanometer))
n = 1960
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
myprot = refined.copy()
pdy.writePDB('../output/sturct1.pdb', refined)
myprot.setCoords(myprot.getCoords() + direction.reshape(len(direction) // 3, 3))
pdy.writePDB('../output/sturct2.pdb', myprot)
direction = np.dot(cmodes, alpha1 * roots[1])
myprot.setCoords(myprot.getCoords() + direction.reshape(len(direction) // 3, 3))
print('RMSD:', pdy.calcRMSD(refined, myprot))
print('len:', len(direction) // 3)