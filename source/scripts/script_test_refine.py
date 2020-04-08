import pylab
from refine import *
import project


# Unit


def test_1():
    print("START TEST 1")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of a system:", -t0 + time.time(), "sec")
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


def test_2():
    print("START TEST 2")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of a system:", -t0 + time.time(), "sec")
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


def test_3():
    print("START TEST 3")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of a system:", -t0 + time.time(), "sec")
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


def test_4():
    print("START TEST 4")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of a system:", -t0 + time.time(), "sec")
    t = np.array([1.0, -2., 0.])
    r = np.identity(3, dtype = float)
    drs.set_rigid(t, r)
    init_pos = drs.get_init_position()
    curr_pos = drs.get_position()
    for i in range(len(init_pos)):
        print("offset:", curr_pos[i] - init_pos[i])
    print("OK")
    print("END TEST 4")


def test_11():
    print("START TEST 11")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    nmw = NMSpaceWrapper(drs, n_modes=5)
    print("construction of NM wrapper:", -t0 + time.time(), "sec")
    t = np.array([1.0, -2., 0.])
    r = rotation_matrix(30, 0, 0)
    nmw.set_rigid(t, r)
    init_pos = drs.get_init_position()
    curr_pos = drs.get_position()
    # pdy.showProtein(drs._ligand_init, A='blue', linewidth=1)
    # pdy.showProtein(drs._ligand, A='red', width=1)
    # pylab.show()
    for i in range(len(init_pos)):
        print("offset:", curr_pos[i] - init_pos[i])
    print("OK")
    print("END TEST 11")


def test_12():
    print("START TEST 12")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    nmw = NMSpaceWrapper(drs, n_modes=5)
    print("construction of NM wrapper:", -t0 + time.time(), "sec")
    t = np.array([1.0, -2., 0.])
    r = rotation_matrix(0, 0, 90)
    nmw.set_rigid(t, r)
    pdy.showProtein(drs._refine_prot, A='green', width=1)
    nmw.set_rigid(np.array([0, 0,  0]), rotation_matrix(0, 0, 90))
    init_pos = drs.get_init_position()
    curr_pos = drs.get_position()
    pdy.showProtein(drs._refine_prot_init, A='blue', linewidth=1)
    pdy.showProtein(drs._refine_prot, A='red', width=1)
    pylab.show()
    for i in range(len(init_pos)):
        print("offset:", curr_pos[i] - init_pos[i])
    print("OK")
    print("END TEST 12")


def test_21():
    print("START TEST 21")
    path = str((project.data_path / "may_complex") / "1dfj.pdb")
    print(path)
    create_system(path, output_path=str(project.output_path / "tmp_system.pdb"))
    t0 = time.time()
    drs = DRSystem(str(project.output_path / "tmp_system.pdb"), 'charmm36.xml', refine="chain A", static="chain B")
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    nmw1 = NMSpaceWrapper(drs, n_modes=10)
    print("INIT ENERGY:", nmw1.get_energy())

    # random noise
    nmw1.set_position(np.random.normal(0.0, 2, 10))
    print("RANDOM ENERGY:", nmw1.get_energy())
    nmw2 = NMSpaceWrapper(drs, n_modes=6)
    optimized = confined_gradient_descent(nmw2)
    print("FINAL ENERGY:", optimized["energies"])

    print("construction of NM wrapper:", -t0 + time.time(), "sec")
    print("OK")
    print("END TEST 21")


def test_22():
    print("START TEST 22")
    path = str((project.data_path / "may_complex") / "1dfj.pdb")
    print(path)
    create_system(path, output_path=str(project.output_path / "tmp_system.pdb"))
    t0 = time.time()
    pc = ProteinComplex(str(project.output_path / "tmp_system.pdb"), "charmm36.xml", ["chain A", "chain B"])
    print("construction of system:", -t0 + time.time(), "sec")
    t0 = time.time()
    print("atom count (chain A)", len(pc.get_coords(0)))
    print("atom count (chain B)", len(pc.get_coords(1)))
    print("INIT ENERGY:", pc.get_energy())
    print("OK")
    print("END TEST 22")

def test():
    path_a = "../output/6awr_chainA_prep.pdb"
    path_b = "../output/6awr_chainB_prep.pdb"
    path = "../output/6awr_full_prep1.pdb"
    # protein = pdy.parsePDB(path)
    chain_a = pdy.parsePDB(path_a)
    chain_b = pdy.parsePDB(path_b)
    protein = pdy.parsePDB(path)
    for atom in protein.iterAtoms():
        print(atom.getIndex(), atom.getCSLabels(), atom.getCoords())
    pdy.showProtein(protein.select("chain A"), A='blue')
    pdy.showProtein(protein.select("chain B"), B='red')
    pylab.show()


if __name__ == "__main__":
    project.setup()
    test_22()
    # a = np.array([[1, 0], [0, 1]])
    # b = np.reshape(a, (4, ))
    # b[0] = 3
    # print(a)
    # print(b)