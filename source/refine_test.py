import pylab
from refine import *


# Unit


def test_1():
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


def test_2():
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


def test_3():
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


def test_4():
    print("START TEST 4")
    path = "../data/2cds.pdb"
    create_system(path)
    t0 = time.time()
    drs = DRSystem("../output/tmp_system.pdb", 'charmm36.xml')
    print("construction of system:", -t0 + time.time(), "sec")
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


if __name__ == "__main__":
    test_11()
    # a = np.array([[1, 0], [0, 1]])
    # b = np.reshape(a, (4, ))
    # b[0] = 3
    # print(a)
    # print(b)