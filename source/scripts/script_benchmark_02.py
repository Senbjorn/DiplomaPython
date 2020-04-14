from core.refine import *
from core.utils import get_random_direction
import os
import pickle
import project
import matplotlib.pyplot as plt
project.setup()


# settings
source_dir = project.data_path / "benchmark" / "modeled"
output_dir = project.output_path / "benchmark_02"
fluctuations_dir = output_dir / "fluctuations"
plots_dir = output_dir / "plots"
sessions_dir = output_dir / "sessions"
trajectories_dir = output_dir / "trajectories"

amber = "amber14/protein.ff14SB.xml"
charmm = "charmm36.xml"
forcefield_name = amber

n_modes = 10
cutoff = 6.5
max_rmsd = 2.0

# save meta info
session = {
    "name": "benchmark_02",
    "forcefield": forcefield_name,
    "n_modes": n_modes,
    "cutoff": cutoff,
    "max_rmsd": max_rmsd,
    "chain_selection": "shortest",
    "modes_method": "stiff_surface_anm",
    "models": {},
}


def save_trajectory(output_path, pc, coords):
    n_states = len(coords)
    trj = None
    for i in range(n_states):
        pc.set_coords(0, coords[i])
        with open(project.output_path / "inter_pdb.pdb", "w") as input_file:
            # pc._omm_protein.writeFile(positions=pc._omm_protein.positions,
            #                           topology=pc._omm_protein.topology,
            #                           file=input_file)
            pc.to_pdb(input_file)
        if trj is not None:
            trj = trj.join(mdt.load(str(project.output_path / "inter_pdb.pdb")))
        else:
            trj = mdt.load(str(project.output_path / "inter_pdb.pdb"))
    trj.save_pdb(output_path)


def calc_rmsd(opt_result, pc, rw, rw1, weights, zero_pos):
    rw.set_position(0, zero_pos)
    p0 = zero_pos
    p1 = opt_result["positions"][0]
    p2 = opt_result["positions"][-1]

    rmsd_record = {}
    rw.set_position(0, p0)
    c1 = pc.get_coords(0)
    rw1.set_position(0, p1)
    c2 = pc.get_coords(0)
    rmsd_record["native-start"] = {"total": rmsd(c1, c2, weights)}

    rw.set_position(0, p0)
    c1 = pc.get_coords(0)
    rw1.set_position(0, p2)
    c2 = pc.get_coords(0)
    rmsd_record["native-result"] = {"total": rmsd(c1, c2, weights)}

    rw1.set_position(0, p1)
    c1 = pc.get_coords(0)
    rw1.set_position(0, p2)
    c2 = pc.get_coords(0)
    rmsd_record["start-result"] = {"total": rmsd(c1, c2, weights)}

    return rmsd_record


def copy_extended_result(result):
    new_result = {}
    new_result["energies"] = [eng for eng in result["energies"]]
    new_result["positions"] = [
        [p[0].copy(), quaternion.as_float_array(p[1]), p[2].copy()] for p in result["positions"]]
    new_result["coords"] = [c.copy() for c in result["coords"]]
    new_result["translation_diff"] = [t.copy() for t in result["translation_diff"]]
    new_result["rotation_diff"] = [quaternion.as_float_array(r) for r in result["rotation_diff"]]
    new_result["mode_diff"] = [m.copy() for m in result["mode_diff"]]
    new_result["rigid_rmsd"] = [rrmsd for rrmsd in result["rigid_rmsd"]]
    new_result["flexible_rmsd"] = [srmsd for srmsd in result["flexible_rmsd"]]
    return new_result


def benchmark(pdb_path, name):
    print("Construction of data structures")

    model = {}
    session["models"][name] = model

    omm_structure = app.PDBFile(pdb_path)
    chains = list(omm_structure.topology.chains())
    # the first chain is the shortest one
    chains.sort(key=lambda c: len(list(c.residues())))
    selections = [f"chain {chain.id}" for chain in chains]
    print("Protein complex...")
    pc = ProteinComplex(pdb_path, forcefield_name, selections, cid=name)
    mode_params = [
        {"nmodes": n_modes, "cutoff": cutoff},
        {"nmodes": 0}
    ]
    print("Restriction...")
    rw = RMRestrictionWrapper(pc, mode_params)

    zero_pos = [np.zeros((3,)), np.quaternion(1, 0, 0, 0), np.zeros((n_modes,))]
    weights = rw._weights[0]
    native_energy = rw.get_energy()
    native_coords = pc.get_coords(0)

    print(f"Native energy: {native_energy}")
    model["native_energy"] = native_energy

    # create anm output dir
    anm_output_dir = fluctuations_dir / name
    if not os.path.exists(str(anm_output_dir)):
        os.mkdir(str(anm_output_dir))
    for i in range(n_modes):
        pdy.showMode(rw._anms[0][i])
        plt.savefig(str(anm_output_dir / f"anm_{name}_{i + 1}.png"), fmt="png")
        plt.clf()

    print("Generating an offset...")
    angle = 10 * np.pi / 180
    while True:
        translation = (rw._c_tensors[1] - rw._c_tensors[0])
        translation = ((translation / np.linalg.norm(translation)) + np.random.uniform(-0.1, 0.1, 3)) * 1.5
        direction = get_random_direction()
        rotation = np.quaternion(np.cos(angle / 2), *(direction * np.sin(angle / 2)))
        modes = np.sum(rw._weights[0]) ** 0.5 * np.random.uniform(-0.3, 0.3, n_modes)
        rw.set_position(0, [translation, rotation, modes])
        if rmsd(native_coords, pc.get_coords(0), weights) <= max_rmsd:
            break
        else:
            angle *= 0.9
    # save initial parameters
    model["angle"] = angle * 180 / np.pi
    model["translation"] = translation
    model["rotation"] = quaternion.as_float_array(rotation)
    model["modes"] = modes

    print("A new restriction...")
    rw1 = RMRestrictionWrapper(pc, mode_params)
    for i in range(n_modes):
        pdy.showMode(rw1._anms[0][i])
        plt.savefig(str(anm_output_dir / f"new_anm_{name}_{i + 1}.png"), fmt="png")
        plt.clf()


    print("New position rw:", rw.get_position(0))
    print("Start energy:", rw1.get_energy())
    result = confined_gradient_descent(rw1, save_path=True, extended_result=True, log=True)

    rw1.set_position(0, result["positions"][0])
    print("New position rw1:", rw1.get_position(0))
    print("Start energy:", rw1.get_energy())
    result_rigid = confined_gradient_descent(rw1, save_path=True, extended_result=True, log=True, mode=CGDMode.RIGID)
    # save the result
    model["result"] = copy_extended_result(result)
    model["result_rigid"] = copy_extended_result(result_rigid)

    # rmsd
    rmsd_record = calc_rmsd(result, pc, rw, rw1, weights, zero_pos)
    model["rmsd_record"] = rmsd_record
    rmsd_record_rigid = calc_rmsd(result_rigid, pc, rw, rw1, weights, zero_pos)
    model["rmsd_record_rigid"] = rmsd_record_rigid

    # add the native complex
    trajecotory = [c.copy() for c in result["coords"]]
    trajecotory.insert(0, native_coords)
    # save trajectories
    save_trajectory(str(trajectories_dir / f"{name}_trajectory.pdb"), pc, trajecotory)

    # build the optimization profile
    coords = result["coords"]
    energies_log10 = np.log10(np.array(result["energies"]) - min(np.min(result["energies"]), native_energy) + 1)
    total_rmsd = np.array([rmsd(coords[i], coords[i + 1], weights) for i in range(len(coords) - 1)])
    rigid_rmsd = np.array([np.linalg.norm(v) for v in result["rigid_rmsd"]])[:-1]
    flexible_rmsd = np.array([np.linalg.norm(v) for v in result["flexible_rmsd"]])[:-1]

    # Energy plot
    fig, ax = plt.subplots(figsize=(12, 12))
    # plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    ax.set_title("Optimization profile (Energy)", fontsize=30)
    ax.plot(range(len(energies_log10)), energies_log10, marker='o', markersize=9, linewidth=3, label="Energy")  # тобы точки было видно
    ax.set_yscale("linear", fontsize=20)
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Value, $\\log_{10}\\left(\\frac{kJ}{mol}\\right)$", fontsize=20)
    ax.legend(prop={'size': 20})
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.savefig(str(plots_dir / f"{name}_plot_energy.png"), fmt="png")
    plt.clf()

    # RMSD plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Optimization profile (RMSD)", fontsize=30)
    # ax.plot(range(1, len(total_rmsd) + 1), total_rmsd, marker='s', markersize=9, linewidth=3, label="Total RMSD")
    ax.plot(range(1, len(rigid_rmsd) + 1), rigid_rmsd, marker='s', markersize=9, linewidth=3, label="Rigid RMSD")
    ax.plot(range(1, len(flexible_rmsd) + 1), flexible_rmsd, marker='s', markersize=9, linewidth=3, label="Flexible RMSD")
    ax.set_yscale("linear", fontsize=20)
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Value, $\AA$", fontsize=20)
    ax.legend(prop={'size': 20})
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.savefig(str(plots_dir / f"{name}_plot_rmsd.png"), fmt="png")
    plt.clf()


if __name__ == "__main__":
    print("Benchmark 02")

    print("Forcefield: " + forcefield_name)
    print("Source directory: " + str(source_dir))
    print("Output directory: " + str(output_dir))
    print("Creating output directory structure...")
    dirs = [output_dir, trajectories_dir, plots_dir, fluctuations_dir, sessions_dir]
    for d in dirs:
        if not os.path.exists(str(d)):
            os.mkdir(str(d))
    print("Scanning source directory...")
    for file_name in os.listdir(str(source_dir)):
        print("File name: " + file_name)
        benchmark(str(source_dir / file_name), file_name[:4])
    print("Saving session...")
    with open(str(sessions_dir / "benchmark_02.session"), "wb") as output_file:
        pickle.dump(session, output_file)
