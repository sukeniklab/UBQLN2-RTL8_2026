import sys
import MDAnalysis as mda
import subprocess
import os
import numpy as np

BASE = "/home/jkniblo/RTL8/data/CALVADOS3COM_2.0_MD_gpu_trial{t}_{c}/{c}/0"
OUTPUT_DIR = "foxs_profiles_new"
STRIDE     = 1
SELECTION  = "protein"
FOXS_QMAX  = 0.35


if len(sys.argv) != 3:
    sys.exit("Usage: run_foxs_trial.py <construct> <trial>")

construct = sys.argv[1]
trial     = int(sys.argv[2])
trial_tag = f"trial{trial}"


construct_dir = os.path.join(OUTPUT_DIR, construct)
trial_dir     = os.path.join(construct_dir, trial_tag)
os.makedirs(trial_dir, exist_ok=True)

base_dir = BASE.format(t=trial, c=construct)
dcd      = os.path.join(base_dir, f"{construct}.dcd")
topology = os.path.join(base_dir, f"{construct}.pdb")

if not os.path.exists(dcd) or not os.path.exists(topology):
    sys.exit(f"[{construct} | {trial_tag}] Missing files: {base_dir}")


def write_frame_pdb(atoms, dst):
    with open(dst, "w") as f:
        for i, atom in enumerate(atoms, start=1):
            x, y, z = atom.position
            resname  = atom.resname[:3].upper()
            resid    = atom.resid
            chain    = atom.segid[0] if atom.segid else "A"
            f.write(
                f"ATOM  {i:5d}  CA  {resname:3s} {chain}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")

u   = mda.Universe(topology, dcd)
sel = u.select_atoms(SELECTION)
n_sampled = len(u.trajectory[::STRIDE])

dat_files = []

for i, _ts in enumerate(u.trajectory[::STRIDE]):
    pdb = os.path.join(trial_dir, f"{construct}_{trial_tag}_{i:05d}.pdb")
    write_frame_pdb(sel, pdb)

    subprocess.run(
        ["foxs", "--residues", f"--max_q={FOXS_QMAX}", pdb],
        check=True, capture_output=True,
    )

    dat = pdb + ".dat"
    if os.path.exists(dat):
        dat_files.append(dat)

    if i % 500 == 0:
        print(f"[{construct} | {trial_tag}] {i}/{n_sampled} frames done", flush=True)

if not dat_files:
    sys.exit(f"[{construct} | {trial_tag}] No .dat files produced.")

all_I, q = [], None
for dat in dat_files:
    data = np.loadtxt(dat, comments="#")
    if q is None:
        q = data[:, 0]
    all_I.append(data[:, 1])

all_I  = np.array(all_I)
I_mean = all_I.mean(axis=0)
I_sem  = all_I.std(axis=0, ddof=1) / np.sqrt(len(all_I))

avg_dat = os.path.join(trial_dir, f"{construct}_{trial_tag}_avg.dat")
np.savetxt(avg_dat,
           np.column_stack([q, I_mean, I_sem]),
           header=f"q  I_mean  I_sem  ({len(dat_files)} frames, stride={STRIDE})",
           comments="# ")