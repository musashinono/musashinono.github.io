# musashinono.github.io
# =============================================================================
# [Refactor] Step 1: Loading CIF & Robust Site Labeling
# -----------------------------------------------------------------------------
# Changes:
# - Added identifying logic for specific oxygen sites (1-O4) and cations.
# - Implemented `site_label` array to persist atom tags during supercell expansion.
# - This enables tracking of "Knock-off" exchange mechanisms in later analysis.
# =============================================================================

import ase.io
from ase.build import make_supercell
import numpy as np
from collections import Counter

# 1. Load Unit Cell
# -------------------------------------------------------
cif_file = "./La-Sr-Ga-O_STD_429463.cif"
atoms_unit = ase.io.read(cif_file)

print("--- Unit Cell Loaded ---")
print(f"Formula: {atoms_unit.get_chemical_formula()}")
print(f"Count  : {len(atoms_unit)} atoms")

# 2. Assign Site Labels (Unit Cell Level)
# -------------------------------------------------------
# cifは1から始まるがASEの読み取りは0から始まる！！
# Defined based on Melilite structure (P-421m) indices in standard CIF:
# La/Sr(4e):0-3, Ga1(2a):4-5, Ga2(4e):6-9
# O1(2c):10-11, O2(4e):12-15, O3(8f):16-23, O4(4e):24-27

site_labels = []

for i, atom in enumerate(atoms_unit):
    if i <= 3:   label = 'La_site' # Will be substituted by Sr later
    elif i <= 5: label = 'Ga1'
    elif i <= 9: label = 'Ga2'
    elif i <= 11: label = 'O1'
    elif i <= 15: label = 'O2'
    elif i <= 23: label = 'O3'
    elif i <= 27: label = 'O4'     # Target interstitial oxygen
    else: label = 'Unknown'
    site_labels.append(label)

# Embed labels into Atoms object (persists through supercell creation)
atoms_unit.new_array('site_label', np.array(site_labels))
print(f"Labels assigned: {set(site_labels)}")

# 3. Create Supercell (2x2x3)
# -------------------------------------------------------
supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
atoms_sc = make_supercell(atoms_unit, supercell_matrix)

print("\n--- Supercell Created ---")
print(f"Total Atoms: {len(atoms_sc)} (28 * 12 = 336)")
print("Cell Parameters:\n", atoms_sc.cell.cellpar())
