# 公開用（解析部は非公開）

トラジェクトリファイルとIDを連携させ任意に解析部で呼び出せるようにしています。
詳細についてはメール等でお問い合わせください。

## M3GNetとASEを用いた分子シミュレーション

**対象**: $La_{1.54}Sr_{0.46}Ga_3O_{7.27}$（多価陽イオン伝導物質の伝導経路解析）

計算セル準備段階：
- 非化学量論（不定比化合物）に基づいた置換・削除を行い、その後追跡IDを振り分けました。

---

# 1. 計算セル作成の保存

ID参照型に変更したため、自由度の高い解析が可能になりました。

## 【1章 ステップ1：CIF読み込みとラベリング】

```python
# =============================================================================
# [Refactor] Step 1: Loading CIF & Robust Site Labeling
# -----------------------------------------------------------------------------
# Changes:
# - Added identifying logic for specific oxygen sites (O1-O4) and cations.
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
```

## 【1章 ステップ2：インデックス特定と検証】

```python
# =============================================================================
# [Refactor] Step 2: Site Identification Verification
# -----------------------------------------------------------------------------
# Changes:
# - Replaced manual index calculation with `get_array('site_label')`.
# - Verifies that atom counts for each site match theoretical values (12x Unit Cell).
# =============================================================================

print("--- Verifying Site Counts in Supercell ---")

# Retrieve labels from the supercell
sc_labels = atoms_sc.get_array('site_label')
counts = Counter(sc_labels)

# Theoretical expectations for 2x2x3 supercell (x12)
expected = {
    'La_site': 48, # 4 * 12
    'Ga1': 24,     # 2 * 12
    'Ga2': 48,     # 4 * 12
    'O1': 24,      # 2 * 12
    'O2': 48,      # 4 * 12
    'O3': 96,      # 8 * 12
    'O4': 48       # 4 * 12
}

# Validation Loop
all_match = True
for label in sorted(expected.keys()):
    actual = counts.get(label, 0)
    theory = expected[label]
    status = "OK" if actual == theory else "MISMATCH"
    if actual != theory: all_match = False

    print(f"  {label.ljust(8)}: {actual} atoms (Expected: {theory}) -> {status}")

if all_match:
    print("\n>>> Step 2 Success: All sites are correctly labeled and tracked.")
else:
    print("\n>>> Step 2 Warning: Site counts do not match. Check CIF or Matrix.")

# Example verification of indices (checking first of each type)
print("\n[Debug] First index for each site:")
seen = set()
for i, label in enumerate(sc_labels):
    if label not in seen:
        print(f"  {label}: Index {i} ({atoms_sc[i].symbol})")
        seen.add(label)
```

## 【1章 ステップ3：LaをSrに置換する】

```python
# =============================================================================
# [Refactor] Step 3: La -> Sr Substitution (Target: La1.54 Sr0.46 ...)
# -----------------------------------------------------------------------------
# Purpose:
#   Introduce Sr dopants into La sites based on the chemical formula.
#   Updates both atomic symbols and tracking labels.
#
# Stoichiometry Logic (Why 11 atoms?):
#   - Target Formula: La_{1.54} Sr_{0.46} Ga_{3} O_{7.27}
#   - Total A-site cations (La + Sr) in formula = 1.54 + 0.46 = 2.00
#   - Sr doping ratio = 0.46 / 2.00 = 23%
#
#   - Current Supercell (2x2x3): Contains 12 Unit Cells.
#   - Total La_sites in Supercell = 4 sites/cell * 12 cells = 48 sites.
#   - Target Sr count = 48 sites * 0.23 = 11.04  -> Round to "11 atoms"
# =============================================================================

import random
import numpy as np

print("--- Step 3: La -> Sr Substitution ---")

# 1. Configuration (Based on logic above)
# -------------------------------------------------------
# Target: 11 Sr atoms (derived from La1.54 Sr0.46)
num_sr_target = 11

print(f"Target Composition: La(1.54) Sr(0.46) Ga(3) O(7.27)")
print(f"Supercell Size: 2x2x3 (12 Formula Units)")
print(f"Calculation: 48 La_sites * (0.46 / 2.00) = 11.04 -> Set Sr = {num_sr_target}")


# 2. Identify Candidate Sites
# -------------------------------------------------------
# Retrieve the label array (created in Step 1)
current_labels = atoms_sc.get_array('site_label')

# Find all indices currently labeled as 'La_site'
la_site_indices = [i for i, label in enumerate(current_labels) if label == 'La_site']

print(f"\nCandidate Sites found: {len(la_site_indices)} (Expected: 48)")


# 3. Perform Random Substitution
# -------------------------------------------------------
# Randomly select 11 indices for Sr
sr_indices = set(random.sample(la_site_indices, num_sr_target))

# Update Symbols AND Labels
count_sr = 0
count_la = 0

for i in la_site_indices:
    if i in sr_indices:
        # Change to Sr
        atoms_sc[i].symbol = 'Sr'
        current_labels[i] = 'Sr'        # Update tracking label
        count_sr += 1
    else:
        # Remain as La (update label from 'La_site' to 'La')
        current_labels[i] = 'La'
        count_la += 1

# Apply the updated labels back to the Atoms object
atoms_sc.set_array('site_label', current_labels)

print(f"Substitution Result:")
print(f"  [Sr] Doped  : {count_sr} atoms")
print(f"  [La] Remains: {count_la} atoms")
```

## 【1章 ステップ4（ランダム選別版）：O4の欠損導入と追跡データID構築】

統計結果の厳密性を担保するためにO4サイトの保持はランダムに行います。

```python
# =============================================================================
# [Refactor] Step 4: Pure Random O4 Selection & CIF Export
# -----------------------------------------------------------------------------
# Purpose:
#   1. Select 7 O4 atoms randomly (No bias).
#   2. Construct Tracking Dictionary.
#   3. Save structure in .traj (for calculation) AND .cif (for presentation/VESTA).
# =============================================================================

import random
import pickle
from ase import Atoms
import ase.io # CIF出力に必要
import numpy as np

print("--- Step 4: O4 Selection & Data Export ---")

# 1. Configuration
# -------------------------------------------------------
num_o4_target = 7
print(f"Configuration:")
print(f"  Target O4 Count : {num_o4_target}")
print(f"  Selection Mode  : Pure Random")

# 2. Identify Candidate Pool
# -------------------------------------------------------
current_labels = atoms_sc.get_array('site_label')
o4_indices_all = [i for i, label in enumerate(current_labels) if label == 'O4']

# 3. Pure Random Selection
# -------------------------------------------------------
selected_indices = random.sample(o4_indices_all, num_o4_target)
o4_indices_keep = set(selected_indices)
o4_indices_remove = set(o4_indices_all) - o4_indices_keep

print(f"\nSelection Result:")
print(f"  Kept {len(o4_indices_keep)} O4 atoms.")
print(f"  Removed {len(o4_indices_remove)} O4 atoms.")


# 4. Reconstruct Structure & Tracking Dictionary
# -------------------------------------------------------
atoms_final = Atoms(cell=atoms_sc.get_cell(), pbc=True)
tracking_data = {}
new_index_counter = 0
final_site_labels = []

for i, atom in enumerate(atoms_sc):
    if i in o4_indices_remove:
        continue

    # Add atom and preserve label
    atoms_final.append(atom)
    label = current_labels[i]
    final_site_labels.append(label)

    # Register to dictionary
    if label not in tracking_data:
        tracking_data[label] = []
    tracking_data[label].append(new_index_counter)

    new_index_counter += 1

# Embed labels (important for some CIF writers)
atoms_final.new_array('site_label', np.array(final_site_labels))


# 5. Verification & Save (Modified)
# -------------------------------------------------------
print("\n--- Final Output ---")
print(f"Total Atoms: {len(atoms_final)} (Expected: 295)")
print(f"Formula    : {atoms_final.get_chemical_formula()}")

# Save as TRAJ (for ASE/Python restart)
outfile_traj = "structure_final.traj"
atoms_final.write(outfile_traj)
print(f"[Save] ASE Trajectory : '{outfile_traj}' (For Calculation)")

# Save as CIF (for VESTA/Presentation)
outfile_cif = "structure_final.cif"
atoms_final.write(outfile_cif)
print(f"[Save] CIF File       : '{outfile_cif}' (For VESTA/Presentation)")

# Save Tracking Data
outfile_pkl = "tracking_data.pkl"
with open(outfile_pkl, "wb") as f:
    pickle.dump(tracking_data, f)
print(f"[Save] Tracking Data  : '{outfile_pkl}' (For Analysis)")

print("\n>>> Step 4 Success: All files generated.")
```

---

# 2. 構造緩和とMD計算（概要）

操作性を重視して1つにまとめています。タイムステップは **2.0 fs** に設定しています。

（※解析部は非公開）
