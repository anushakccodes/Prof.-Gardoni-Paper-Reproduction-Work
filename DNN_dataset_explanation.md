# DNN Dataset Explanation

**Paper:** *Deep learning-based surrogate capacity models and multi-objective fragility estimates for reinforced concrete frames*  
**Authors:** Lili Xing, Paolo Gardoni, Ge Song, Ying Zhou  
**Journal:** Computer Methods in Applied Mechanics and Engineering

---

## 1. What Each Dataset File Contains

### 1.1 `1. Dataset/DNN_dataset/RandomPushResults[1–50]/`

There are **50 folders**, each named `RandomPushResults1` through `RandomPushResults50`.  
Each folder contains three metadata files and roughly 2,000 individual pushover-curve files.

| File | Content |
|------|---------|
| `PushOver_Random_availableS.txt` | Two-column table mapping sequential row numbers to scenario IDs |
| `PushOver_Random_InputVariablesR.txt` | 23-column matrix of sampled structural/material input features |
| `PushOver_Random_CapacityPointsR.txt` | 10-column matrix of computed capacity-point outputs |
| `PushOver_Random_S{id}.txt` | Individual pushover curve for scenario `{id}` |

### 1.2 `1. Dataset/Python_DataProcess/`

| File | Content |
|------|---------|
| `RandomPushover_DL_Total_Train_new80.txt` | Pre-processed training split — 105,116 rows × 31 columns |
| `RandomPushover_DL_Total_Test_new80.txt`  | Pre-processed test split — 5,533 rows × 31 columns |

These files compile all valid raw scenarios, apply unit normalisation, and provide the exact dataset used by the paper authors for DNN training.

---

## 2. How the Files Are Structured

### 2.1 `PushOver_Random_availableS.txt`

Plain text, whitespace-separated, two columns, no header:

```
1    3
2    5
3    6
...
```

- Column 1: sequential row number (1-indexed position within the folder's arrays)
- Column 2: scenario ID number from the original Latin hypercube sampling run

Some IDs are missing because scenarios whose finite-element analysis failed to converge were discarded during the simulation phase (not in the post-processing step).

### 2.2 `PushOver_Random_InputVariablesR.txt`

Plain text, whitespace-separated, **23 columns**, no header.  
Each row corresponds to one scenario (same row order as `availableS`).  
Values are the raw sampled input parameter values (not normalized).

### 2.3 `PushOver_Random_CapacityPointsR.txt`

Plain text, whitespace-separated, **10 columns**, no header.  
Each row corresponds to one scenario (same row order as `availableS`).

| Column | Symbol | Unit | Description |
|--------|--------|------|-------------|
| 0 | Dy | mm | Top displacement at yield point |
| 1 | Vy | kN | Base shear at yield — from top-displacement curve |
| 2 | Du | mm | Top displacement at peak point |
| 3 | Vu | kN | Peak base shear — from top-displacement curve (= col 9) |
| 4 | IDy_abs | mm | Max inter-story drift at yield (absolute, not ratio) |
| 5 | IDu_abs | mm | Max inter-story drift at peak (absolute, not ratio) |
| 6 | IDy | fraction | Max inter-story drift ratio at yield (dimensionless) |
| 7 | IVy | kN | Base shear at yield — from inter-story drift curve |
| 8 | IDu | fraction | Max inter-story drift ratio at peak (dimensionless) |
| 9 | IVu | kN | Peak base shear — from inter-story drift curve (= col 3) |

**Key observation:** Columns 3 and 9 are always identical (`Vu == IVu`). This is because the peak base shear is a structural property — it does not depend on which axis (top displacement or inter-story drift) the pushover curve is expressed in.

### 2.4 `PushOver_Random_S{id}.txt`

Plain text, whitespace-separated, **4 columns**, no header.  
Each row is one analysis step along the pushover loading path:

| Column | Description | Unit |
|--------|-------------|------|
| 0 | Step index (1-indexed) | — |
| 1 | Top displacement | mm |
| 2 | Base shear | kN |
| 3 | Maximum inter-story drift | fraction |

These files contain the full force-deformation history from which the capacity points (yield and peak) are extracted.

### 2.5 Pre-Processed Files (`RandomPushover_DL_Total_*.txt`)

Plain text, whitespace-separated, **31 columns**, no header.  
Columns 0–22 are the 23 input features (same units as the raw InputVariables files).  
Columns 23–30 are the 8 DNN output targets, with units converted from the raw CapacityPoints files:

| Col | Symbol | Unit | Conversion from raw |
|-----|--------|------|---------------------|
| 23 | Dy | m | raw col 0 (mm) ÷ 1000 |
| 24 | Vy | kN | raw col 1 — as-is |
| 25 | Du | m | raw col 2 (mm) ÷ 1000 |
| 26 | Vu | kN | raw col 3 — as-is |
| 27 | IDy | % | raw col 6 (fraction) × 100 |
| 28 | IVy | kN | raw col 7 — as-is |
| 29 | IDu | % | raw col 8 (fraction) × 100 |
| 30 | IVu | kN | raw col 9 — as-is (= col 26) |

---

## 3. How the Features and Targets Were Identified

### 3.1 Input Features

Table 1 of the paper lists 29 features total, of which **23 are independently sampled variables** and 6 are derived deterministically from those 23 (e.g., `eps_c0 = 2 * fc / Ec`, `fct = 0.1 * fc`). Only the 23 sampled variables appear in the input files.

Each column in `PushOver_Random_InputVariablesR.txt` was matched to a paper variable by comparing:
- **Range:** The observed min/max of each column against the paper's stated distribution bounds.
- **Mean:** The observed mean against the paper's stated mean value.
- **Units:** Physical plausibility (e.g., column 0 range [3.6, 10] uniquely matches B (m)).

All 23 columns were confirmed without ambiguity.

### 3.2 Target Variables

The paper's section 3.2 states:

> "two capacity points (yielding and peak points) are computed as output `y = [Dy, Vy, Du, Vu]` from the shear force–top displacement curve or `y = [IDy, IVy, IDu, IVu]` from the shear force–maximum inter-story drift curve."

The 31 columns in the pre-processed files (23 inputs + 8 outputs) match this exactly.

The 10 columns in the raw CapacityPoints file were decoded by:
1. Checking that `col3 == col9` always → these are `Vu` and `IVu` (peak base shear, identical).
2. Checking that `col0` (range ~[-27, 219] mm) matches displacements in mm → `Dy`.
3. Checking that `col6` (range [0, 0.04]) matches a dimensionless drift fraction → `IDy`.
4. Confirming unit conversions by comparing column statistics between the raw and pre-processed files for the overlapping populations.
5. Noting that `IDu` (col 29) has a hard cap at 2.4 % in the pre-processed file, exactly matching the paper's stated limit: *"the pushover analysis was able to proceed until the maximum inter-story drift reached the prescribed limit of 2.4%."*

---

## 4. How the Final DataFrame Was Constructed

The notebook `3. Codes/EDA.ipynb` builds the final DataFrame in two ways:

### 4.1 From Pre-Processed Files (Recommended)

```python
train_arr = np.loadtxt("RandomPushover_DL_Total_Train_new80.txt")
test_arr  = np.loadtxt("RandomPushover_DL_Total_Test_new80.txt")

df_train = pd.DataFrame(train_arr, columns=ALL_COLS)  # ALL_COLS = 23 inputs + 8 outputs
df_test  = pd.DataFrame(test_arr,  columns=ALL_COLS)
df       = pd.concat([df_train, df_test], ignore_index=True)
```

A `split` column (`"train"` / `"test"`) is added to distinguish the two splits.

### 4.2 From Raw Folder Data (for Inspection / Custom Preprocessing)

Each of the 50 folders is loaded, stacked, and unit-converted:

```python
for folder in folders:
    iv_list.append(np.loadtxt(f"{folder}/PushOver_Random_InputVariablesR.txt"))
    cp_list.append(np.loadtxt(f"{folder}/PushOver_Random_CapacityPointsR.txt"))

iv_all = np.vstack(iv_list)      # shape: (100100, 23)
cp_all = np.vstack(cp_list)      # shape: (100100, 10)

df_raw_outputs = pd.DataFrame({
    "Dy" : cp_all[:, 0] / 1000,   # mm -> m
    "Vy" : cp_all[:, 1],
    "Du" : cp_all[:, 2] / 1000,   # mm -> m
    "Vu" : cp_all[:, 3],
    "IDy": cp_all[:, 6] * 100,    # fraction -> %
    "IVy": cp_all[:, 7],
    "IDu": cp_all[:, 8] * 100,    # fraction -> %
    "IVu": cp_all[:, 9],
})
```

This raw DataFrame (100,100 rows) includes scenarios with convergence failures (negative Dy/Du values). These were excluded by the authors when creating the pre-processed files, which is why the pre-processed set has a larger effective sample count of 110,649 (including re-runs from additional sampling batches).

---

## 5. Assumptions Made While Interpreting the Dataset

| # | Assumption | Evidence / Justification |
|---|-----------|--------------------------|
| 1 | Column order in `InputVariablesR.txt` matches Table 1 row order exactly | All 23 columns verified by range, mean, and unit checks against Table 1 |
| 2 | `CapacityPoints` col 3 == col 9 (Vu == IVu) | Verified numerically: `np.allclose(cp[:, 3], cp[:, 9])` is True for all folders |
| 3 | `CapacityPoints` cols 4 and 5 are raw absolute inter-story drift values (mm), not used directly as DNN outputs | Their magnitude (0–2705 mm) is inconsistent with the 8 named DNN outputs; cols 6/8 (fraction) are the drift-ratio equivalents |
| 4 | Pre-processed file outputs are in meters (Dy, Du) and percent (IDy, IDu) | Confirmed by: (a) IDu max = 2.3999% ≈ paper's 2.4% limit; (b) statistical ranges are physically consistent with tall RC frame buildings |
| 5 | Rows in `InputVariablesR` and `CapacityPointsR` are aligned (same row = same scenario) | Both files always have the same row count as `availableS` (2,002 per folder) |
| 6 | The `availableS` file is informational only — it is not needed to index into the other two files | The InputVariables and CapacityPoints files already contain only the successful scenarios in sequential order |
| 7 | The pre-processed files include scenarios from more than 50 × 2,002 = 100,100 raw scenarios | The paper states the dataset was built iteratively in batches of 2,000 until R² > 0.9, reaching 80,000 pairs; the 50 folders likely represent 40 batches × 2,002 + re-runs |

---

## 6. Column Name Reference

| Column | Paper Symbol | Description | Unit |
|--------|-------------|-------------|------|
| 0 | B | Breadth of one span | m |
| 1 | D | Depth of one span | m |
| 2 | H | Story height | m |
| 3 | B_num | Number of spans (X) | — |
| 4 | D_num | Number of spans (Y) | — |
| 5 | H_num | Number of storeys | — |
| 6 | colWidth | Column width (square) | m |
| 7 | beamDepth | Beam depth | m |
| 8 | beamRat | Beam depth-to-width ratio | — |
| 9 | Asc | Steel bar area in column | mm² |
| 10 | Asb | Steel bar area in beam | mm² |
| 11 | t | Slab thickness | m |
| 12 | cover1 | Concrete cover – column | m |
| 13 | cover2 | Concrete cover – beam | m |
| 14 | Ec | Concrete elastic tangent | ×10¹⁰ Pa |
| 15 | nu_c | Concrete Poisson ratio | — |
| 16 | fc | Concrete compressive strength | ×10⁶ Pa |
| 17 | fcuRat | Ratio fcu / fc | — |
| 18 | eps_cu | Concrete ultimate strain (offset) | — |
| 19 | Es | Steel elastic tangent | ×10¹⁰ Pa |
| 20 | nu_s | Steel Poisson ratio | — |
| 21 | fsy | Steel yield strength | ×10⁶ Pa |
| 22 | eta | Strain-hardening ratio | — |
| 23 | Dy | Top displacement at yield | m |
| 24 | Vy | Base shear at yield | kN |
| 25 | Du | Top displacement at peak | m |
| 26 | Vu | Peak base shear | kN |
| 27 | IDy | Max inter-story drift at yield | % |
| 28 | IVy | Base shear at yield (ISD curve) | kN |
| 29 | IDu | Max inter-story drift at peak | % |
| 30 | IVu | Peak base shear (ISD curve) = Vu | kN |
