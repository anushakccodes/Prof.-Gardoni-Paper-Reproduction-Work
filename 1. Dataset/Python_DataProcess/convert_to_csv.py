import pandas as pd
from pathlib import Path

data_dir = Path(r"c:\Users\DELL\Desktop\Prof. Gardoni Paper Reproduction Work\1. Dataset\Python_DataProcess")

cols = [
    'B', 'D', 'H', 'B_num', 'D_num', 'H_num',
    'colWidth', 'beamDepth', 'beamRat', 'Asc', 'Asb',
    't', 'cover1', 'cover2', 'Ec', 'nu_c', 'fc',
    'fcuRat', 'eps_cu', 'Es', 'nu_s', 'fsy', 'eta',
    'Dy', 'Vy', 'Du', 'Vu', 'IDy', 'IVy', 'IDu', 'IVu'
]

for split, fname in [
    ('Train', 'RandomPushover_DL_Total_Train_new80.txt'),
    ('Test',  'RandomPushover_DL_Total_Test_new80.txt'),
]:
    src = data_dir / fname
    dst = data_dir / fname.replace('.txt', '.csv')
    df = pd.read_csv(src, sep='\t', header=None, names=cols)
    df.to_csv(dst, index=False)
    print(f"{split}: {len(df):,} rows -> {dst.name}")

print("Done.")
