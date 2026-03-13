
import os

from src.utils.parse_resonance_files import process_file_resonance
from scripts.step1_calculate_fbcsp import calculate_csp_in_bands

DATA_FOLDER =  r".\data\test\03_06 Artem"
MODES = ["left-right",  'right-rest', 'left-rest']

if __name__ == "__main__":
    for record in os.listdir(DATA_FOLDER):
        if record in ["01_222sec_112.hdf", "02_224sec_111.hdf"]:
            continue
        print(f"===== {record} =====")
        eeg, idxs_rest, idxs_right, idxs_left, xy, Fs = process_file_resonance(os.path.join(DATA_FOLDER, record))

        for mode in MODES:
            print(f"--- {mode} ")
            if mode == "left-right":
                idxs1 = idxs_left 
                idxs2 = idxs_right
            elif mode == "right-rest":
                idxs1 = idxs_right
                idxs2 = idxs_rest
            elif mode == "left-rest":
                idxs1 = idxs_left
                idxs2 = idxs_rest

            # ==== universal part ==== 
            calculate_csp_in_bands(eeg, Fs, idxs1, idxs2, xy, edges_ms=250, bands=[[8, 10], [10, 12], [12, 14], [8, 12], [10, 14]], spectr=True,
                                folder_output=os.path.join(r"./results/csp_components/03_03 Artem", record[:-4], mode))