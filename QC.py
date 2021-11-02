import os
import random
import numpy as np
from FlowCytometryTools import FCMeasurement
from file_params import *
from utils import *

good_markers = list()
bad_markers = list()
sizes = list()
os.chdir(FILES_PATH)

for file_BL, file_D28 in zip(os.listdir(BL_PATH), os.listdir(D28_PATH)):
    print(f"Baseline: {file_BL} day 28:{file_D28}")
    BL = FCMeasurement(ID="BL", datafile=f"BL/{file_BL}").data
    # BL = BL.data

    D28 = FCMeasurement(ID="D28", datafile=f"D28/{file_D28}").data
    # D28 = D28.data

    min_len = min(len(BL), len(D28))  # select the smallest dataset (try dataset.size ?)

    # Generate random indexes to select the data
    indexes = random.sample(range(0, min_len), min_len)

    # indexes = random.sample(range(0, min(len(BL), len(D28))), 20000)
    sizes.append(len(indexes))
    BL = BL.iloc[indexes, ]
    D28 = D28.iloc[indexes, ]

    columns_to_drop = ['Time', 'Event_length', 'Center', 'Offset', 'Width',
                       'Residual', 'File_Number', '102Pd', '103Rh', '104Pd',
                       '105Pd', '106Pd', '108Pd', '110Pd', '190BCKG',
                       '191Ir', '193Ir', '80ArAr', '131Xe_conta', '140Ce_Beads',
                       '208Pb_Conta', '127I_Conta', '138Ba_Conta']

    BL = BL.drop(columns_to_drop, axis=1)
    D28 = D28.drop(columns_to_drop, axis=1)

    for mbl, md28 in zip(BL, D28):
        print(f"- {mbl}")

        mbl_str, md28_str = str(mbl), str(md28)

        # TODO : Expliquer la condition en commentaire

        condition = (unimodal(BL[mbl_str]) == unimodal(BL[md28_str])) & (
                    spread(BL[mbl_str]) < 15 * spread(D28[md28_str])) & (
                                spread(D28[mbl_str]) < 15 * spread(BL[md28_str])) & (
                                BL[mbl_str].mean() < 15 * D28[md28_str].mean()) & (
                                D28[md28_str].mean() < 15 * BL[mbl_str].mean())

        if condition:
            if mbl in bad_markers:
                write_info(text="BAD")
            else:
                try:
                    good_markers.remove(mbl)
                except ValueError:
                    write_info(text="ValueError line 56", kind="[WARNING]")

                good_markers.append(mbl)
                write_info(text="GOOD")
        else:
            try:
                bad_markers.remove(mbl)
            except ValueError:
                write_info(text="ValueError line 64", kind="[WARNING]")
            try:
                good_markers.remove(mbl)
            except ValueError:
                write_info(text="ValueError line 68", kind="[WARNING]")
            bad_markers.append(mbl)
            write_info("GOOD")

with open(f"{RESULTS_PATH}/QC_results.txt", "w") as textfile:
    textfile.write("GOOD:\n")
    for element in np.unique(good_markers):
        textfile.write(f"{element}\n")

    textfile.write("\nBAD:\n")
    for element in np.unique(bad_markers):
        textfile.write(f"{element}\n")

    textfile.write("\nN CELLS:\n")
    textfile.write(f"{sizes}\n")
    textfile.write("PARAMETERS:\\means: 10 / spread: 10")  # CHANGE
