#!/usr/bin/env python

import uproot
import numpy as np
import vector
import subprocess
import os
import csv

files = {
    "mc20_13TeV_MC_PowhegPythia8EvtGen_NNPDF3_AZNLO_ggZH125_vvbb.root": "https://opendata.cern.ch/record/80012/files/mc20_13TeV_MC_PowhegPythia8EvtGen_NNPDF3_AZNLO_ggZH125_vvbb_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_0.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_1.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_1",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_2.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_2",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_3.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_3",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_4.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_4",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_5.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_5",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_6.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_6",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_0.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_file_index.json_0",
    # "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW.root": "https://opendata.cern.ch/record/80014/files/mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_file_index.json_0",
    "data_1.root": "https://opendata.cern.ch/record/80001/files/data16_13TeV_Run_00296939_file_index.json_7",
    "data_2.root": "https://opendata.cern.ch/record/80001/files/data16_13TeV_Run_00296939_file_index.json_8",
    "data_3.root": "https://opendata.cern.ch/record/80001/files/data16_13TeV_Run_00296939_file_index.json_9",
    "data_4.root": "https://opendata.cern.ch/record/80001/files/data16_13TeV_Run_00296939_file_index.json_10",
}


# # https://opendata.atlas.cern/docs/data/for_research/metadata
# # https://opendata.atlas.cern/files/metadata.csv
# def get_weights(file):
#     file = file.replace("mc20_13TeV_MC_", "").replace(".root", "")
#     with open("metadata.csv", mode="r") as csv_file:
#         reader = csv.reader(csv_file)
#         for row in reader:
#             if file in str(row[1]):
#                 print(row)
#                 w = float(row[3]) * float(row[3]) / float(row[7])
#     return w


def download_files():
    """
    Downloads required ROOT files using wget and saves them with specified names.
    Suppresses wget output.
    """
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            with open(os.devnull, "w") as devnull:  # Suppress output
                subprocess.run(
                    ["wget", "-O", filename, url],
                    stdout=devnull,
                    stderr=devnull,
                    check=True,
                )
        else:
            print(f"{filename} already exists, skipping download.")


def make_ntuple(input_file, output_file, weight, unc=1.0):
    f = uproot.open(input_file)
    btag = f["CollectionTree"]["BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pb"].array(
        library="np"
    )
    jets_pt = f["CollectionTree"]["AnalysisJetsAuxDyn.pt"].array(library="np")
    jets_eta = f["CollectionTree"]["AnalysisJetsAuxDyn.eta"].array(library="np")
    jets_phi = f["CollectionTree"]["AnalysisJetsAuxDyn.phi"].array(library="np")
    jets_m = f["CollectionTree"]["AnalysisJetsAuxDyn.m"].array(library="np")

    vars = [
        "j1_pt",
        "j1_eta",
        "j1_phi",
        "j1_m",
        "j2_pt",
        "j2_eta",
        "j2_phi",
        "j2_m",
        "h_pt",
        "h_eta",
        "h_phi",
        "h_m",
        "weight",
        "bool_btag_1",
        "bool_btag_2",
        "my_sf_unc_up",
        "my_sf_unc_down",
    ]

    # Initialize ntup with keys for each systematic variation
    ntup = {}
    for v in vars:
        ntup[v] = []
    for event in range(len(jets_pt)):
        # no jet trigger in opendata...
        jet_indices = np.arange(len(jets_pt[event]))

        if len(jet_indices) < 2:
            continue
        # Select jets with b-tags above the threshold
        btag_indices = np.where(btag[event] > 0.85)[0]

        # # Skip events with no b-tagged jets to keep test files small
        if len(btag_indices) < 1:
            continue

        btag_1 = len(btag_indices) == 1  # Exactly one b-tag
        btag_2 = len(btag_indices) >= 2  # Two or more b-tags

        # Assign the first b-tagged jet
        j1_idx = 0
        j1 = vector.obj(
            pt=jets_pt[event][j1_idx] * unc,
            eta=jets_eta[event][j1_idx],
            phi=jets_phi[event][j1_idx],
            mass=jets_m[event][j1_idx],
        )

        # if btag_1:

        j2_idx = 1
        j2 = vector.obj(
            pt=jets_pt[event][j2_idx] * unc,
            eta=jets_eta[event][j2_idx],
            phi=jets_phi[event][j2_idx],
            mass=jets_m[event][j2_idx],
        )
        h = j1 + j2

        # Fill ntuple
        ntup["j1_pt"].append(j1.pt)
        ntup["j1_eta"].append(j1.eta)
        ntup["j1_phi"].append(j1.phi)
        ntup["j1_m"].append(j1.mass)

        ntup["j2_pt"].append(j2.pt)
        ntup["j2_eta"].append(j2.eta)
        ntup["j2_phi"].append(j2.phi)
        ntup["j2_m"].append(j2.mass)

        ntup["h_pt"].append(h.pt)
        ntup["h_eta"].append(h.eta)
        ntup["h_phi"].append(h.phi)
        ntup["h_m"].append(h.mass)

        # Append orthogonal btag flags
        ntup["bool_btag_1"].append(1 if btag_1 else 0)
        ntup["bool_btag_2"].append(1 if btag_2 else 0)

        # Append weights
        ntup["weight"].append(weight)
        ntup["my_sf_unc_up"].append(1.2)
        ntup["my_sf_unc_down"].append(0.8)

    with uproot.recreate(output_file) as root_file:
        root_file["FilteredTree"] = {key: np.array(val) for key, val in ntup.items()}
    print(f"Output written to {output_file}")


def merge_jetjet_files(output_files, merged_file):
    merged_data = {}
    for output_file in output_files:
        with uproot.open(output_file) as f:
            tree = f["FilteredTree"]
            for branch in tree.keys():
                if branch not in merged_data:
                    merged_data[branch] = tree[branch].array(library="np")
                else:
                    merged_data[branch] = np.concatenate(
                        [merged_data[branch], tree[branch].array(library="np")]
                    )

    with uproot.recreate(merged_file) as root_file:
        root_file["FilteredTree"] = merged_data
    print(f"Merged file written to {merged_file}")


if __name__ == "__main__":
    # turn on if you really want to do this
    # download_files()

    output_files = []
    for f_name in files:
        if "ggZH125" in f_name:
            make_ntuple(f_name, "/Users/fred/dev/tomatos/tests/files/ggZH125_vvbb/NOSYS.root", weight=1e-2, unc=1.0)
            make_ntuple(
                f_name, "/Users/fred/dev/tomatos/tests/files/ggZH125_vvbb/JET_PT_1UP.root", weight=1e-2, unc=1.1
            )
            make_ntuple(
                f_name, "/Users/fred/dev/tomatos/tests/files/ggZH125_vvbb/JET_PT_1DOWN.root", weight=1e-2, unc=0.9
            )
        else:
            output_file = f"jetjet_{f_name.split('_')[-1]}"
            make_ntuple(f_name, output_file, weight=1)
            output_files.append(output_file)

    merge_jetjet_files(output_files, "/Users/fred/dev/tomatos/tests/files/bkg/NOSYS.root")
