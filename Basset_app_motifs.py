from __future__ import print_function
import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import itertools
import math
import glob

def get_filter_matrix(motif_path):
    os.chdir(motif_path)
    short_predefined_filters = dict()
    long_predefined_filters = dict()
    short_count = 0
    long_count = 0
    TFs = []
    for file in glob.glob("*_motif_df.csv"):
        TF = file[:file.index("_")]
        TFs.append(TF)
        matrix = pd.read_csv(file,sep="\t",header=0,index_col=0)
        matrix = matrix[["A","C","G","T"]]
        matrix = matrix.T
        array = matrix.values
        if matrix.shape[1] <= 15:
            empty_array = np.full((4,15),0.25)
            empty_array[:,:array.shape[1]] = array
            individual_filter = torch.FloatTensor(empty_array)
            short_predefined_filters[short_count] = individual_filter
            short_count += 1
        else:
            empty_array = np.full((4,25),0.25)
            empty_array[:,:array.shape[1]] = array
            individual_filter = torch.FloatTensor(empty_array)
            long_predefined_filters[long_count] = individual_filter
            long_count += 1

    return(TFs,short_predefined_filters,long_predefined_filters)

def return_selected_motifs():
    selected_motifs = ['AHR', 'ALX1', 'ARNT2', 'ARNT2', 'ARX', 'ASCL1', 'ATF1', 'ATF2', 'ATF3', 'ATF4', 'ATF7', 'ATOH1', 'BACH1', 'BACH2', 'BARX1', 'BARX2', 'BCL6', 'CDX1', 'CDX2', 'CEBPA', 'CEBPB', 'CEBPD', 'CEBPE', 'CEBPG', 'CEBPZ', 'CLOCK', 'CREB1', 'CREB3', 'CREM', 'CRX', 'CUX1', 'CUX2', 'DBP', 'DDIT3', 'DLX2', 'DLX3', 'DLX5', 'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5', 'E2F6', 'E2F7', 'E4F1', 'EGR3', 'EGR4', 'ELF1', 'ELF2', 'ELF3', 'ELK1', 'ELK3', 'ELK4', 'EMX1', 'EMX2', 'EOMES', 'EPAS1', 'ETS1', 'ETS2', 'ETV1', 'ETV3', 'ETV4', 'ETV5', 'FEV', 'FIGLA', 'FLI1', 'FOS', 'FOSB', 'FOSL1', 'FOSL2', 'FOXA1', 'FOXA2', 'FOXF2', 'FOXH1', 'FOXI1', 'FOXJ2', 'FOXK1', 'FOXM1', 'FOXO1', 'FOXO4', 'FOXO6', 'FOXP1', 'FOXP2', 'FOXP3', 'FOXQ1', 'GABPA', 'GATA2', 'GATA3', 'GATA4', 'GATA6', 'GCM1', 'GCM2', 'GFI1', 'GLI1', 'GLI2', 'GLI3', 'GLIS2', 'GMEB2', 'GRHL1', 'GRHL2', 'GSC', 'HAND1', 'HES5', 'HES7', 'HEY1', 'HIC1', 'HIF1A', 'HLF', 'HMGA1', 'HMX2', 'HNF1B', 'HNF4A', 'IKZF1', 'IRF7', 'IRF9', 'IRX3', 'ISL1', 'ISL2', 'JUN', 'JUNB', 'JUND', 'KLF12', 'KLF12', 'KLF13', 'KLF4', 'KLF5', 'LEF1', 'LHX2', 'LHX3', 'LHX4', 'LHX6', 'MAFB', 'MAFG', 'MAFK', 'MAX', 'MAZ', 'MEF2A', 'MEF2B', 'MEF2D', 'MEIS1', 'MEIS2', 'MEOX2', 'MITF', 'MIXL1', 'MLX', 'MSX2', 'MXI1', 'MYB', 'MYC', 'MYCN', 'MYF6', 'MYOG', 'MZF1', 'NFAT5', 'NFE2', 'NFIL3', 'NFKB1', 'NFKB2', 'NFYA', 'NFYB', 'NFYC', 'NOBOX', 'NOTO', 'NR0B1', 'NR1D1', 'NR1I2', 'NR2C1', 'NR2C2', 'NR2E3', 'NR4A1', 'NR4A2', 'NR4A3', 'NR5A2', 'NR6A1', 'NRL', 'OLIG1', 'OLIG2', 'OLIG3', 'OTX2', 'OVOL1', 'OVOL2', 'PATZ1', 'PAX3', 'PAX4', 'PAX6', 'PAX7', 'PAX8', 'PBX3', 'PDX1', 'PITX1', 'PITX2', 'PPARA', 'PPARD', 'PPARG', 'PRDM1', 'PROP1', 'PRRX1', 'PRRX2', 'PTF1A', 'RARB', 'RARG', 'RELB', 'RFX1', 'RFX2', 'RORA', 'RUNX1', 'RUNX2', 'RUNX3', 'RXRB', 'RXRG', 'SALL4', 'SCRT1', 'SIX1', 'SIX2', 'SMAD3', 'SNAI1', 'SNAI2', 'SOX13', 'SOX15', 'SOX17', 'SOX3', 'SOX4', 'SOX5', 'SOX9', 'SP1', 'SPDEF', 'SPIC', 'SRY', 'STAT3', 'STAT4', 'STAT6', 'TBR1', 'TBX20', 'TBX21', 'TBX3', 'TBX5', 'TEAD1', 'TEAD2', 'TEAD4', 'TEF', 'TFDP1', 'TGIF1', 'USF1', 'VDR', 'VSX1', 'VSX2', 'WT1', 'XBP1', 'ZEB1', 'ZFP42', 'ZIC1']
    return(selected_motifs)

def get_selected_filter_matrix(motif_path):
    selected_motifs = return_selected_motifs()
    os.chdir(motif_path)
    short_predefined_filters = dict()
    long_predefined_filters = dict()
    short_count = 0
    long_count = 0
    short_range_TFs = []
    long_range_TFs = []
    TFs = []
    for file in glob.glob("*_motif_df.csv"):
        TF = file[:file.index("_")]
        if TF in selected_motifs:
            TFs.append(TF)
            matrix = pd.read_csv(file,sep="\t",header=0,index_col=0)
            matrix = matrix[["A","C","G","T"]]
            matrix = matrix.T
            array = matrix.values
            if matrix.shape[1] <= 15:
                short_range_TFs.append(TF)
                empty_array = np.full((4,15),0.25)

                motif_width = array.shape[1]
                gap = (15-motif_width)//2

                empty_array[:,gap:(gap+array.shape[1])] = array
                individual_filter = torch.FloatTensor(empty_array)
                short_predefined_filters[short_count] = individual_filter
                short_count += 1
            else:
                long_range_TFs.append(TF)
                empty_array = np.full((4,25),0.25)

                motif_width = array.shape[1]
                gap = (25-motif_width)//2

                empty_array[:,gap:(gap+array.shape[1])] = array
                individual_filter = torch.FloatTensor(empty_array)
                long_predefined_filters[long_count] = individual_filter
                long_count += 1

    return(TFs,short_predefined_filters,long_predefined_filters)
