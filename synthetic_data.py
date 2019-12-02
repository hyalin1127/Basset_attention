from __future__ import print_function
import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle
from random import choices
import random
from collections import defaultdict
import h5py

def paired_distances_check(values):
    list_1 = values[0]
    list_2 = values[1]
    distances_list = []
    for i in list_1:
        for j in list_2:
            distances_list.append(abs(i-j))
    return(np.min(distances_list))

def reverse_motif(motif):
    motif = motif[::-1]
    new_motif = []
    for i in motif:
        if i == "A":
            new_motif.append("T")
        if i == "T":
            new_motif.append("A")
        if i == "C":
            new_motif.append("G")
        if i == "G":
            new_motif.append("C")
    return(new_motif)

def main(nmotif,motif,TFcomb,seqlen,nseq,type):
    """
    generate nseq sequences that are random with inserted motif combinations
    each cell has its own combination of motifs (cellcomb)
    """
    output = open("STAT316_synthetic_%s_data.csv" %(type),'wt')

    seqlookup = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1]}
    maxmotiflen=25

    R = random.Random()
    for i in range(nseq):
        seq  = R.choices( ['A','C','G','T'], k=seqlen)
        k    = R.sample(range(len(TFcomb)),1)[0]
        TF   = TFcomb[k]
        pos  = R.sample( range(seqlen-maxmotiflen), nmotif )
        for elem in pos:
            j = R.sample( range(len(TF)), 1 )[0]
            seq[elem:(elem+len(motif[TF[j]]))] = motif[TF[j]]

        cellcode = len(TFcomb)*[0]
        cellcode[k] = 1
        seqcode = np.zeros((4,seqlen))

        for i in range(len(seq)):
            seqcode[:,i] = seqlookup[seq[i]]

        seqcode = ((seqcode.reshape(1,seqlen*4)).tolist())[0]
        output.write("%s\n" %(','.join(['%d' % elem for elem in (cellcode+seqcode)])))

    return

def main_with_positional_effect_hdf5(nmotif,motif,TFcomb,seqlen,nseq,type):
    """
    generate nseq sequences that are random with inserted motif combinations
    each cell has its own combination of motifs (cellcomb)
    """

    #with h5py.File("/n/scratchlfs/xiaoleliu_lab/cchen/Cistrome_imputation/basset_model/data/synthetic_data/Basset_%s_bin_matrix.hdf5" %(type),'w') as output:
    with h5py.File("Basset_%s_bin_matrix.hdf5" %(type),'w') as output:

        DNase_hdf5 = output.create_dataset("DNase", dtype=np.float32, shape=(nseq,len(TFcomb)*2),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)
        seq_hdf5 = output.create_dataset("sequence", dtype=np.float32, shape=(nseq,4,1024),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)

        seqlookup = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1] }
        maxmotiflen = 25

        R = random.Random()
        for i in range(nseq):
            seq  = R.choices( ['A','C','G','T'], k=seqlen)
            k    = R.sample(range(len(TFcomb)),1)[0]
            TF   = TFcomb[k]
            pos  = R.sample( range(seqlen-maxmotiflen), nmotif )

            TF_position_record = defaultdict(list)
            for elem in pos:
                j = R.sample( range(len(TF)), 1 )[0]
                if random.uniform(0, 1)>0.5:
                    seq[elem:(elem+len(motif[TF[j]]))] = motif[TF[j]]
                else:
                    seq[elem:(elem+len(motif[TF[j]]))] = reverse_motif(motif[TF[j]])
                TF_position_record[TF[j]].append(elem)

            cellcode = len(TFcomb)*2*[0]
            if len(list(TF_position_record.keys()))==2:
                minimal_distance = paired_distances_check(list(TF_position_record.values()))
                if k in [0,1,2]:
                    if minimal_distance < (seqlen*9//10):
                        if random.uniform(0, 1)>0.0:
                            cellcode[2*k] = 1
                        if random.uniform(0, 1)>0.0:
                            cellcode[2*k+1] = 1
                if k in [3,4,5]:
                    if minimal_distance < (seqlen*2//5):
                        if random.uniform(0, 1)>0.0:
                            cellcode[2*k] = 1
                        if random.uniform(0, 1)>0.0:
                            cellcode[2*k+1] = 1
                if k in [6,7]:
                    if random.uniform(0, 1)>0.0:
                        cellcode[2*k] = 1
                    if random.uniform(0, 1)>0.0:
                        cellcode[2*k+1] = 1

            seqcode = np.zeros((4,seqlen))

            for g in range(len(seq)):
                seqcode[:,g] = seqlookup[seq[g]]

            DNase_hdf5[i,:] = cellcode
            seq_hdf5[i,:,:] = seqcode

        return

if __name__ == '__main__':

    # definition of TF motifs
    motif = {"FOXA1":"TGTTTACTTTGG", "ESR1":"AGGTCACCCTGACCT","KLF5":"AGGGTGGGGCTGGG","GRHL2":"AACCTGTTTGAC","OTX2":"AGGGGATTAAC","E2F1":"GGGGGGCGGGAAG","MYB":"GGTGGCAGTTGG"}

    # combinations of TFs in different cell types
    TFcomb   = [ ('FOXA1','ESR1'), ('FOXA1','KLF5'), ('ESR1','KLF5'), ('ESR1','GRHL2'), ('KLF5','GRHL2'), ('KLF5','OTX2'),('E2F1','E2F1'),('MYB','MYB') ]
    # total number of sequences to generate
    #nseq     = 2000
    # length of each sequence
    seqlen   = 1024
    # number of motif instances to insert into each sequence
    nmotif   = 10

    main_with_positional_effect_hdf5(nmotif,motif,TFcomb,seqlen,nseq=2000,type = "syn_train_posi")
    main_with_positional_effect_hdf5(nmotif,motif,TFcomb,seqlen,nseq=200,type = "syn_test_posi")
