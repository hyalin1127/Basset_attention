# Basset_attention

This repository is for Harvard STAT316: Deep learning and regulatory genomics course.


Predicting chromatin accessibility from sequence data; attention mechanism
====================================================================================
This repository contains the following two functions:
```
* Generate synthetic data
* Train a deep learning model with self attention mechanism for chromain accessibility prediction
```

# Generate synthetic data #
Executing the following command will generate one train file and one test file: 
```
python synthetic_data_stat316.py
```
The out put will be:
```
STAT316_synthetic_train_posi_data.csv  
STAT316_synthetic_test_posi_data.csv  
```
Ex: 
```
sgRNA         gene     sample1_readcount     sample2_readcount...
gene1_gRNA_1  gene1    557                   421
gene1_gRNA_2  gene1    295                   128
     .          .       .
     .          .       .
     .          .       .
gene2_gRNA_1  gene2    173                   68
gene2_gRNA_2  gene2    85                    38
     .          .       .
     .          .       .
     .          .       .
```
# Installation #
