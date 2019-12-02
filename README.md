# Basset_attention

For Harvard STAT316: Deep learning and regulatory genomics course.  
This repository aims to predict chromain accessibility from seuquence data. Different from original Basset model, it employes self-attention mechanism from Transformer.

# Model structure


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
In each file, each line contains 6 + 1024\*4 binary numbers.  
The first six: Chromain accessible or not (binary) in 6 cell lines.  
The last 1024\*4: The one-hot encoding of sequence (A,T,C,G) in surrounding 1,024 base pairs.

# Train model #
Executing the following command will perform model training and test data prediction.
```
Usage: Basset_real_main_v7.py -i train_file_name -t test_file_name -m motif_path

Deep learning with self-attention.

Options:
  --version             show program's version number and exit
  -h, --help            Show this help message and exit.
  -m MOTIF_PATH, --motif_path=MOTIF_PATH
                        Motif path
  -t TRAIN_DATA, --train_data=TRAIN_DATA
                        Train data name
  -s TEST_DATA, --test_data=TEST_DATA
                        Test data name
```
Example: 
```
python Basset_STAT316_rpos_multihead.py -t Basset_syn_train_posi_bin_matrix.hdf5 -s Basset_syn_test_posi_bin_matrix.hdf5
```
