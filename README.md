# Basset_attention

For Harvard STAT316: Deep learning and regulatory genomics course.  
This repository aims to predict chromain accessibility from seuquence data. Different from original Basset model, it employes self-attention mechanism from Transformer.

# Model structure
![model stucture](https://github.com/hyalin1127/Basset_attention/blob/master/Basset_attention_model.png)

# Scripts

This repository contains the following two functions:
```
* Generate synthetic data
* Train a deep learning model with self attention mechanism for chromain accessibility prediction
```

# Generate synthetic data #
Executing the following command will generate one train file and one test file: 
```
python synthetic_data.py
```
The out put will be:
```
Basset_syn_train_posi_bin_matrix.hdf5   
Basset_syn_test_posi_bin_matrix.hdf5 
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
python Basset_real_main_v7.py -t Basset_syn_train_posi_bin_matrix.hdf5 -s Basset_syn_test_posi_bin_matrix.hdf5 -m ./motif_folder
```

Performance:
![Precision recall](https://github.com/hyalin1127/Basset_attention/blob/master/Precision_recall.png)
