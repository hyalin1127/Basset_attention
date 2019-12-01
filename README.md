# Basset_attention

For Harvard STAT316: Deep learning and regulatory genomics course.  
Thsi repository aims to predict chromain accessibility from seuquence data. Different from original Basset model, it employes self-attention mechanism from Transformer.

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
In each file, each line contains 6 + 1024*4 binary numbers



# Installation #
