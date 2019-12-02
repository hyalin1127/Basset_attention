from __future__ import print_function
import torch
import torch.utils.data as data
import h5py
from pathlib import Path

def get_train_loader_hdf5(datainput_path,train_chromosome,batch_size):
    train_data = H5Dataset_train(datainput_path,train_chromosome)
    loader_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': 16, 'drop_last': True, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data,**loader_params)
    return(train_loader)

def get_test_loader_hdf5(datainput_path,test_chromosome,batch_size,sample_number):
    test_data = H5Dataset_test(datainput_path,test_chromosome,sample_number)
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16, 'drop_last': True, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data,**loader_params)
    return(test_loader)

def get_test_loader_hdf5_with_coordinates(datainput_path,test_chromosome,batch_size,sample_number):
    test_data = H5Dataset_test_with_coordinates(datainput_path,test_chromosome,sample_number)
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16, 'drop_last': True, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data,**loader_params)
    return(test_loader)

class H5Dataset_train(data.Dataset):
    #https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3
    def __init__(self, datainput_path,chromosome):
        super(H5Dataset_train, self).__init__()
        h5_file = h5py.File("/%s/hg38_%s_basset_file.hdf5" %(datainput_path,chromosome),'r')
        self.sequence = torch.from_numpy(h5_file.get('sequence')[()]).float()
        self.DNase = torch.from_numpy(h5_file.get('DNase')[()]).float()
        self.DNase = self.DNase[:,range(1,101,10)]
        self.DNase += 0.056
        self.DNase = self.DNase * 0.9
        del h5_file

    def __getitem__(self, index):
        return (self.sequence[index,:,:],self.DNase[index,:])

    def __len__(self):
        return self.DNase.shape[0]

class H5Dataset_test(data.Dataset):
    def __init__(self, datainput_path,chromosome,sample_number):
        super(H5Dataset_test, self).__init__()
        h5_file = h5py.File("/%s/hg38_%s_basset_file_complete_for_test_%s.hdf5" %(datainput_path,chromosome,sample_number),'r')
        self.sequence = torch.from_numpy(h5_file.get('sequence')[()]).float()
        self.DNase = torch.from_numpy(h5_file.get('DNase')[()]).float()
        self.DNase = self.DNase[:,range(1,101,10)]
        #self.coordinates = torch.from_numpy(h5_file.get('coordinates')[()]).float()
        del h5_file

    def __getitem__(self, index):
        return (self.sequence[index,:,:],self.DNase[index,:])

    def __len__(self):
        return self.DNase.shape[0]

class H5Dataset_test_with_coordinates(data.Dataset):
    def __init__(self, datainput_path,chromosome,sample_number):
        super(H5Dataset_test_with_coordinates, self).__init__()
        h5_file = h5py.File("/%s/hg38_%s_basset_file_complete_for_test_%s.hdf5" %(datainput_path,chromosome,sample_number),'r')
        self.sequence = torch.from_numpy(h5_file.get('sequence')[()]).float()
        self.DNase = torch.from_numpy(h5_file.get('DNase')[()]).float()
        self.DNase = self.DNase[:,range(1,101,10)]
        self.coordinates = torch.from_numpy(h5_file.get('coordinates')[()]).float()
        del h5_file

    def __getitem__(self, index):
        return (self.sequence[index,:,:],self.DNase[index,:],self.coordinates[index,:])

    def __len__(self):
        return self.DNase.shape[0]
