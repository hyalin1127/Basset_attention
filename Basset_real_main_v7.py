from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.metrics import average_precision_score
import pickle
import itertools
import math

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_auc_score

from optparse import OptionParser
from Basset_app_IO_v6 import *
from Basset_app_motifs import *
from Basset_app_attention_v6 import *
from Basset_app_CNN import *
from Basset_app_early_stopping import *
from torch.optim.lr_scheduler import StepLR

# Configuring device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepare_optparser():
    usage = "usage: %prog -i train_file_name -t test_file_name -m motif_path"
    description = "Deep learning with self-attention."
    optparser = OptionParser(version="%prog v1.00", description=description, usage=usage, add_help_option=False)
    optparser.add_option("-h","--help",action="help",help="Show this help message and exit.")
    optparser.add_option("-m","--motif_path",dest="motif_path",type="string",
                         help="Motif path")
    optparser.add_option("-t","--train_data",dest="train_data",type="string",
                         help="Train data name")
    optparser.add_option("-s","--test_data",dest="test_data",type="string",
                         help="Test data name")
    (options,args) = optparser.parse_args()
    return(options)

class main_model(nn.Module):
    def __init__(self,model_name,num_DNase_classes,num_filter,seqlen,maxpoolsize):
        super().__init__()
        #
        self.num_filter = num_filter
        self.sequence_length = seqlen
        self.maxpoolsize = maxpoolsize

        self.CNN = SimpleCNN(model_name,num_filter,seqlen,self.maxpoolsize)
        self.fc_dropout = nn.Dropout(0.3)

        self.cell_embedding_dim = 16
        self.cell_embedding = nn.Embedding(num_DNase_classes,self.cell_embedding_dim)

        if "convolution" in model_name:
            #self.fc_convolutional = nn.Linear(int(num_filter*4) * int(self.sequence_length/(self.maxpoolsize*self.maxpoolsize*self.maxpoolsize))* 1, num_DNase_classes)
            self.fc_convolutional = nn.Linear(int(num_filter*2) * int(self.sequence_length/(self.maxpoolsize*self.maxpoolsize))* 1, 1)

        if "no_pe" in model_name or "with_pe" in model_name:

            # Define attention layer
            self.N = 1 # number of extra attention layer
            self.d_input = num_filter + self.cell_embedding_dim
            #self.d_input = num_filter
            #self.d_output = 4*self.d_input # syn v3....v8
            self.d_output = 64#self.d_input # syn v9
            self.h = 4 #heads
            self.d_k = int((self.d_output)/(self.h)) #d_model // heads

            # relative positional embedding
            pos_k = 8
            self.rpos_k_embed = nn.Embedding(pos_k*2+1,self.d_output)
            self.rpos_v_embed = nn.Embedding(pos_k*2+1,self.d_output)

            # ------------------------------------------------------------------

            self.layer1 = EncoderLayer(self.d_input,self.d_output,self.h)
            self.layers = clones(EncoderLayer(self.d_output,self.d_output,self.h), self.N)

            self.postmodification = nn.Sequential( # syn v8
                nn.BatchNorm1d(self.d_output),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.MaxPool1d(kernel_size = maxpoolsize, stride = maxpoolsize))

            self.fc_attention = nn.Linear(self.d_output* int(self.sequence_length/(self.maxpoolsize*self.maxpoolsize))* 1, 64)

            self.fc_output = nn.Linear(64, 1)


    def forward(self, x, model_name,mapping,embedding_index):
        out = self.CNN(x,model_name)  # bs*num_filter*npos

        bs,num_filter,npos = out.shape
        #print(self.cell_embedding.weight.data[0,0])
        embedding = self.cell_embedding(embedding_index)
        embedding = embedding.view(1,self.cell_embedding_dim,1)
        embedding = torch.cat([embedding]*bs,dim=0)
        embedding = torch.cat([embedding]*npos,dim=2)

        out = torch.cat((out,embedding),1)

        if "convolution" in model_name:
            out = out.contiguous().view(-1, int(self.num_filter*2) * int(self.sequence_length/(self.maxpoolsize*self.maxpoolsize)) * 1)
            #out = out.contiguous().view(-1, int(self.num_filter*4) * int(self.sequence_length/(self.maxpoolsize*self.maxpoolsize*self.maxpoolsize)) * 1)
            out = self.fc_dropout(out)
            out = self.fc_convolutional(out)
            return(out)

        if "no_pe" in model_name or "with_pe" in model_name:
            #mapping = (self.mapping).to(device)
            x,y,z = mapping.shape
            mapping = mapping.view(y,z)
            rpos_k = self.rpos_k_embed(mapping)
            rpos_v = self.rpos_v_embed(mapping)

            del x,y,z,mapping

            out = out.transpose(1,2)  # 128*256*1024

            # Attention
            out = self.layer1(out,model_name,rpos_k,rpos_v)
            for i in range(self.N):
                out = self.layers[i](out,model_name,rpos_k,rpos_v)

            out = out.transpose(1,2)
            out = self.postmodification(out) # syn v8
            out = out.contiguous().view(-1, self.d_output * int(self.sequence_length/(self.maxpoolsize*self.maxpoolsize)) * 1)
            #out = self.fc_dropout(out) # so far best: not dropout, no label smoothing
            out = self.fc_dropout(F.relu(out))
            out = self.fc_attention(out)
            out = self.fc_output(self.fc_dropout(F.relu(out))) # v4
            return(out)

def trainNet_embedding(model,num_epochs, batch_size,learning_rate,train_chromosomes,test_chromosomes,model_name,num_DNase_classes,sequence_length, maxpoolsize,weight):
    # Optimizer
    #optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    my_list = ["cell_embedding.weight"]
    base_params = [] # should be changed into torch.nn.ParameterList()
    specific_params = [] # should be changed into torch.nn.ParameterList()

    for name, param in model.module.named_parameters():
    #for name, param in model.named_parameters():
        if name in my_list:
            specific_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.Adam([
    {"params":base_params},
    {"params":specific_params, "lr": 0.0003}
    ],lr=learning_rate)

    #scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    #es = EarlyStopping(patience=3)

    pos_weight_vectors = [weight]#*num_DNase_classes
    pos_weight_vectors = torch.FloatTensor(pos_weight_vectors).to(device)

    mapping = get_mapping(sequence_length,maxpoolsize)
    mapping = mapping.to(device)

    #criterion = nn.BCEWithLogitsLoss(reduction='none',pos_weight = pos_weight_vectors)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight_vectors)

    # Loss record
    loss_record = open("/%s/Basset_model_loss_record_%s.txt" %(model_path,model_name),'wt')
    test_record = open("/%s/Basset_model_test_record_%s.txt" %(model_path,model_name),'wt')

    latest_scores = [0]*10

    # Loop for n_epochs
    if "no_pe" in model_name:
        model.module.rpos_k_embed.requires_grad = False
        model.module.rpos_v_embed.requires_grad = False

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # train part
        model.train()
        running_loss = 0.0
        #scheduler.step()

        if epoch < 30:
            #model.CNN.conv1short.requires_grad = False
            model.module.CNN.conv1short.requires_grad = False
        else:
            #model.CNN.conv1short.requires_grad = True
            model.module.CNN.conv1short.requires_grad = True

        for train_chromosome in train_chromosomes:
            for DNase_index in range(num_DNase_classes):
                torch_DNase_index = torch.LongTensor([[DNase_index],[DNase_index]])
                torch_DNase_index = torch_DNase_index.to(device)

                train_loader = get_train_loader_hdf5("%s/train_folder" %(datainput_path),train_chromosome,batch_size)
                train_loader_iter = iter(train_loader) #https://stackoverflow.com/questions/53280967/pytorch-nextitertraining-loader-extremely-slow-simple-data-cant-num-worke

                for i in range(len(train_loader)):
                    images, labels = next(train_loader_iter)
                    images = images.to(device) # Dimentions of images: batch_size, channels, length

                    label = labels[:,DNase_index:DNase_index+1]
                    label = label.to(device)

                    # Forward pass
                    outputs = model(images, model_name,mapping,torch_DNase_index)
                    loss = criterion(outputs, label)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    running_loss += loss.item()

                del images,loss,outputs,labels
                torch.cuda.empty_cache()

            del train_loader,train_loader_iter
            torch.cuda.empty_cache()

        loss_record.write("%s\n" %str(running_loss))

        # validation part
        if epoch > 0 and epoch%10 == 0:
            model.eval()
            with torch.no_grad():
                scores_record = []
                labels_record = []
                for test_chromosome in test_chromosomes:
                    for DNase_index in range(num_DNase_classes):
                        torch_DNase_index = torch.LongTensor([[DNase_index],[DNase_index]])
                        torch_DNase_index = torch_DNase_index.to(device)

                        for sample_number in [0,1,6]:
                            test_loader = get_test_loader_hdf5("%s/test_folder" %datainput_path,test_chromosome,batch_size,sample_number)
                            test_loader_iter = iter(test_loader)
                            for i in range(len(test_loader_iter)):
                                images, labels = next(test_loader_iter)
                                images = images.to(device)

                                label = labels[:,DNase_index:DNase_index+1]
                                label = label.to(device)

                                # Forward pass
                                outputs = model(images, model_name,mapping,torch_DNase_index)
                                outputs = torch.sigmoid(outputs)

                                labels_record += label.squeeze().tolist()
                                scores_record += outputs.squeeze().tolist()

                            del outputs,images,labels
                            torch.cuda.empty_cache()

                #labels = list(itertools.chain(*labels_record))
                #scores = list(itertools.chain(*scores_record))

                labels = labels_record
                scores = scores_record

                average_precision = average_precision_score(labels, scores)
                test_record.write('Average precision-recall score: {0:0.4f}\n'.format(average_precision))
                auc = roc_auc_score(labels, scores)
                test_record.write('AUC: {0:0.4f}\n'.format(auc))

                del labels,scores, labels_record,scores_record
                torch.cuda.empty_cache()

def testNet_embedding(model,batch_size,test_chromosomes,model_name,num_DNase_classes,sequence_length, maxpoolsize):
    mapping = get_mapping(sequence_length,maxpoolsize)
    mapping = mapping.to(device)

    model.eval()
    with torch.no_grad():
        info_dict = {i:torch.Tensor() for i in range(num_DNase_classes)}

        scores_record = []
        labels_record = []
        for test_chromosome in test_chromosomes:
            for sample_number in [1,2,3,4,5,6,7,8,9,10,11]:
                test_loader = get_test_loader_hdf5_with_coordinates("%s/test_folder" %datainput_path,test_chromosome,batch_size,sample_number)
                test_loader_iter = iter(test_loader)

                for i in range(len(test_loader_iter)):
                    images, labels, coordinates = next(test_loader_iter)
                    images = images.to(device)
                    #coordinates = coordinates.to(device)

                    for DNase_index in range(num_DNase_classes):
                        torch_DNase_index = torch.LongTensor([[DNase_index],[DNase_index]])
                        #torch_DNase_index = torch.LongTensor([DNase_index])
                        torch_DNase_index = torch_DNase_index.to(device)

                        label = labels[:,DNase_index:DNase_index+1]
                        #label = label.to(device)

                        # Forward pass
                        outputs = model(images, model_name,mapping,torch_DNase_index)
                        outputs = torch.sigmoid(outputs)
                        outputs = outputs.to('cpu')
                        #labels_record += label.squeeze().tolist()
                        #scores_record += outputs.squeeze().tolist()

                        temp_info = torch.cat([label,outputs,coordinates],1)
                        info_dict[DNase_index] = torch.cat([info_dict[DNase_index],temp_info],0)


                    del outputs
                    torch.cuda.empty_cache()

        for DNase_index in range(num_DNase_classes):
            info_dict[DNase_index] = (info_dict[DNase_index]).to('cpu')
            np.savetxt("/%s/Basset_model_info_record_final_evaluation_%s_%s.txt" %(model_path,model_name,str(DNase_index)),(info_dict[DNase_index]).numpy())

def main():
    opts=prepare_optparser()

    # Settings:
    datainput_path = "."
    motif_path = opts.motif_path
    model_path = "."
    train_chromosomes = [opts.train_data]
    test_chromosomes = [opts.test_data]


    # Fixed hyperparameters
    num_DNase_classes = 10#213
    num_filter = 252
    seqlen = 1024
    maxpoolsize = 8

    # Hyper-parameters
    num_epochs = 50
    batch_size = 384
    learning_rate = 0.00003 # 0.005 best for two layers when batch size is 384
    weight = 10

    for model_name in ["Basset_model_real_no_pe_known_motif_embedding"]:
        parameters = {'batch_size':batch_size,'model_name':model_name,'num_DNase_classes':num_DNase_classes,'sequence_length':seqlen,'maxpoolsize':maxpoolsize}

        # Build the model
        model = main_model(model_name,num_DNase_classes,num_filter,seqlen,maxpoolsize)

        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        TFs,short_predefined_filters,long_predefined_filters = get_selected_filter_matrix(motif_path)
        for i,predefined_filter in short_predefined_filters.items():
            (model.CNN.conv1short.weight.data)[i,:,:] = predefined_filter
        del TFs,short_predefined_filters,long_predefined_filters

        # parallization
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model,device_ids=[0,1])
        model.to(device)

        #-----------------------------------------------------------------------
        # Retraining
        model.load_state_dict(torch.load('/%s/Basset_model_Basset_model_real_no_pe_known_motif_embedding.ckpt' %(model_path)))
        #-----------------------------------------------------------------------

        # Train the model
        trainNet_embedding(model, num_epochs=num_epochs, learning_rate= learning_rate,train_chromosomes = train_chromosomes,test_chromosomes = test_chromosomes,weight=weight,**parameters)

        # Save the model checkpoint
        torch.save(model.state_dict(), '/%s/Basset_model_%s.ckpt' %(model_path,model_name))
        torch.save(model, '/%s/Basset_model_%s.tmp' %(model_path,model_name)) # Save the whole model

        #-----------------------------------------------------------------------

        # Test the model
        #model.load_state_dict(torch.load('/%s/Basset_model_%s.ckpt' %(model_path,model_name)))

        # to load model to cpu.
        #model.load_state_dict(torch.load('/%s/Basset_model_%s.ckpt' %(model_path,model_name), map_location=lambda storage, loc: storage),strict=False)

        #testNet_embedding(model,test_chromosomes = test_chromosomes,**parameters)

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        sys.stderr.write("User interrupt me! ;-) Bye!\n")
        sys.exit(0)
