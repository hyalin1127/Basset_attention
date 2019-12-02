import torch
import torch.nn as nn

class SimpleCNN(torch.nn.Module):
    def __init__(self,model_name,num_filter,seqlen,maxpoolsize):
        super(SimpleCNN, self).__init__()

        # Custom filter weight initialization
        self.conv1short = nn.Conv1d(4, num_filter, kernel_size=15, stride=1, padding=7)

        # Model construction
        self.layer1short = nn.Sequential(
            nn.BatchNorm1d(num_filter),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size = maxpoolsize, stride = maxpoolsize))

        if "convolution" in model_name:
            self.layer2 = nn.Sequential(
                nn.Conv1d(num_filter, num_filter*2, kernel_size=5, stride=1, padding=3),
                nn.BatchNorm1d(int(num_filter*2)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.MaxPool1d(kernel_size = maxpoolsize, stride = maxpoolsize))

            self.layer3 = nn.Sequential(
                nn.Conv1d(num_filter*2, num_filter*4, kernel_size=5, stride=1, padding=3),
                nn.BatchNorm1d(int(num_filter*4)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.MaxPool1d(kernel_size = maxpoolsize, stride = maxpoolsize))

    def forward(self, x, model_name):
        # x shape: bs * 4 * 1024
        out_convolutional_forward = self.conv1short(x) # bs * filter_size * 128
        x = torch.flip(x,[1,2])
        out_convolutional_backward = self.conv1short(x)
        out_convolutional_backward = torch.flip(out_convolutional_backward,[2])
        out_convolutional = torch.max(out_convolutional_forward,out_convolutional_backward)

        out_convolutional = self.layer1short(out_convolutional)

        del out_convolutional_forward,out_convolutional_backward

        if "convolution" in model_name:
            out_convolutional = self.layer2(out_convolutional)
            #out_convolutional = self.layer3(out_convolutional)

        return(out_convolutional)
