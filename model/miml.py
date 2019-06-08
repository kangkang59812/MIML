import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from collections import OrderedDict
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class MIML(nn.Module):

    def __init__(self, L=1024, K=20, batch_size=8, fine_tune=True):
        """
        Arguments:
            L (int):
                number of labels
            K (int):
                number of sub categories
        """
        super(MIML, self).__init__()
        self.L = L
        self.K = K
        self.batch_size = batch_size
        # pretrained ImageNet VGG
        base_model = torchvision.models.vgg16(pretrained=True)
        base_model = list(base_model.features)[:-1]
        self.base_model = nn.Sequential(*base_model)
        self.fine_tune(fine_tune)

        self.sub_concept_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(512, 512, 1)),
            ('dropout1', nn.Dropout(0)),  # (-1,512,14,14)
            ('conv2', nn.Conv2d(512, K*L, 1)),
            # input need reshape to (-1,L,K,H*W)
            ('maxpool1', nn.MaxPool2d((K, 1))),
            # reshape input to (-1,L,H*W), # permute(0,2,1)
            ('softmax1', nn.Softmax(dim=2)),
            # permute(0,2,1) # reshape to (-1,L,1,H*W)
            ('maxpool2', nn.MaxPool2d((1, 196)))
        ]))
        # self.conv1 = nn.Conv2d(512, 512, 1))

        # self.dropout1=nn.Dropout(0.5)

        # self.conv2=nn.Conv2d(512, K*L, 1)
        # # input need reshape to (-1,L,K,H*W)
        # self.maxpool1=nn.MaxPool2d((K, 1))
        # # reshape input to (-1,L,H*W)
        # # permute(0,2,1)
        # self.softmax1=nn.Softmax(dim = 2)
        # # permute(0,2,1)
        # # reshape to (-1,L,1,H*W)
        # self.maxpool2=nn.MaxPool2d((1, 196))
        # # squeeze()

    def forward(self, x):
        # IN:(8,3,224,224)-->OUT:(8,512,14,14)
        base_out = self.base_model(x)
        # C,H,W = 512,14,14
        _, C, H, W = base_out.shape
        # OUT:(8,512,14,14)

        conv1_out = self.sub_concept_layer.dropout1(
            self.sub_concept_layer.conv1(base_out))

        # shape -> (n_bags, (L * K), n_instances, 1)
        conv2_out = self.sub_concept_layer.conv2(conv1_out)
        # shape -> (n_bags, L, K, n_instances)
        conv2_out = conv2_out.reshape(-1, self.L, self.K, H*W)
        # shape -> (n_bags, L, 1, n_instances),remove dim: 1
        maxpool1_out = self.sub_concept_layer.maxpool1(conv2_out).squeeze(2)

        # softmax
        permute1 = maxpool1_out.permute(0, 2, 1)
        softmax1 = self.sub_concept_layer.softmax1(permute1)
        permute2 = softmax1.permute(0, 2, 1)
        # reshape = permute2.unsqueeze(2)
        reshape = permute2.reshape(-1, self.L, 1, H*W) # predictions_instancelevel
        # shape -> (n_bags, L, 1, 1)
        maxpool2_out = self.sub_concept_layer.maxpool2(reshape)
        out = maxpool2_out.squeeze()

        return out

    def fine_tune(self, fine_tune=True):
        # only fine_tune the last three convs
        layer = -6
        for p in self.base_model.parameters():
            p.requires_grad = False
        for c in list(self.base_model.children())[-6:]:
            for p in c.parameters():
                p.requires_grad = True


if __name__ == "__main__":
    model = MIML()
    out = model(torch.randn(8, 3, 224, 224))
    print(out.shape)
    summary(model.cuda(), (3, 224, 224), 8)
