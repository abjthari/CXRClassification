import torch
import torchvision
import torch.nn.functional as F

class EOE(torch.nn.Module):   
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        
        self.classifier = torch.nn.Linear(8, 4)
        
    def forward(self, out):
        out1 = self.modelA(out/2)
        out2 = self.modelB(out/2)

        x = torch.cat((out1, out2), dim=1)
        x = self.classifier(F.relu(x))
        return x