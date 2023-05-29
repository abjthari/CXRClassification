import torch
import torchvision
import torch.nn.functional as F

class EnsembleModel(torch.nn.Module):   
    def __init__(self, modelA, modelB, modelC):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        
        
        
        self.classifier = torch.nn.Linear(2004, 4)
        
    def forward(self, out):
        out1 = self.modelA(out/3)
        out2 = self.modelB(out/3)
        out3 = self.modelC(out/3)

        x = torch.cat((out1, out2, out3), dim=1)
        x = self.classifier(F.relu(x))
        return x
    
    
