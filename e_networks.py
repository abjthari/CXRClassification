import torch
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_vgg16(pretrained=False, out_features=None, path=None):
    model = torchvision.models.vgg16(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:-1],
            torch.nn.Linear(in_features=4096, out_features=out_features)
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)

def get_inception_v3(pretrained=False, out_features=None, path=None):
    model = torchvision.models.inception_v3(pretrained=pretrained)
    if out_features is not None:
         model.AuxLogits.fc = torch.nn.Linear(768, 4)
         model.fc = torch.nn.Linear(in_features=2048, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)

def get_squeezenet1_0(pretrained=False, out_features=None, path=None):
    model = torchvision.models.squeezenet1_0(pretrained=pretrained)
    model.aux_logits=False
    if out_features is not None:
         model.classifier=torch.nn.Conv2d(512, 4, kernel_size=(3,3), stride=(1,1))
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)

def get_efficientnet_b0(pretrained=False, out_features=None, path=None):
    model = torchvision.models.efficientnet_b0(pretrained=pretrained)
    model.aux_logits=False
    if out_features is not None:
         model.fc = torch.nn.Linear(in_features=2048, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)

def get_alexnet(pretrained=False, out_features=None, path=None):
    model = torchvision.models.alexnet(pretrained=pretrained)
    if out_features is not None:
         model.fc = torch.nn.Linear(in_features=4096, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)


def get_resnet18(pretrained=False, out_features=None, path=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)


def get_densenet121(pretrained=False, out_features=None, path=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Linear(
            in_features=1024, out_features=out_features
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)
