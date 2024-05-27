import torch
import torchvision.models as models
import argparse
import os

print(">>>>> Detect a device to allocate model Start ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
print(">>>>> Detect a device to allocate model End")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.pt")
    args = parser.parse_args()
    print(">>>>> Download Pretrained Resnet50 Start ...")
    resnet50 = models.resnet50(pretrained=True)
    print(">>>>> Download Pretrained Resnet50 End")
    dummy_input = torch.randn(1, 3, 224, 224)
    print(">>>>> Convert Pretrained Resnet50 to Inference Mode Start ...")
    resnet50 = resnet50.eval()
    print(">>>>> Convert Pretrained Resnet50 to Inference Mode End")

    print(">>>>> Allocate Pretrained Resnet50 on Device (GPU/CPU) Start ...")
    resnet50.to(device)
    print(">>>>> Allocate Pretrained Resnet50 on Device (GPU/CPU) End")

    #torch.jit.script
    #Scripting a function or nn.Module will inspect the source code, 
    #compile it as TorchScript code using the TorchScript compiler, 
    #and return a ScriptModule or ScriptFunction.
        
    print(">>>>> Convert Pretrained Resnet50 to TorchScript format Start ...")
    resnet50_jit = torch.jit.script(resnet50)
    print(">>>>> Convert Pretrained Resnet50 to TorchScript format End")
        
    print(">>>>> Save Pretrained Resnet50 in TorchScript format Start ...")
    resnet50_jit.save(args.save)
    print(">>>>> Save Pretrained Resnet50 in TorchScript format Start ...")

    print("Saved {}".format(args.save))