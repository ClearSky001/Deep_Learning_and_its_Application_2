import pytest
import torch

from lab_04_resnet import IdentityBlock, ConvBlock, ResNet50, get_model
#from torchvision import models

def test_IdentityBlock_score_25():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 256, 56, 56)  # Example input (batch_size, channels, height, width)
    
    block = IdentityBlock(256, 64)
    assert sum(p.numel() for p in block.parameters()) == 70400, "IdentityBlock parameter number does not match"
    
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, -0.01)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, -0.01)

        #for name, param in block.named_parameters():
        #    print(f"{name:<30} Param shape: {str(param.shape):<30} Weight : {torch.sum(param).item()}")

        output = block(input_tensor)
        #print(torch.sum(output).item())
        assert torch.sum(output).item() == pytest.approx(393712.40625, abs=1e-2), "IdentityBlock forward pass gave different value"
        

    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.constant_(m.weight, 0)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 0)
                torch.nn.init.constant_(m.bias, 0)

        output = block(input_tensor)
        assert torch.all(input_tensor == output), "IdentityBlock shortcut path seems to be wrong"

    assert output.shape == input_tensor.shape, "Output shape should be the same as input shape in IdentityBlock"


    block = IdentityBlock(128, 512)
    assert sum(p.numel() for p in block.parameters()) == 3479552, "IdentityBlock parameter number does not match"
    

def test_ConvBlock_score_25():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 256, 56, 56)  # Example input (batch_size, channels, height, width)
    
    block = ConvBlock(256, 128, 2)
    assert sum(p.numel() for p in block.parameters()) == 379392, "ConvBlock parameter number does not match"
    
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, -0.01)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, -0.01)

        #for name, param in block.named_parameters():
        #    print(f"{name:<30} Param shape: {str(param.shape):<30} Weight : {torch.sum(param).item()}")

        output = block(input_tensor)
        # print(torch.sum(output).item())
        assert torch.sum(output).item() == pytest.approx(434.92108154296875, abs=1e-2), "ConvBlock forward pass gave different value"
        

    block = ConvBlock(512, 128, 2)
    assert sum(p.numel() for p in block.parameters()) == 543232, "ConvBlock parameter number does not match"

    block = ConvBlock(256, 256, 2)
    assert sum(p.numel() for p in block.parameters()) == 1184768, "ConvBlock parameter number does not match"

    block = ConvBlock(256, 256, 1)
    assert sum(p.numel() for p in block.parameters()) == 1184768, "ConvBlock parameter number does not match"


def test_ResNet50_score_25():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 3, 64, 64)  # Example input (batch_size, channels, height, width)
    
    model = ResNet50(1000)
    #model = models.resnet50()
    assert sum(p.numel() for p in model.parameters()) == 25557032, "ResNet50 parameter number does not match"
    
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.weight, -0.01)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, -0.01)

        #for name, param in model.named_parameters():
        #    print(f"{name:<30} Param shape: {str(param.shape):<30} Weight : {torch.sum(param).item()}")

        output = model(input_tensor)
        #print(torch.sum(output).item())
        assert torch.sum(output).item() == pytest.approx(-10.0, abs=1e-2), "ResNet50 forward pass gave different value"
        

    model = ResNet50(10)
    assert sum(p.numel() for p in model.parameters()) == 23528522, "ResNet50 parameter number does not match"


def test_transfer_learning_score_25():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(1, 3, 64, 64)  # Example input (batch_size, channels, height, width)

    config = {
        'model_name': 'resnet50',
        'pretrained' : 'IMAGENET1K_V2',
    }
    model = get_model("resnet50", 100, config)
    for name, param in model.named_parameters():
        if name.startswith("fc.") or name.startswith("layer4."):
            assert param.requires_grad == True, "layer4 and fc layer should have requires_grad = True"
        else:
            assert param.requires_grad == False, "layers except for layer4 and fc should have requires_grad = False"

    assert model.fc.out_features == 100, "fc layer should have out_features same as num_classes"
