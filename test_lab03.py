import pytest
import inspect

import torch
from torchvision import transforms
from lab_03_cnn import CustomImageDataset, SimpleCNN, train_loop, evaluation_loop, train_main_MLP

def test_custom_dataset_score_25():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    custom_datset = CustomImageDataset(root_dir = 'resources/cat_dog_images', 
                                    metadata_filename = "meta.csv",
                                    transform = transform)
    assert len(custom_datset) == 6, "CustomImageDataset __len__ gave wrong value" 

    test_vals = {
        0 : [torch.Size([3, 800, 1200]), 1156067.0, 1],
        1 : [torch.Size([3, 900, 1200]), 1468958.5, 0],
        2 : [torch.Size([3, 800, 1200]), 1054946.0, 1],
        3 : [torch.Size([3, 900, 1200]), 1658777.625, 0],
        4 : [torch.Size([3, 900, 1200]), 1459332.75, 1],
        5 : [torch.Size([3, 900, 742]), 893281.3125, 0],
    }
    for i, (X, y) in enumerate(custom_datset):
        test_shape, test_sum, test_label = test_vals[i]
        assert X.shape == test_shape, f"CustomImageDataset {i}-th image data is not correct"
        assert X.sum().item() == pytest.approx(test_sum, abs=1e-2), f"CustomImageDataset {i}-th image data is not correct"
        assert test_label == y, f"CustomImageDataset {i}-th label is not correct"


    custom_datset = CustomImageDataset(root_dir = 'resources/cat_dog_images', 
                                    metadata_filename = "meta_imbalanced.csv",
                                    transform = transform)
    assert len(custom_datset) == 5, "CustomImageDataset __len__ gave wrong value" 

    test_vals = {
        0 : [torch.Size([3, 900, 1200]), 1468958.5, 0],
        1 : [torch.Size([3, 800, 1200]), 1054946.0, 1],
        2 : [torch.Size([3, 900, 1200]), 1658777.625, 0],
        3 : [torch.Size([3, 900, 1200]), 1459332.75, 1],
        4 : [torch.Size([3, 900, 742]), 893281.3125, 0],
    }
    for i, (X, y) in enumerate(custom_datset):
        test_shape, test_sum, test_label = test_vals[i]
        assert X.shape == test_shape, f"CustomImageDataset {i}-th image data is not correct"
        assert X.sum().item() == pytest.approx(test_sum, abs=1e-2), f"CustomImageDataset {i}-th image data is not correct"
        assert test_label == y, f"CustomImageDataset {i}-th label is not correct"


def test_wandb_score_25():
    source_code = inspect.getsource(train_loop)
    assert "wandb.log" in source_code, "wandb.log was not called in train_loop()"

    source_code = inspect.getsource(evaluation_loop)
    assert "wandb.log" in source_code, "wandb.log was not called in evaluation_loop()"

    source_code = inspect.getsource(train_main_MLP)
    assert "wandb.init" in source_code, "wandb.init was not called in train_main_MLP()"
    source_code = inspect.getsource(train_main_MLP)
    assert "wandb.finish()" in source_code, "wandb.finish() was not called in train_main_MLP()"

def test_cnn_score_25():
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.rand(4, 1, 28, 28)  # Example input (batch_size, channels, height, width)


    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 45226  
    assert total_params == expected_params, f"Total number of SimpleCNN model parameters should be {expected_params}, but got {total_params}."


    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.weight, -0.01)  # Setting a constant weight for simplicity
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, -0.01)

        #for name, param in model.named_parameters():
        #    print(f"{name:<30} Param shape: {str(param.shape):<30} Weight : {torch.sum(param).item()}")

        output = model(input_tensor)
        #print(torch.sum(output).item())
        assert torch.sum(output).item() == pytest.approx(-0.3999999761581421, abs=1e-2), "SimpleCNN forward pass gave different value"

        assert output.shape == (4, 10), "Output shape of SimpleCNN is not what expected"

def test_gitignore_score_25():
    # Path to the .gitignore file
    gitignore_path = "../.gitignore"
    
    # Initialize flags for checking if both entries exist at the start of a line
    wandb_found = False
    checkpoints_found = False
    
    # Open the .gitignore file and check line-by-line
    with open(gitignore_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()  # Remove any leading/trailing whitespaces
            if stripped_line == 'wandb':
                wandb_found = True
            if stripped_line == 'checkpoints':
                checkpoints_found = True
    
    # Assert that both 'wandb' and 'checkpoints' are found at the start of lines
    assert wandb_found, "'wandb' not found in .gitignore"
    assert checkpoints_found, "'checkpoints' not found in .gitignore"