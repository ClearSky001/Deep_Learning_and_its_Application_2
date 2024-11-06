import pytest
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from unet import JointRandomHorizontalFlip, DoubleConv, Down, Up, UNet, calculate_mIoU

DATA_ROOT_DIR = "/datasets"



def test_horizontal_flip_score_1():
    img = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).unsqueeze(0)  # Shape (1, 2, 2)
    mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)  # Shape (2, 2)
    img_pil = F.to_pil_image(img)
    mask_pil = Image.fromarray(mask.numpy())

    torch.manual_seed(0)
    transform = JointRandomHorizontalFlip(p=0.5)  # Set p=1.0 to always flip

    flipped_img_pil, flipped_mask_pil = transform(img_pil, mask_pil)
    flipped_img = torch.from_numpy(np.array(flipped_img_pil, dtype=np.uint8)).unsqueeze(0)
    flipped_mask = torch.from_numpy(np.array(flipped_mask_pil, dtype=np.uint8))

    expected_flipped_img = torch.tensor([[2, 1], [4, 3]], dtype=torch.uint8).unsqueeze(0)
    expected_flipped_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)

    assert torch.equal(flipped_img, expected_flipped_img), "Image not flipped correctly"
    assert torch.equal(flipped_mask, expected_flipped_mask), "Mask not flipped correctly"


    flipped_img_pil, flipped_mask_pil = transform(img_pil, mask_pil)
    flipped_img = torch.from_numpy(np.array(flipped_img_pil, dtype=np.uint8)).unsqueeze(0)
    flipped_mask = torch.from_numpy(np.array(flipped_mask_pil, dtype=np.uint8))

    expected_img = img
    expected_mask = mask
    assert torch.equal(flipped_img, expected_img), "Image should not be flipped"
    assert torch.equal(flipped_mask, expected_mask), "Mask should not be flipped"

# def test_double_conv_score_1():
#     in_channels = 3
#     out_channels = 8
#     model = DoubleConv(in_channels, out_channels)
#     total_params = sum(p.numel() for p in model.parameters())
#     assert total_params == 824, "DoubleConv parameter number does not match"

#     input_tensor = torch.randn(16, in_channels, 32, 32)

#     output = model(input_tensor)
#     assert output.shape == (16, out_channels, 32, 32), f"DoubleConv output shape is incorrect."


def test_down_block_score_2():
        in_channels = 8
        out_channels = 16

        model = Down(in_channels, out_channels, dropout_prob=0.5)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 3520, "Down block parameter number does not match"

        torch.manual_seed(42)
        input_tensor = torch.randn(16, in_channels, 64, 64)

        
        with torch.no_grad():
            for name, param in model.named_parameters():
                print("layer ", name)
                if param.requires_grad:
                    torch.manual_seed(123)
                    param.copy_(torch.randn_like(param))


        model.train()
        torch.manual_seed(0)
        output = model(input_tensor)
        assert output.shape == (16, out_channels, 32, 32), f"Down block output shape is incorrect."

        test_val1 = output.sum().item()
        test_val2 = output[1, 8, 12, 10:15].detach().numpy()
        print(test_val1, f"[{', '.join([str(x) for x in test_val2])}]")
        assert output.sum().item() == pytest.approx(96662.203125, rel=1e-5), "Down block Forward pass gave different value"
        assert np.isclose(test_val2, [0.22965656, 2.546472, 2.48545, 0.0, 0.0], rtol=1e-4).all(),"Down block Forward pass gave different value"


def test_up_block_score_2():
        in_channels = 16
        out_channels = 8

        model = Up(in_channels, out_channels)

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 2280, "Up block parameter number does not match"

        torch.manual_seed(42)
        upsampling_input = torch.randn(16, in_channels, 16, 16)
        skip_connection = torch.randn(16, in_channels // 2, 32, 32)

        with torch.no_grad():
            for name, param in model.named_parameters():
                print("layer ", name)
                if param.requires_grad:
                    torch.manual_seed(123)
                    param.copy_(torch.randn_like(param))


        model.train()
        torch.manual_seed(0)
        output = model(upsampling_input, skip_connection)
        
        assert output.shape == (16, out_channels, 32, 32), f"Up block output shape is incorrect."

        test_val1 = output.sum().item()
        test_val2 = output[1, 5, 12, 10:15].detach().numpy()
        print(test_val1, f"[{', '.join([str(x) for x in test_val2])}]")
        assert output.sum().item() == pytest.approx(10133.8759765625, rel=1e-5), "Up block Forward pass gave different value"
        assert np.isclose(test_val2, [0.1589329, 0.21392046, 0.0, 0.15554062, 0.1784415], rtol=1e-4).all(),"Up block Forward pass gave different value"

def test_unet_score_3():
    in_channels = 3
    num_classes = 10
    model = UNet(in_channels, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 31038218, "Up block parameter number does not match"

    torch.manual_seed(42)
    input_tensor = torch.randn(16, in_channels, 256, 256)

    with torch.no_grad():
        for name, param in model.named_parameters():
            print("layer ", name)
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))


    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)
    
    assert output.shape == (16, num_classes, 256, 256), f"UNet output shape is incorrect."

    test_val1 = output.sum().item()
    test_val2 = output[1, 5, 12, 10:15].detach().numpy()
    print(test_val1, f"[{', '.join([str(x) for x in test_val2])}]")
    assert output.sum().item() == pytest.approx(57232036.0, rel=1e-5), "Up block Forward pass gave different value"
    assert np.isclose(test_val2, [-1.958639, -4.139183, -3.1430478, -2.1344676, -4.88827], rtol=1e-4).all(),"Up block Forward pass gave different value"


def test_mIoU_score_2():
    """Test the calculate_mIoU function."""
    num_classes = 3
    batch_size = 16
    H, W = 32, 32

    torch.manual_seed(0)
    output = torch.randn(batch_size, num_classes, H, W)
    target = torch.randint(0, num_classes, (batch_size, H, W))

    miou = calculate_mIoU(output, target, num_classes)

    assert miou == pytest.approx(0.2038558106440083, abs=1e-5), "calculate_mIoU gave different value"

    output = torch.zeros(batch_size, num_classes, H, W)
    for c in range(num_classes):
        output[:, c][target == c] = 10.0  # High score for correct class

    miou = calculate_mIoU(output, target, num_classes)
    assert miou == 1.0, f"mIoU should be 1.0 when predictions are perfect, got {miou}"

    # Create completely wrong predictions
    wrong_target = (target + 1) % num_classes
    miou = calculate_mIoU(output, wrong_target, num_classes)
    assert miou == 0.0, f"mIoU should be 0.0 when predictions are completely wrong, got {miou}"
