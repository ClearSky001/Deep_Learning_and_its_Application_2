import pytest
import torch
import numpy as np

from ViT import MultiHeadSelfAttention, ViT


def test_MHA_score_5():
    hidden_dim = 768
    num_heads = 12
    dropout_prob = 0.1
    seq_length = 96

    torch.manual_seed(42)
    input_tensor = torch.randn(4, seq_length, hidden_dim)

    model = MultiHeadSelfAttention(
        hidden_dim=hidden_dim,
        num_head=num_heads,
        dropout_prob=dropout_prob,
    )
    

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 2362368, "ViT model parameter number does not match the expected value."

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)

    assert output.shape == (4, seq_length, hidden_dim), "Output shape does not match the expected shape."

    test_val1 = output.sum().item()
    test_val2 = output[1, 7, 5:10].detach().numpy()
    print(test_val1, f"[{', '.join([str(x) for x in test_val2])}]")
    assert output.sum().item() == pytest.approx(-393990.5, rel=1e-5), "Forward pass gave different value"
    assert np.isclose(test_val2, [3.676529, -76.27301, -96.88431, 759.3391, 325.83307], rtol=1e-2).all(),"Forward pass gave different value"



def test_ViT_score_5():
    image_size = 224
    patch_size = 16
    num_channels = 3
    num_classes = 1000
    hidden_dim = 768
    num_transformer_layers = 12
    num_heads = 12
    feedforward_dim = 3072
    dropout_prob = 0.1

    torch.manual_seed(42)
    input_tensor = torch.randn(4, num_channels, image_size, image_size)

    model = ViT(
        image_size=image_size,
        num_channels=num_channels,
        patch_size=patch_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_transformer_layers=num_transformer_layers,
        num_head=num_heads,
        feedforward_dim=feedforward_dim,
        dropout_prob=dropout_prob,
    )
    

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 86567656, "ViT model parameter number does not match the expected value."

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.manual_seed(123)
                param.copy_(torch.randn_like(param))

    model.train()
    torch.manual_seed(0)
    output = model(input_tensor)

    assert output.shape == (4, num_classes), "Output shape does not match the expected shape."

    test_val1 = output.sum().item()
    test_val2 = output[1,5:10].detach().numpy()
    print(test_val1, f"[{', '.join([str(x) for x in test_val2])}]")
    assert output.sum().item() == pytest.approx(-8405.7529296875, rel=1e-3), "Forward pass gave different value"
    assert np.isclose(test_val2, [-45.429604,-50.441277,-80.59231,58.097385,-7.7294383], rtol=1e-2).all(),"Forward pass gave different value"

