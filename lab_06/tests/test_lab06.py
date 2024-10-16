import pytest
import torch
import numpy as np

from yolo import YOLODataset, load_VOC_YOLO_datasets, YOLOLoss, intersection_over_union, non_max_suppression, YOLOv1Resnet18

DATA_ROOT_DIR = "/datasets"
def test_YOLODataset_score_3():
    train_dataset, test_dataset = load_VOC_YOLO_datasets(DATA_ROOT_DIR, S = 6, B = 3)

    targets = test_dataset[64][1][3][1]
    assert targets.sum().item() == pytest.approx(13.588615417480469, abs=1e-4), "YOLODataset target tensor value is different"
    assert torch.isclose(targets[0], torch.tensor([0.3846, 0.8700, 0.3195, 0.8040, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]), rtol=1e-04).all(), "YOLODataset target tensor value is different"
    
    targets = test_dataset[4][1][3][1]
    assert targets.sum().item() == pytest.approx(0, abs=1e-4), "YOLODataset target tensor value is different"

    targets = test_dataset[5][1][3][2]
    assert torch.isclose(targets[0], torch.tensor([0.9340, 0.4880, 5.5320, 2.2400, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]), rtol=1e-04).all(), "YOLODataset target tensor value is different"

def test_YOLOLoss_score_2():
    criterion = YOLOLoss(lambda_coord = 10, lambda_noobj=0.2)
    train_dataset, test_dataset = load_VOC_YOLO_datasets(DATA_ROOT_DIR, S = 7, B = 2)

    def get_loss(idx):
        X, target = test_dataset[idx]
        torch.manual_seed(idx) 
        y_pred = torch.randn_like(target)
        loss = criterion(y_pred.unsqueeze(0), target.unsqueeze(0)).item()
        return loss

    assert get_loss(3) == pytest.approx(226.64144897460938, rel = 1e-4), "YOLOLoss forward gave different value"
    assert get_loss(6) == pytest.approx(409.28594970703125, rel = 1e-4), "YOLOLoss forward gave different value"
    assert get_loss(10) == pytest.approx(88.74713897705078, rel = 1e-4), "YOLOLoss forward gave different value"

def test_iou_score_2():
    torch.manual_seed(0) 
    box1, box2 = torch.rand(8, 4), torch.rand(8, 4)

    iou = intersection_over_union(box1, box2)

    assert iou.shape == torch.Size([8, 1]), "intersection_over_union function gave different shape"

    assert torch.isclose(iou.squeeze(), torch.tensor([0.0, 0.35670939087867737, 0.0, 0.2356027215719223, 0.02377595193684101, 0.1724998950958252, 0.0, 0.0]), rtol=1e-04).all(), "intersection_over_union function gave wrong value. Check your implementation"

    box1 = torch.tensor([[3, 2, 2, 2]]) # torch.tensor([[2, 1, 4, 3]])
    box2 = torch.tensor([[2, 3, 2, 2]]) #torch.tensor([[1, 2, 3, 4]])

    assert intersection_over_union(box1, box2) < 1, "The IoU value area must be always smaller or equal than 1"
    assert np.isclose(intersection_over_union(box1, box2), 0.14285714), "intersection_over_union function gave wrong value. Check your implementation"

    ## Test case 2: boxes do not intersect
    box1 = torch.tensor([[2, 3, 2, 2]])# torch.tensor([[1,2,3,4]])
    box2 = torch.tensor([[6, 7, 2, 2]])# torch.tensor([[5,6,7,8]])
    assert intersection_over_union(box1, box2) == 0, "Intersection must be 0"

    ## Test case 3: boxes intersect at vertices only
    box1 = torch.tensor([[1.5, 1.5, 1, 1]]) # torch.tensor([[1,1,2,2]])
    box2 = torch.tensor([[2.5, 2.5, 1, 1]]) # torch.tensor([[2,2,3,3]])
    assert intersection_over_union(box1, box2) == 0, "Intersection at vertices must be 0"

    ## Test case 4: boxes intersect at edge only
    box1 = torch.tensor([[2, 2, 2, 2]]) # torch.tensor([[1,1,3,3]])
    box2 = torch.tensor([[2.5, 3.5, 1, 1]]) # torch.tensor([[2,3,3,4]])
    assert intersection_over_union(box1, box2) == 0, "Intersection at edges must be 0"

def test_nms_score_2():
    # unit test 
    torch.manual_seed(0)
    class_preds = torch.randint(20, (64,1))
    scores = torch.rand((64,1))
    boxes = torch.rand(64, 4)
    nms_input = torch.cat((class_preds, scores, boxes), dim = 1).tolist()

    nms_result = non_max_suppression(nms_input, 0.5, 0.5)
    assert len(nms_result) == 33
    assert torch.isclose(torch.tensor(nms_result[10]), torch.tensor([19.0, 0.8369089365, 0.9384382367, 0.1752943992, 0.44311922788, 0.6432467699])).all()

    nms_result = non_max_suppression(nms_input, 0.5, 0.6)
    assert len(nms_result) == 25

    nms_result = non_max_suppression(nms_input, 0.2, 0.6)
    assert len(nms_result) == 22

def test_model_score_1():
    input = torch.randn(64, 3, 224, 224)
    model = YOLOv1Resnet18(S = 7, B = 2, C = 20)
    output = model(torch.randn(64, 3, 224, 224))
    assert output.shape == torch.Size((64, 7, 7, 2, 25)), "YOLOv1Resnet18 output shape is different"
    assert sum(p.numel() for p in model.parameters()) == 123978706, "YOLOv1Resnet18 model parameter number is different"

    model = YOLOv1Resnet18(S = 6, B = 3, C = 15)
    output = model(torch.randn(64, 3, 224, 224))
    assert output.shape == torch.Size((64, 6, 6, 3, 20)), "YOLOv1Resnet18 output shape is different"
    assert sum(p.numel() for p in model.parameters()) == 95527600, "YOLOv1Resnet18 model parameter number is different"
