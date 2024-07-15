import torch
import pytest
from plinear import PLinear
import torch.nn as nn


def test_basic_forward():
    model = PLinear(10, 5)
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 5), "Output shape mismatch"

def test_basic_zero_input():
    model = PLinear(10, 5)
    x = torch.zeros(1, 10)
    output = model(x)
    assert torch.all(output == 0), "Output should be all zeros for zero input"

def test_basic_weights_update():
    model = PLinear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    x = torch.randn(1, 10)
    target = torch.randn(1, 5)
    output = model(x)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # 단순히 학습이 진행되는지 확인 (기본적인 가중치 업데이트)
    for param in model.parameters():
        assert param.grad is not None, "Gradient should not be None after backward"

def test_basic_edge_case():
    model = PLinear(10, 5)
    x = torch.randn(1, 10) * 1e6  # 아주 큰 값의 입력
    output = model(x)
    assert output.shape == (1, 5), "Output shape mismatch with edge case input"

# pytest를 사용하여 basic 키워드로 필터링
if __name__ == "__main__":
    pytest.main(["-k", "basic"])
