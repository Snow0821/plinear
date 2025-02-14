import torch
import torch.nn as nn
import torch.nn.functional as F

def bin_to_sign(tensor):
    return tensor * 2 - 1

def sign_to_bin(tensor):
    return (tensor + 1) / 2

def gumbel_softmax(logits, temperature=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    # y_soft = F.softmax(y / temperature, dim=-1)
    
    # y_hard = torch.zeros_like(y_soft).scatter_(1, y_soft.argmax(dim=1, keepdim=True), 1.0)
    # y = torch.round(y_hard - y_soft.detach() + y_soft)
    y_hard = torch.zeros_like(y).scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
    y = torch.round(y_hard - y.detach() + y)
    return y

def check_ternary_tensor(tensor, tensor_name="tensor"):
    assert torch.all((tensor == -1) | (tensor == 1) | (tensor == 0)), f"{tensor_name} contains values other than -1 or 0 or 1! as {tensor}"

def check_binary_tensor(tensor, tensor_name="tensor"):
    assert torch.all((tensor == 0) | (tensor == 1)), f"{tensor_name} contains values other than 0 or 1! as {tensor}"

def my_softmax(y):
    out = torch.zeros_like(y).scatter_(1, y.argmax(dim =1, keepdim=True), 1.0)
    out = torch.round(out - y.detach() + y)
    return out

def random_topk(selector, k=5):
    """
    mask: (batch_size, num_classes) 크기의 가중치 텐서
    k: 선택할 top-k 개수
    """
    topk_values, topk_indices = selector.topk(k, dim=1)  # Top-k 선택
    random_choice = torch.randint(0, k, (selector.size(0), 1), device=selector.device)  # 각 배치에서 하나 랜덤 선택
    selected_indices = topk_indices.gather(1, random_choice)  # 선택된 인덱스 가져오기

    # 최종 마스크 생성 (0으로 초기화 후 선택된 인덱스에만 1)
    final_mask = torch.zeros_like(selector).scatter_(1, selected_indices, 1.0)

    return torch.round(final_mask - selector.detach() + selector)

def weighted_random(selector, temperature=1.0):
    """
    mask: (batch_size, num_classes) 크기의 가중치 텐서
    temperature: Softmax의 날카로움을 조절하는 파라미터 (낮을수록 최대값에 집중)
    """
    softmax_probs = F.softmax(selector / temperature, dim=1)  # Softmax로 확률 분포 생성
    selected_indices = torch.multinomial(softmax_probs, num_samples=1)  # 확률에 따라 하나 선택

    # 선택된 인덱스를 1로 설정한 마스크 생성
    final_mask = torch.zeros_like(selector).scatter_(1, selected_indices, 1.0)

    return final_mask - selector.detach() + selector


class SparseBtnn_Selector(nn.Module):
    def __init__(self, x, y):
        super(SparseBtnn_Selector, self).__init__()
        self.selector = nn.Linear(x, y, bias=False)
        torch.nn.init.xavier_uniform_(self.selector.weight)

    def forward(self, x):
        # check_binary_tensor(x, "input")
        selector = self.selector.weight
        masked = weighted_random(selector)
        out = F.linear(x, masked)
        # check_binary_tensor(out, "selector")
        return out

class Compacted_Nand(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.a = SparseBtnn_Selector(x, y)
        self.b = SparseBtnn_Selector(x, y)
    
    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        out = 1 - (a * b)
        # check_binary_tensor(out, "nand_result")
        return out

class SparseBtnn_And(nn.Module):
    def __init__(self, x, y):
        super(SparseBtnn_And, self).__init__()
        self.a = SparseBtnn_Selector(x, y)
        self.b = SparseBtnn_Selector(x, y)
    
    def forward(self, x):
        a = self.a(x)
        b = self.b(x)

        return a * b

class SparseBtnn_Not(nn.Module):
    def __init__(self, *dim):
        super(SparseBtnn_Not, self).__init__()
        self.a = nn.parameter.Parameter(torch.randn(*dim),)

    def forward(self, x):
        # a = self.a.expand(x.shape[-1], -1)
        a = self.a
        qa = (a > 0).float() - a.detach() + a
        # x = x.permute(1, 0)

        return qa + x - 2 * qa * x
    
class SparseBtnn_Nand(nn.Module):
    def __init__(self, x, y):
        super(SparseBtnn_Nand, self).__init__()
        self.a = SparseBtnn_And(x, y)
        self.n = SparseBtnn_Not(y)
    
    def forward(self, x):
        out = self.a(x)
        out = self.n(out)
        return out



class SparseBtnn_Nand_Multihead(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads):
        super(SparseBtnn_Nand_Multihead, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            SparseBtnn_Nand(input_dim, output_dim) for _ in range(n_heads)
        ])

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        output = torch.cat(head_outputs, dim=-1) 
        return output

class Conv2d_Selector(nn.Module):
    def __init__(self, x, y, kernel_size, stride=1, padding=0, groups = 1):
        super(Conv2d_Selector, self).__init__()
        self.conv = nn.Conv2d(x, y, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = sign_to_bin(x)
        conv = self.conv.weight

        selector = conv.view(conv.size(0), -1)
        mask = weighted_random(selector)
        mask = mask.view_as(conv)

        y = F.conv2d(
            x, 
            mask, 
            bias=None,               
            stride=self.conv.stride, 
            padding=self.conv.padding, 
            dilation=self.conv.dilation, 
            groups=self.conv.groups
        )
        y = bin_to_sign(y)
        check_binary_tensor(y, "conv2d selector")
        return y
    
class Conv2d_Nand(nn.Module):
    def __init__(self, x, y, kernel_size, stride=1, padding=0, groups = 1):
        super().__init__()
        self.a = Conv2d_Selector(x, y, kernel_size, stride=stride, padding=padding, groups = groups)
        self.b = Conv2d_Selector(x, y, kernel_size, stride=stride, padding=padding, groups = groups)
    
    def forward(self, x):
        a = sign_to_bin(self.a(x))
        b = sign_to_bin(self.b(x))
        out = bin_to_sign(1 - (a * b))
        check_binary_tensor(out, "conv2d Nand")
        return out

