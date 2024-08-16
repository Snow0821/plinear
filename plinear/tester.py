import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from plinear.vis import save_weight_images, create_animation_from_images, plot_metrics, save_confusion_matrix
import os
from plinear.plinear import PLinear_Complex as PL, PLinear

class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho value: {rho}"
        self.rho = rho
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # 파라미터를 로컬 최대치로 이동

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.base_optimizer.step()  # Sharpness-Aware 최적화 수행

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # closure가 전체 forward-backward 과정을 수행

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        # 모든 파라미터 그룹에 대해, p.grad가 None이 아닌 파라미터들의 리스트를 생성
        grads = [
            p.grad.norm(p=2).to(p.device)
            for group in self.param_groups for p in group['params']
            if p.grad is not None
        ]
        
        # 만약 grads 리스트가 비어있다면 (모든 그래디언트가 None이라면)
        if len(grads) == 0:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        
        # grads 리스트가 비어있지 않다면, norm 계산
        norm = torch.norm(torch.stack(grads), p=2)
        
        return norm

def ExampleTester(model, domains, num_epochs, path, train_loader, test_loader):
    def train(model, train_loader, criterion, optimizer):
        total_loss = 0
        total_samples = 0

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += inputs.size(0)
            loss.backward()
            optimizer.step()

        print(f"Loss : {total_loss / total_samples} ")
            # if using SAM
            # def closure():
            #     optimizer.zero_grad()
            #     outputs = model(inputs)
            #     loss = criterion(outputs, labels)
            #     loss.backward()
            #     return loss

            # optimizer.step(closure)  # Second Optimization of SAM

    def evaluation(model, test_loader, path, epoch):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                preds = output.argmax(dim=1, keepdim=True)
                all_preds.extend(preds.view(-1).cpu().tolist())
                all_labels.extend(target.view(-1).cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)

        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)

        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print()

        save_confusion_matrix(confusion_matrix(all_labels, all_preds), path, epoch)
        save_weight_images(model, path, epoch)
        
        return
    
    def save_animations(model, num_epochs, accuracy_list, recall_list, precision_list, path, domains):
        for i, layer in enumerate(model.children()):
            for domain in domains:
                if isinstance(layer, PL) or isinstance(layer, PLinear):
                    create_animation_from_images(path, f'layer_{i + 1}', num_epochs, domain)
        create_animation_from_images(path,'cm', num_epochs)

        # Plot accuracy, recall, precision over epochs and save to file
        epochs = range(1, num_epochs + 1)
        plot_metrics(epochs, accuracy_list, recall_list, precision_list, path)
        return
    os.makedirs(path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # base_optimizer = optim.Adam
    # optimizer = SAM(model.parameters(), base_optimizer, lr=1, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1)
    # optimizer = optim.SGD(model.parameters(), lr = 1)
    accuracy_list = []
    recall_list = []
    precision_list = []

    print("\nTraining Begin")
    # Training and evaluation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer)
        evaluation(model, test_loader, path, epoch)
    
    save_animations(model, num_epochs, accuracy_list, recall_list, precision_list, path, domains)

    # Assert final accuracy
    assert accuracy_list[-1] > 0.5, "Final accuracy should be higher than 50%"