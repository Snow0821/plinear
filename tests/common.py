import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from plinear.vis import save_weight_images, create_animation_from_images, plot_metrics, save_confusion_matrix
import os
from plinear.plinear import PLinear_Complex as PL, PLinear

def test_model(model, domains, num_epochs, path, train_loader, test_loader):
    def train(model, train_loader):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        return

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
        precision = precision_score(all_labels, all_preds, average='macro')

        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)  # Basic SGD optimizer

    accuracy_list = []
    recall_list = []
    precision_list = []

    # Training and evaluation loop
    for epoch in range(num_epochs):
        train(model, train_loader)
        evaluation(model, test_loader, path, epoch)
    
    save_animations(model, num_epochs, accuracy_list, recall_list, precision_list, path, domains)

    # Assert final accuracy
    assert accuracy_list[-1] > 0.5, "Final accuracy should be higher than 50%"