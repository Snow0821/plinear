import numpy as np
import matplotlib.pyplot as plt
from plinear.plinear import PF
import os
import imageio

def save_image(data, result_path, filename, cmap='bwr', vmin=-1, vmax=1, aspect='auto', title=None):
    os.makedirs(result_path, exist_ok=True)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join(result_path, filename))
    plt.close()

def save_weight_images(model, result_path, epoch):
    for i, layer in enumerate(model.children()):
        if hasattr(layer, 'linear_pos') and hasattr(layer, 'linear_neg'):
            pos_weights = PF.posNet(layer.linear_pos.weight).cpu().numpy()
            neg_weights = PF.posNet(layer.linear_neg.weight).cpu().numpy()
            combined_weights = (pos_weights - neg_weights).reshape(layer.linear_pos.weight.shape[0], -1)
            save_image(combined_weights, f"{result_path}/layer_{i+1}", f"{epoch+1}_Real.png", title=f'Real Weights - Layer {i+1} - Epoch {epoch+1}')

        if hasattr(layer, 'real_pos') and hasattr(layer, 'real_neg'):
            rp = PF.posNet(layer.real_pos.weight).cpu().numpy()
            rn = PF.posNet(layer.real_neg.weight).cpu().numpy()
            real = (rp - rn).reshape(layer.real_pos.weight.shape[0], -1)
            save_image(real, f"{result_path}/layer_{i+1}", f"{epoch+1}_Real.png", title=f'Real Weights - Layer {i+1} - Epoch {epoch+1}')

        if hasattr(layer, 'complex_pos') and hasattr(layer, 'complex_neg'):
            cp = PF.posNet(layer.complex_pos.weight).cpu().numpy()
            cn = PF.posNet(layer.complex_neg.weight).cpu().numpy()
            complex = (cp - cn).reshape(layer.real_pos.weight.shape[0], -1)
            save_image(complex, f"{result_path}/layer_{i+1}", f"{epoch+1}_Complex.png", title=f'Complex Weights - Layer {i+1} - Epoch {epoch+1}')
    return

def save_confusion_matrix(cm, result_path, epoch):
    save_image(cm, result_path+'/cm', f"{epoch+1}.png", cmap='Blues', vmin=0, vmax=np.max(cm), title=f'Confusion Matrix - Epoch {epoch+1}')

def create_animation_from_images(path, folder, num_epochs, postfix = ''):
    images = []
    for epoch in range(num_epochs):
        filename = f"{path}/{folder}/{epoch+1}{'_' if postfix else ''}{postfix}.png"
        images.append(imageio.imread(filename))
    imageio.mimsave(f"{path}/{folder}{'_' if postfix else ''}{postfix}.gif", images, duration=100)

import csv

def plot_metrics(epochs, accuracy_list, recall_list, precision_list, result_path):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy_list, label='Accuracy')
    plt.plot(epochs, recall_list, label='Recall')
    plt.plot(epochs, precision_list, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Metrics over Epochs')
    plt.legend()
    plt.savefig(os.path.join(result_path, "metrics_over_epochs.png"))
    plt.close()

    with open(os.path.join(result_path, "metrics_table.csv"), "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epoch", "Accuracy", "Recall", "Precision"])
        for epoch in range(len(epochs)):
            csvwriter.writerow([epochs[epoch], accuracy_list[epoch], recall_list[epoch], precision_list[epoch]])

