import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from plinear.plinear import PF
import os
import imageio

def save_weight_image(layer, result_path, layer_index, epoch):
    layer_path = f"{result_path}/layer_{layer_index+1}"
    os.makedirs(layer_path, exist_ok=True)
    
    pos_weights = PF.posNet(layer.linear_pos.weight).cpu().numpy()
    neg_weights = PF.posNet(layer.linear_neg.weight).cpu().numpy()
    combined_weights = pos_weights - neg_weights
    reshaped_weights = combined_weights.reshape(layer.linear_pos.weight.shape[0], -1)

    plt.imshow(reshaped_weights, cmap='bwr', vmin=-1, vmax=1, aspect='auto')
    plt.title(f'Combined Weights - Layer {layer_index+1} - Epoch {epoch+1}')
    plt.savefig(f"{layer_path}/epoch_{epoch+1}.png")
    plt.close()

def create_animation_from_images(result_path, layer_index, num_epochs):
    layer_path = f"{result_path}/layer_{layer_index+1}"
    images = []
    for epoch in range(num_epochs):
        filename = f"{layer_path}/epoch_{epoch+1}.png"
        images.append(imageio.imread(filename))
    imageio.mimsave(f"{result_path}/weights_viz_layer_{layer_index+1}.gif", images, duration=10000/num_epochs)

def create_confusion_matrix_animation(confusion_matrices, result_path, num_epochs):
    fig, ax = plt.subplots(figsize=(10, 7))

    def update_confusion_matrix(epoch):
        ax.clear()
        cm = confusion_matrices[epoch]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
        tick_marks = np.arange(10)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

    ani = FuncAnimation(fig, update_confusion_matrix, frames=np.arange(num_epochs), repeat = False)
    writer=PillowWriter(fps=num_epochs / 10)
    ani.save(result_path + "confusion_matrix.gif", writer=writer)
    plt.close(fig)

def plot_metrics(epochs, accuracy_list, recall_list, precision_list, result_path):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy_list, label='Accuracy')
    plt.plot(epochs, recall_list, label='Recall')
    plt.plot(epochs, precision_list, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Metrics over Epochs')
    plt.legend()
    plt.savefig(result_path + "metrics_over_epochs.png")
    plt.close()
