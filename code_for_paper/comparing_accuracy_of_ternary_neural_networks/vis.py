import csv
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
from code_for_paper.comparing_accuracy_of_ternary_neural_networks import bitLinear, ternaries
from plinear.plinear import PLinear, PF
from PIL import Image
from torch import nn

def pad_to_square(arr, pad_value=0):
                h, w = arr.shape[:2]
                max_dim = max(h, w)
                pad_height = max_dim - h
                pad_width = max_dim - w
                padded_arr = np.pad(
                    arr,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode='constant',
                    constant_values=pad_value
                )
                return padded_arr

def save_parameters(model, dname, width, depth, model_name):
    for name, module in model.named_modules():
        if isinstance(module, PLinear):
            rp = PF.posNet(module.real_pos.weight).cpu().numpy()
            rn = PF.posNet(module.real_neg.weight).cpu().numpy()
            real = rp - rn
            categories = np.full(real.shape, 0, dtype=int)
            categories[real == 1.0] = 1
            categories[real == 0.0] = 2
            categories[real == -1.0] = 3
            color_map = np.array([[0, 0, 0], [0, 0, 240], [240, 240, 240], [240, 0, 0]], dtype=np.uint8)
            
            image = color_map[categories]
            img = Image.fromarray(image)

            path = f"results/{dname}/depth_{depth}_width_{width}/weights"
            filepath = f"{path}/{model_name}_{name}.png"
            os.makedirs(path, exist_ok=True)
            img.save(filepath, format = 'PNG', optimize = True)

        elif isinstance(module, ternaries.ternary):
            tern_w = module.weight
            tern_w_np = module.tern(tern_w).cpu().numpy()

            categories = np.full(tern_w_np.shape, 0, dtype=int)
            categories[tern_w_np == 1.0] = 1
            categories[tern_w_np == 0.0] = 2
            categories[tern_w_np == -1.0] = 3
            
            color_map = np.array([[0, 0, 0], [0, 0, 240], [240, 240, 240], [240, 0, 0]], dtype=np.uint8)
            
            image = color_map[categories]
            img = Image.fromarray(image)
            path = f"results/{dname}/depth_{depth}_width_{width}/weights"
            filepath = f"{path}/{model_name}_{name}.png"
            os.makedirs(path, exist_ok=True)
            img.save(filepath, format = 'PNG', optimize = True)

        elif isinstance(module, nn.Linear):
            w = module.weight.cpu().numpy()

            max_abs_val = np.abs(w).max()
            if max_abs_val == 0:
                 max_abs_val = 1

            negative_mask = w < 0
            positive_mask = w > 0

            color_map = np.full((w.shape[0], w.shape[1], 3), [240, 240, 240], dtype=np.uint8)
            color_map[negative_mask, 0] = ((-w[negative_mask] / max_abs_val) * 240).astype(np.uint8)
            color_map[negative_mask, 1] = 0
            color_map[positive_mask, 2] = ((w[positive_mask] / max_abs_val) * 240).astype(np.uint8)
            color_map[positive_mask, 1] = 0

            img = Image.fromarray(color_map)
            path = f"results/{dname}/depth_{depth}_width_{width}/weights"
            filepath = f"{path}/{model_name}_{name}.png"
            os.makedirs(path, exist_ok=True)
            img.save(filepath, format = 'PNG', optimize = True)


            
             

def plot_from_csv(dname, width, depth, tag):
    path = f"results/{dname}"
    filepath = f"{path}/{tag}/depth_{depth}_width_{width}.png"
    os.makedirs(f"{path}/{tag}", exist_ok=True)
    df = pd.read_csv(f"{path}/depth_{depth}_width_{width}/csv/{tag}.csv")
    plt.figure(figsize=(12, 6))
    
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        plt.plot(model_data['Epoch'], model_data['data'], label=f"{model_name}")
    
    plt.title(f"{dname}_depth : {depth}_width : {width}, {tag}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{tag}")
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, -0.05), borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Plot saved as {filepath}")
    plt.close()

def save_results_to_csv(dname, model_name, width, depth, data, tag):
    path = f"results/{dname}/depth_{depth}_width_{width}/csv"
    filepath = f"{path}/{tag}.csv"
    os.makedirs(path, exist_ok=True)
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Epoch', 'data'])
        for epoch, v in enumerate(data):
            writer.writerow([model_name, epoch + 1, v])