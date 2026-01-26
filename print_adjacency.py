import os
import torch
import matplotlib.pyplot as plt
from torcheeg.models import DGCNN


# Visualize as a circular connectivity graph
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
import numpy as np


os.makedirs('plots', exist_ok=True)
model_type = ''

# Load model
model = DGCNN(in_channels=5, num_electrodes=62, hid_channels=32, num_layers=2, num_classes=3)
model.load_state_dict(torch.load(f'ckpts/dgcnn_seed_model{model_type}.pth', map_location='cpu'))

# Get learned adjacency matrix
A = model.A.detach().numpy()

print("Adjacency Matrix Shape:", A.shape)

# check if its symmetric
is_symmetric = np.allclose(A, A.T)
print("Is the adjacency matrix symmetric?", is_symmetric)


# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(A, cmap='coolwarm', aspect='equal')
plt.colorbar(label='Weight')
plt.title('Learned Adjacency Matrix (62 Electrodes)')
plt.xlabel('Electrode')
plt.ylabel('Electrode')
plt.tight_layout()
plt.savefig('plots/adjacency_matrix.png', dpi=150)
plt.show()


# Define electrode names for 62-channel SEED dataset
electrode_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]

# Group electrodes by region for better visualization
# Left hemisphere electrodes
lh_electrodes = [name for name in electrode_names if name.endswith('1') or
                 name.endswith('3') or name.endswith('5') or name.endswith('7')]

# Midline electrodes
mid_electrodes = [name for name in electrode_names if name.endswith('Z')]

# Right hemisphere electrodes
rh_electrodes = [name for name in electrode_names if name.endswith('2') or
                 name.endswith('4') or name.endswith('6') or name.endswith('8')]

# Create node order: left -> midline -> right (for better visual grouping)
node_order = lh_electrodes + mid_electrodes + rh_electrodes

# Create circular layout with group boundaries
node_angles = circular_layout(
    electrode_names,
    node_order,
    start_pos=90,
    group_boundaries=[0, len(lh_electrodes), len(lh_electrodes) + len(mid_electrodes)]
)

# Assign colors by region
node_colors = []
for name in electrode_names:
    if name in lh_electrodes:
        node_colors.append('steelblue')
    elif name in mid_electrodes:
        node_colors.append('green')
    else:
        node_colors.append('crimson')

# Show only the strongest connections (top N)
n_lines = 30 

fig, ax = plt.subplots(figsize=(10, 10), facecolor='black', subplot_kw=dict(polar=True))
plot_connectivity_circle(
    A,
    electrode_names,
    n_lines=n_lines,
    node_angles=node_angles,
    node_colors=node_colors,
    title='DGCNN Learned Connectivity (62 Electrodes)',
    ax=ax,
    colormap='coolwarm',
    fontsize_names=8,
    fontsize_title=12,
    padding=2.0,
    show=True
)

fig.savefig(f'plots/adjacency_circular_{model_type}.png', dpi=150, facecolor='black')
