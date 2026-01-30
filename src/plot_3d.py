import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_3d_evolution(data_file, output_file):
    print(f"Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file, header=None)
    except Exception as e:
        print(f"Error: {e}")
        return

    raw_times = df.iloc[:, 0].values
    raw_data = df.iloc[:, 1:].values

    N_total = raw_data.shape[1]
    M_steps = raw_data.shape[0]

    print(f"Data loaded: {M_steps} time steps, {N_total} grid points.")

    target_N = 400
    stride_x = max(1, N_total // target_N)

    target_T = 200
    stride_t = max(1, M_steps // target_T)

    subset_data = raw_data[::stride_t, ::stride_x]
    subset_times = raw_times[::stride_t]

    x = np.linspace(0, 1.0, N_total)
    subset_x = x[::stride_x]

    print(f"Generating 3D mesh (reduction: X stride {stride_x}, T stride {stride_t})...")

    X_grid, T_grid = np.meshgrid(subset_x, subset_times)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X_grid, T_grid, subset_data, cmap='plasma',
                           edgecolor='none', alpha=0.9, antialiased=True)

    ax.set_title("Burgers' Equation (Space-Time Evolution)", fontsize=14)
    ax.set_xlabel("Space (x)", fontsize=11, labelpad=10)
    ax.set_ylabel("Time (t)", fontsize=11, labelpad=10)
    ax.set_zlabel("Velocity (u)", fontsize=11, labelpad=10)

    ax.view_init(elev=45, azim=240)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("Velocity u(x,t)", rotation=270, labelpad=15)

    print(f"Saving 3D plot to {output_file}...")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/simulation_data.csv')
    parser.add_argument('--out', type=str, default='wynik_3d_v2.png')
    args = parser.parse_args()

    plot_3d_evolution(args.file, args.out)