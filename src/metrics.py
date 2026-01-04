import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def calculate_and_plot_metrics(metrics_file, N_grid_filter=None):
    if not os.path.exists(metrics_file):
        print(f"File {metrics_file} not found.")
        return

    df = pd.read_csv(metrics_file)
    
    # Filter by grid size if specified, otherwise take the most common or ask
    if N_grid_filter:
        df = df[df['N_grid'] == N_grid_filter]
    else:
        # If multiple grid sizes exist, pick the most frequent one to analyze
        if df['N_grid'].nunique() > 1:
            print("Multiple grid sizes found. Analyzing the one with most entries.")
            most_common_N = df['N_grid'].mode()[0]
            df = df[df['N_grid'] == most_common_N]
            print(f"Selected N_grid = {most_common_N}")
    
    # Sort by N_proc
    df = df.sort_values('N_proc')
    
    # Group by N_proc and take mean time (in case multiple runs)
    df_grouped = df.groupby('N_proc')['Total_Time'].mean().reset_index()
    
    # Find Sequential Time (T1)
    if 1 in df_grouped['N_proc'].values:
        T1 = df_grouped[df_grouped['N_proc'] == 1]['Total_Time'].values[0]
    else:
        print("No sequential run (N_proc=1) found. Cannot calculate Speedup/Efficiency correctly.")
        T1 = df_grouped['Total_Time'].max() # Fallback estimate (usually wrong but prevents crash)
        print(f"Assuming T1 = {T1} (Slowest run) for demonstration.")

    # Calculate Metrics
    df_grouped['Speedup'] = T1 / df_grouped['Total_Time']
    df_grouped['Efficiency'] = df_grouped['Speedup'] / df_grouped['N_proc']
    
    # Karp-Flatt: e = (1/S - 1/p) / (1 - 1/p)
    # Only valid for p > 1
    def karp_flatt(row):
        p = row['N_proc']
        s = row['Speedup']
        if p == 1:
            return 0.0
        return ((1/s) - (1/p)) / (1 - (1/p))
    
    df_grouped['Karp_Flatt'] = df_grouped.apply(karp_flatt, axis=1)

    print("\nCalculated Metrics:")
    print(df_grouped)
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Speedup
    axs[0].plot(df_grouped['N_proc'], df_grouped['Speedup'], marker='o', label='Measured')
    axs[0].plot(df_grouped['N_proc'], df_grouped['N_proc'], 'k--', label='Ideal Linear')
    axs[0].set_title('Speedup')
    axs[0].set_xlabel('Number of Processors')
    axs[0].set_ylabel('Speedup')
    axs[0].legend()
    axs[0].grid(True)
    
    # Efficiency
    axs[1].plot(df_grouped['N_proc'], df_grouped['Efficiency'], marker='o', color='g')
    axs[1].set_title('Efficiency')
    axs[1].set_xlabel('Number of Processors')
    axs[1].set_ylabel('Efficiency')
    axs[1].set_ylim(0, 1.1)
    axs[1].grid(True)
    
    # Karp-Flatt
    axs[2].plot(df_grouped['N_proc'][1:], df_grouped['Karp_Flatt'][1:], marker='o', color='r')
    axs[2].set_title('Karp-Flatt Metric')
    axs[2].set_xlabel('Number of Processors')
    axs[2].set_ylabel('e')
    axs[2].grid(True)
    
    plt.tight_layout()
    output_img = 'metrics_plot.png'
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/metrics.csv')
    parser.add_argument('--N', type=int, help="Grid size to filter by")
    args = parser.parse_args()
    
    calculate_and_plot_metrics(args.file, args.N)
