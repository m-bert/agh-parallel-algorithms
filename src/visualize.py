import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import argparse

def create_animation(data_file, output_file, fps, max_duration):
    print(f"Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file, header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Extract Time and Data
    raw_times = df.iloc[:, 0].values
    raw_data = df.iloc[:, 1:].values
    
    total_frames = len(raw_times)
    N = raw_data.shape[1]
    
    # Automatic subsampling to keep animation short
    target_frames = int(max_duration * fps)
    stride = max(1, total_frames // target_frames)
    
    times = raw_times[::stride]
    data = raw_data[::stride]
    
    print(f"Loaded {total_frames} frames. Subsampling by stride {stride} to {len(times)} frames.")
    print(f"Estimated duration: {len(times)/fps:.2f} seconds.")
    
    x = np.linspace(0, 1.0 - 1.0/N, N)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(x, data[0], color='#1f77b4', lw=2)
    
    time_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, fontsize=12, 
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    
    ax.set_ylim(0, 2.0)
    ax.set_xlim(0, 1.0)
    ax.set_title("Burgers' Equation (Parallel Output)")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    def update(frame):
        line.set_ydata(data[frame])
        time_text.set_text(f"t = {times[frame]:.3f}")
        return line, time_text
    
    # interval is ms between frames. 1000/fps
    interval = 1000 / fps
    ani = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)
    
    try:
        print(f"Saving animation to {output_file}...")
        ani.save(output_file, writer='ffmpeg', fps=fps)
        print("Done.")
    except Exception as e:
        print(f"FFMPEG failed ({e}), trying .gif")
        ani.save(output_file.replace('.mp4', '.gif'), writer='pillow', fps=fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/simulation_data.csv')
    parser.add_argument('--out', type=str, default='animation.mp4')
    parser.add_argument('--fps', type=int, default=30, help="Frames per second")
    parser.add_argument('--duration', type=float, default=10.0, help="Target max duration in seconds")
    args = parser.parse_args()
    
    create_animation(args.file, args.out, args.fps, args.duration)
