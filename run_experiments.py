import subprocess
import argparse
import os
import time


def run_experiments(max_procs, N, steps, T, repeats, skip):
    metrics_file = 'data/metrics.csv'
    data_file = 'data/simulation_data.csv'
    
    # Clean previous metrics and data
    if os.path.exists(metrics_file):
        os.remove(metrics_file)
        print(f"Removed old {metrics_file}")

    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"Removed old {data_file}")

    print(f"Starting experiments with Grid N={N}, Time T={T}")

    procs_list = range(1, max_procs + 1)

    for p in procs_list:
        print(f"--- Running with {p} processors ---")
        # Don't save data for performance runs efficiently, or maybe only for seq?
        # User wants speedup, so I/O should be minimized or consistent. 
        # If I want to measure calc speed, I should probably disable heavy I/O frequency.
        # But for 'simulation_data.csv', if I overwrite it every time, I only keep the last run.
        # That's fine for animation (we only need one).

        for r in range(repeats):

            cmd = [
                "mpirun", "-n", str(p),
                "python3", "src/solver.py",
                "--N", str(N),
                "--T", str(T),
                "--mu", "0.001",
                "--out", data_file,
                "--metrics", metrics_file,
                "--skip", str(skip)
            ]

            # Save data only for sequential run (p=1) to verify correctness
            if p == 1 and r == 0:
                cmd.append("--save")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running with {p} procs: {e}")
                break

    print("Experiments completed.")
    print("Generating metrics plot...")

    subprocess.run(["python3", "src/metrics.py", "--file", metrics_file])

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_procs', type=int, default=4)
    parser.add_argument('--N', type=int, default=2000, help="Grid size (larger N better for showing speedup)")
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--repeats', type=int, default=5, help="Number of repetitions")
    parser.add_argument('--skip', type=int, default=100, help="Save interval")
    args = parser.parse_args()

    run_experiments(args.max_procs, args.N, 0, args.T, args.repeats, args.skip)