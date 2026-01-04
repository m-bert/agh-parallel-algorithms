import numpy as np
import time
import argparse
import csv
from mpi4py import MPI
import os

class ParallelBurgersSolver:
    def __init__(self, N, mu, T, dt, output_file, metrics_file, save_interval, save_data=False):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.N_global = N
        self.mu = mu
        self.T_final = T
        self.dt = dt
        self.output_file = output_file
        self.metrics_file = metrics_file
        self.save_interval = save_interval
        self.save_data = save_data
        
        # Domain Decomposition
        # Calculate local size and offset
        self.n_local = self.N_global // self.size
        remainder = self.N_global % self.size
        
        if self.rank < remainder:
            self.n_local += 1
            self.start_idx = self.rank * self.n_local
        else:
            self.offset_bias = remainder * (self.n_local + 1)
            self.start_idx = self.offset_bias + (self.rank - remainder) * self.n_local
            
        self.end_idx = self.start_idx + self.n_local
        
        self.dx = 1.0 / self.N_global
        
        # Determine neighbors
        self.left_neighbor = (self.rank - 1) % self.size
        self.right_neighbor = (self.rank + 1) % self.size
        
        # Initialize field
        self.u_local = np.zeros(self.n_local)
        self.initialize_field()
        
        # Buffers for ghost cells (padding of 2)
        self.halo_width = 2
        
    def initial_condition(self, x_global_indices):
        # Gaussian peak initial condition (matches main.py)
        # u(x,0) = 0.5 + 1.0 * exp(-(x - 0.2)^2 / (2 * 0.05^2))
        x = x_global_indices * self.dx
        return 0.5 + 1.0 * np.exp(-((x - 0.2)**2) / (2 * 0.05**2))

    def initialize_field(self):
        global_indices = np.arange(self.start_idx, self.end_idx)
        self.u_local = self.initial_condition(global_indices)

    def exchange_halos(self):
        # Periodic Boundary Conditions
        # Logic:
        # - Left Ghost receives from Left Neighbor's Right Boundary
        # - Right Ghost receives from Right Neighbor's Left Boundary
        # Because of the Ring Topology (rank-1, rank+1 modulo size),
        # this effectively implements P.B.C. globally.
        
        # Send to left, receive from right
        # Send first 2 elements to left helper
        # Recv 2 elements from right into right ghost
        
        # Send to right, receive from left
        # Send last 2 elements to right neighbor
        # Recv 2 elements from left into left ghost

        # We need a buffer that is n_local + 4 (2 left, 2 right)
        # u is currently n_local.
        
        # Buffers
        send_left = self.u_local[:self.halo_width].copy()
        send_right = self.u_local[-self.halo_width:].copy()
        
        recv_left = np.empty(self.halo_width, dtype=np.float64)
        recv_right = np.empty(self.halo_width, dtype=np.float64)
        
        # Periodic Boundary conditions handled by neighbor logic (modulo arithmetic)
        
        reqs = []
        # Non-blocking sends/recvs slightly better, 
        # But Sendrecv is easier to write correctly for ring.
        
        # 1. Send Left, Recv Right
        self.comm.Sendrecv(sendbuf=send_left, dest=self.left_neighbor,
                           recvbuf=recv_right, source=self.right_neighbor)
                           
        # 2. Send Right, Recv Left
        self.comm.Sendrecv(sendbuf=send_right, dest=self.right_neighbor, 
                           recvbuf=recv_left, source=self.left_neighbor)
        
        return recv_left, recv_right

    def compute_rhs_explicit(self, u_padded):
        # u_padded includes ghosts: [L0, L1, U0, ..., Un-1, R0, R1]
        # Length = 2 + n_local + 2
        
        # We compute update for indices corresponding to U0...Un-1
        # Which are indices 2 ... 2+n_local-1 in u_padded
        
        n = self.n_local
        du = np.zeros(n)
        lam = self.dt / self.dx
        w = 0.5 # constant from main.py
        
        # Precompute commonly used values or vectorise 
        # Vectorized version for speed inside the loop
        
        # Slice for the core domain
        # i goes from 0 to n-1 (local)
        # k corresponds to u_padded index. k = i + 2.
        
        # Slices in u_padded:
        # k range: 2 to n+1
        # k+1 range: 3 to n+2
        # k-1 range: 1 to n
        # k+2 range: 4 to n+3 ... wait, max index is n+3 (length is n+4). correct.
        
        u_k   = u_padded[2 : n+2]
        u_kp1 = u_padded[3 : n+3]
        u_km1 = u_padded[1 : n+1]
        u_km2 = u_padded[0 : n]
        
        # 1. Advection (Rusanov)
        F_ip1 = 0.5 * u_kp1**2
        F_im1 = 0.5 * u_km1**2
        
        phi_ip1 = 0.5 * np.abs(u_kp1 + u_k) * (u_kp1 - u_k)
        phi_im1 = 0.5 * np.abs(u_km1 + u_km2) * (u_km1 - u_km2) # Corrected logic to match main.py stencil offset
        # In main.py:
        # phi_im1 = 0.5 * np.abs(u[k-1] + u[k-2]) * (u[k-1] - u[k-2])
        # My u_km1 matches u[k-1], u_km2 matches u[k-2]. Correct.
        
        advection = -0.5*lam*(F_ip1 - F_im1) + 0.5*w*lam*(phi_ip1 - phi_im1) # This is actually 'b' contribution in main.py
        # Wait, compute_rhs in main.py returns 'b'. 
        # b[i] = u[k] - ... 
        # So it returns u_new if solved explicitly? No.
        # main.py: b = u - adv_term.
        # Implicit eq: (1 - diff) u_new = b
        # If I want explicit: u_new = u + adv_term_explicit + diff_term_explicit
        
        # Let's verify 'advection' sign.
        # Eq: u_t + 0.5(u^2)_x = mu u_xx
        # u^{n+1} = u^n - dt * (0.5 u^2)_x + ...
        # main.py 'b' has -0.5*lam*(...) 
        # Yes, so 'advection' calculated here represents dt * (- (u^2/2)_x + dissipation).
        
        # 2. Diffusion (Standard Central Difference)
        # mu * u_xx
        # u_xx ~ (u_{i+1} - 2u_i + u_{i-1}) / dx^2
        diffusion = self.mu * (u_kp1 - 2*u_k + u_km1) / (self.dx**2)
        
        du = advection + self.dt * diffusion
        
        return du


    def solve(self):
        # Stability Check
        # Diffusion Requirement: dt <= dx^2 / (2*mu)
        limit_diff = (self.dx ** 2) / (2 * self.mu)
        
        # Advection CFL: dt <= dx / max_u (approx max_u ~ 2.0 based on init cond)
        limit_adv = self.dx / 2.0
        
        limit = min(limit_diff, limit_adv)
        
        if self.dt > limit:
            if self.rank == 0:
                print(f"Warning: dt={self.dt} is unstable. dx={self.dx}, mu={self.mu}.")
                print(f"Adjusting dt to {limit} for stability.")
            self.dt = limit
            
        t = 0.0
        step_count = 0
        
        # Recalculate lambdas with new dt
        # Note: In compute_rhs_explicit, we use self.dt directly.
        # But we should ensure consistency if other params depended on specific dt.
        # Here logic uses self.dt in the loop or method.
        
        frames = []
        timestamps = []
        
        # Pre-allocate padded array to avoid reallocation in loop
        # [GhostL (2), Interior (n_local), GhostR (2)]
        self.u_padded = np.zeros(self.n_local + 2 * self.halo_width)

        # Log initial state
        if self.save_data:
            self.gather_and_save(t, 0)
        
        # Time Loop
        start_time = MPI.Wtime()
        
        while t < self.T_final:
            t += self.dt
            step_count += 1
            
            # 1. Exchange
            ghost_L, ghost_R = self.exchange_halos()
            
            # 2. Update padded array in-place
            # Middle: Interior
            self.u_padded[self.halo_width : -self.halo_width] = self.u_local
            # Left Ghost
            self.u_padded[:self.halo_width] = ghost_L
            # Right Ghost
            self.u_padded[-self.halo_width:] = ghost_R
            
            # 3. Compute Update
            # Pass pre-allocated buffer
            delta_u = self.compute_rhs_explicit(self.u_padded)
            
            # 4. Update
            self.u_local += delta_u
            
            # 5. Save/Log
            if self.save_data and step_count % self.save_interval == 0:
                self.gather_and_save(t, step_count)
                
        end_time = MPI.Wtime()
        total_time = end_time - start_time
        
        # Save metrics
        if self.rank == 0:
            self.save_metrics(total_time, step_count)
            print(f"Simulation finished in {total_time:.4f}s on {self.size} processors.")

    def gather_and_save(self, t, step):
        # Gather all u_local to root
        # Using Gather (blocking) for simplicity. 
        # For huge runs, you'd write MPI-IO or per-rank files.
        # But user wants csv with speedup etc, so simple gather is fine for N~1000.
        
        # Gather sizes first (technically static, but good for robustness)
        # recvcounts = self.comm.gather(len(self.u_local), root=0)
        
        # Gatherv is needed because n_local might vary (load balancing reminders)
        
        recvbuf = None
        if self.rank == 0:
            recvbuf = np.empty(self.N_global, dtype=np.float64)
            # Calculate counts and displacements
            counts = [self.N_global // self.size + (1 if i < (self.N_global % self.size) else 0) for i in range(self.size)]
            # check correctness
            displs = [0] + list(np.cumsum(counts)[:-1])
            recvbuf_args = [recvbuf, counts, displs, MPI.DOUBLE]
        else:
            recvbuf_args = None
            
        self.comm.Gatherv(self.u_local, recvbuf_args, root=0)
        
        if self.rank == 0:
            # Append to Data CSV
            # Format: t, u0, u1, ...
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [t] + list(recvbuf)
                writer.writerow(row)

    def save_metrics(self, duration, steps):
        # Append to Metrics CSV
        # N_proc, N_grid, Time, Steps, TimePerStep
        # Check if file exists to write header
        file_exists = os.path.isfile(self.metrics_file)
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['N_proc', 'N_grid', 'Total_Time', 'Steps', 'Time_Per_Step'])
            writer.writerow([self.size, self.N_global, duration, steps, duration/steps if steps > 0 else 0])

def main():
    parser = argparse.ArgumentParser(description="Parallel Burgers Solver")
    parser.add_argument('--N', type=int, default=200, help="Grid points")
    parser.add_argument('--mu', type=float, default=0.01, help="Viscosity")
    parser.add_argument('--T', type=float, default=2.5, help="Final Time")
    parser.add_argument('--dt', type=float, default=0.001, help="Time Step")
    parser.add_argument('--out', type=str, default='data/simulation_data.csv', help="Output data file")
    parser.add_argument('--metrics', type=str, default='data/metrics.csv', help="Metrics file")
    parser.add_argument('--skip', type=int, default=100, help="Save interval (default: 100)")
    parser.add_argument('--save', action='store_true', help="Enable saving simulation data")
    
    args = parser.parse_args()
    
    # Ensure data directory exists (only rank 0 needs to check really, but safe for all)
    if not os.path.exists(os.path.dirname(args.out)):
        try:
            os.makedirs(os.path.dirname(args.out))
        except:
            pass
            
    solver = ParallelBurgersSolver(args.N, args.mu, args.T, args.dt, args.out, args.metrics, args.skip, args.save)
    
    # Sync before starting
    solver.comm.Barrier()
    solver.solve()

if __name__ == "__main__":
    main()
