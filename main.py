import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class BurgersAnimationWithTime:
    def __init__(self, N=200, mu=0.01, T_final=2.5, dt=0.005, w=0.5):
        self.N = N
        self.mu = mu
        self.T_final = T_final
        self.dt = dt
        self.w = w
        
        # Grid setup
        self.dx = 1.0 / N
        self.x = np.linspace(0, 1.0 - self.dx, N)
        self.r = (self.mu * self.dt) / (self.dx ** 2)
        
        # Build Periodic Implicit Matrix A
        main_diag = (1 + 2 * self.r) * np.ones(N)
        off_diag = -self.r * np.ones(N - 1)
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N), format='lil')
        A[0, N-1] = -self.r
        A[N-1, 0] = -self.r
        self.A = A.tocsr()

    def initial_condition(self, x):
        # "Higher beginning" (0.5) + "One peak" (Gaussian)
        return 0.5 + 1.0 * np.exp(-((x - 0.2)**2) / (2 * 0.05**2))

    def compute_rhs(self, u):
        lam = self.dt / self.dx
        u_padded = np.pad(u, (2, 2), mode='wrap')
        b = np.zeros(self.N)
        
        for i in range(self.N):
            k = i + 2
            F_ip1 = 0.5 * u_padded[k+1]**2
            F_im1 = 0.5 * u_padded[k-1]**2
            
            # Rusanov Dissipation (phi)
            phi_ip1 = 0.5 * np.abs(u_padded[k+1] + u_padded[k]) * (u_padded[k+1] - u_padded[k])
            phi_im1 = 0.5 * np.abs(u_padded[k-1] + u_padded[k-2]) * (u_padded[k-1] - u_padded[k-2])
            
            b[i] = u_padded[k] - 0.5*lam*(F_ip1 - F_im1) + 0.5*self.w*lam*(phi_ip1 - phi_im1)
        return b

    def run_simulation(self):
        u = self.initial_condition(self.x)
        self.frames = [u.copy()]
        self.timestamps = [0.0]
        t = 0
        while t < self.T_final:
            b = self.compute_rhs(u)
            u = spsolve(self.A, b)
            t += self.dt
            self.frames.append(u.copy())
            self.timestamps.append(t)
        return self.frames, self.timestamps

    def save_animation(self, filename='burgers_with_time.mp4'):
        print("Solving equations...")
        data, times = self.run_simulation()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot(self.x, data[0], color='#1f77b4', lw=2)
        
        # Add a text label for the time 't' in the upper right corner
        time_text = ax.text(0.8, 1.8, '', fontsize=12, 
                            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        
        ax.set_ylim(0, 2.0)
        ax.set_xlim(0, 1.0)
        ax.set_title("Burgers' Equation: Rusanov Method (Periodic)")
        ax.set_xlabel("Spatial coordinate (x)")
        ax.set_ylabel("Velocity (u)")
        ax.grid(True, linestyle='--', alpha=0.6)

        def update(frame):
            line.set_ydata(data[frame])
            # Update the text string for the current time step
            time_text.set_text(f"t = {times[frame]:.3f}")
            return line, time_text

        print(f"Rendering {len(data)} frames...")
        ani = FuncAnimation(fig, update, frames=len(data), interval=20, blit=True)
        
        try:
            ani.save(filename, writer='ffmpeg', fps=30)
            print(f"Animation saved as {filename}")
        except Exception as e:
            print("FFMPEG not found. Saving as GIF instead.")
            ani.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=30)
        
        plt.close()

if __name__ == "__main__":
    sim = BurgersAnimationWithTime()
    sim.save_animation()