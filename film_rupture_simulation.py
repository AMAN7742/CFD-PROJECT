import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Parameters
# -----------------------------
C = 1.0
A = 1.0
B = 0.1
k = 1 / np.sqrt(2)
L = 2 * np.pi / k
N = 300           # Grid points (start with 300 for stability, increase later)
dx = L / N
dt = 1e-6         # Smaller time step for stability
T = 2.0
steps = int(T / dt)
x = np.linspace(0, L, N)

# -----------------------------
# Initial Condition
# -----------------------------
h = 1 + B * np.cos(k * x)
h_new = h.copy()

# -----------------------------
# Prepare Animation
# -----------------------------
frames = []
frame_interval = int(0.01 / dt)  # save every 0.01s

def compute_derivatives(h):
    h_x = np.zeros_like(h)
    h_xx = np.zeros_like(h)
    h_xxx = np.zeros_like(h)

    h_x[1:-1] = (h[2:] - h[:-2]) / (2 * dx)
    h_x[0] = h_x[1]
    h_x[-1] = h_x[-2]

    h_xx[1:-1] = (h[2:] - 2 * h[1:-1] + h[:-2]) / dx**2
    h_xx[0] = h_xx[1]
    h_xx[-1] = h_xx[-2]

    h_xxx[2:-2] = (h[4:] - 2*h[3:-1] + 2*h[1:-3] - h[0:-4]) / (2 * dx**3)
    h_xxx[0] = h_xxx[2]
    h_xxx[1] = h_xxx[2]
    h_xxx[-2] = h_xxx[-3]
    h_xxx[-1] = h_xxx[-3]

    return h_x, h_xx, h_xxx

# -----------------------------
# Time Loop
# -----------------------------
for step in range(steps):
    # Clip to avoid overflow
    h_safe = np.clip(h, 1e-6, 10)

    h_x, _, h_xxx = compute_derivatives(h_safe)

    flux1 = (h_safe**3) * h_xxx
    flux1_x = np.zeros_like(h)
    flux1_x[1:-1] = (flux1[2:] - flux1[:-2]) / (2 * dx)
    flux1_x[0] = flux1_x[1]
    flux1_x[-1] = flux1_x[-2]

    flux2 = h_x / h_safe
    flux2_x = np.zeros_like(h)
    flux2_x[1:-1] = (flux2[2:] - flux2[:-2]) / (2 * dx)
    flux2_x[0] = flux2_x[1]
    flux2_x[-1] = flux2_x[-2]

    # Time evolution (Euler)
    dhdt = -1/(3*C) * flux1_x - A * flux2_x
    h_new = h + dt * dhdt

    # Enforce boundary condition: flat at both ends
    h_new[0] = h_new[1]
    h_new[-1] = h_new[-2]

    # Prevent negative heights
    h_new = np.clip(h_new, 1e-6, 10)

    h = h_new.copy()

    # Save for animation
    if step % frame_interval == 0:
        frames.append(h.copy())

# -----------------------------
# Animation
# -----------------------------
fig, ax = plt.subplots()
line, = ax.plot(x, frames[0])
ax.set_ylim(0, 1)   # ✅ Y-axis limit fixed here
ax.set_xlabel('x')
ax.set_ylabel('h(x,t)')
ax.set_title('Thin Film Evolution')

def animate(i):
    line.set_ydata(frames[i])
    ax.set_title(f't = {i * 0.01:.2f} s')
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
ani.save("thin_film_evolution.mp4", writer='ffmpeg', dpi=200)

print("✅ Animation saved as thin_film_evolution.mp4")
