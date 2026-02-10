import torch
import numpy as np

def generate_gravity_data(n_samples=1000, n_steps=200, n_particles=5, dt=0.01, device='cpu'):
    """
    Generates N-body gravity interaction data.
    Returns: Tensor of shape (n_samples, n_steps, n_particles, 6)
             where 6 = (px, py, pz, vx, vy, vz)
    """
    # Position and velocity initialization
    # Coordinate range: [-1, 1], Velocity range: [-0.5, 0.5]
    pos = torch.randn(n_samples, n_particles, 3, device=device) * 0.5
    vel = torch.randn(n_samples, n_particles, 3, device=device) * 0.5
    
    # Mass distribution (randomly assigned between [0.5, 1.5])
    mass = torch.rand(n_samples, n_particles, 1, device=device) + 0.5
    
    G = 1.0  # Gravitational constant
    
    def get_acc(p, m):
        diff = p.unsqueeze(2) - p.unsqueeze(1) 
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-3
        direction = -diff
        force_magnitude = (G * m.unsqueeze(2) * m.unsqueeze(1)) / (dist ** 3)
        mask = ~torch.eye(n_particles, device=device).bool().unsqueeze(0).unsqueeze(-1)
        force = (direction * force_magnitude * mask).sum(dim=2)
        return force / m

    acc = get_acc(pos, mass)
    trajectory = []
    
    for _ in range(n_steps):
        trajectory.append(torch.cat([pos, vel], dim=-1))
        
        # Velocity Verlet (Leapfrog) Integration
        # 1. v(t + dt/2) = v(t) + a(t) * dt/2
        vel_half = vel + acc * (dt / 2.0)
        
        # 2. x(t + dt) = x(t) + v(t + dt/2) * dt
        pos = pos + vel_half * dt
        
        # Boundary enforcement
        box_bound = 2.0
        hit_wall = (pos.abs() > box_bound)
        vel_half[hit_wall] *= -1.0 
        pos[hit_wall] = torch.sign(pos[hit_wall]) * box_bound
        
        # 3. a(t + dt) = Force(x(t + dt)) / m
        acc = get_acc(pos, mass)
        
        # 4. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        vel = vel_half + acc * (dt / 2.0)
        
    return torch.stack(trajectory, dim=1)

if __name__ == "__main__":
    print("Generating sample data...")
    t0 = torch.tensor(0.0)
    data = generate_gravity_data(n_samples=10, n_steps=200, n_particles=3)
    print(f"Data shape: {data.shape}")
    print("Sample 0, Particle 0, Step 0-5:\n", data[0, :5, 0, :])
