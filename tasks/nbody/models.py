import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Append parent directory to system path for Model package visibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import algebra
except ImportError:
    # Contingency for alternative execution environments
    try:
        from . import algebra
    except ImportError:
        import kernel as algebra

from Model.model import VersorBlock, normalize_cl41, VersorLinear, VersorActivation

# Try to import C++ extension
try:
    if os.environ.get("VERSOR_NO_CPP"):
        raise ImportError("Forced Python Fallback")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp'))
    import versor_cpp
    HAS_CPP_CORE = True
    print("Versor C++ Core Accelerated: ON")
except ImportError as e:
    HAS_CPP_CORE = False
    print(f"Versor C++ Core Accelerated: OFF (Using Python Fallback). Error: {e}")

class StandardTransformer(nn.Module):
    def __init__(self, input_dim=6, n_particles=5, d_model=128, n_head=4, n_layers=2):
        super().__init__()
        self.input_dim = input_dim * n_particles
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.1) # Stochastic learnable positional encoding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, self.input_dim)
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B, S, -1) # (B, S, N*6)
        
        emb = self.embedding(x_flat) + self.pos_encoder[:, :S, :]
        
        # Generation of a causal temporal mask
        # M(i, j) = -\infty \text{ if } j > i \text{ else } 0
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        
        out = self.transformer(emb, mask=mask)
        pred = self.head(out)
        
        return pred.reshape(B, S, N, D)

class VersorRotorRNN(nn.Module):
    """
    Recurrent Neural Network using Geometric Algebra.
    Optimized for Physics stability.
    NOTE: Sequential implementation for recurrent mode benchmarking.
    """
    def __init__(self, input_dim=6, d_mv=32, hidden_channels=16, n_particles=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.d_mv = d_mv
        self.n_particles = n_particles
        
        # Particle-agnostic projections
        self.proj_in = nn.Linear(input_dim, hidden_channels * 32)
        self.proj_out = nn.Linear(hidden_channels * 32, input_dim)
        
    def forward(self, x):
        # x: (B, S, N, D)
        B, S, N, D = x.shape
        
        # Initial spinor status per particle
        # Shape: (Batch, Particles, Hidden_Channels, 32)
        psi = torch.zeros(B, N, self.hidden_channels, 32, device=x.device)
        psi[..., 0] = 1.0 
        
        outputs = []
        # Project each particle's state independently
        # (B, S, N, D) -> (B, S, N, Hidden, 32)
        x_embs = self.proj_in(x).reshape(B, S, N, self.hidden_channels, 32)
        
        if HAS_CPP_CORE:
            # C++ Core Implementation (Recursive Rotor Accumulator)
            # Parallelized O(L) scan on CPU/extension
            # x_embs: (B, S, N, Hidden, 32)
            
            # Ensure CPU contiguity
            x_cpu = x_embs.detach().cpu()
            
            # Run C++ Core
            # Returns (B, S, N, H, 32)
            psi_seq = versor_cpp.rra_scan_forward(x_cpu)
            
            # Move back to original device
            psi_seq = psi_seq.to(x.device)
            
            # Parallel Output Projection
            # Flatten B and S: (B*S, N, H, 32)
            B, S, N, H, _ = x_embs.shape
            psi_flat = psi_seq.reshape(B*S, N, -1) # (B*S, N, H*32)
            
            # Linear Projection: (B*S, N, D)
            pred_delta = self.proj_out(psi_flat)
            
            # Residual Connection
            outputs = x.reshape(B*S, N, D) + pred_delta
            
            return outputs.reshape(B, S, N, D)

        # Fallback Python Implementation
        for t in range(S):
            # 1. Delta-Rotor generator for each particle
            u_t = x_embs[:, t] # (B, N, Hidden, 32)
            
            # 2. Cayley map to create localized rotors
            delta_r = u_t.clone()
            delta_r[..., 0] += 1.0 
            delta_r = algebra.manifold_normalization(delta_r)
            
            # 3. Multiplicative Update (Per-particle spinor rotation)
            # psi: (B, N, H, 32), delta_r: (B, N, H, 32)
            psi = algebra.geometric_product(delta_r, psi)
            psi = algebra.manifold_normalization(psi)
            
            # 4. Output Projection
            out_emb = psi # (B, N, H, 32)
            pred_delta = self.proj_out(out_emb.reshape(B, N, -1)) # (B, N, D)
            outputs.append(x[:, t] + pred_delta)
            
        return torch.stack(outputs, dim=1) # (B, S, N, D)

class GraphNetworkSimulator(nn.Module):
    """
    Relational inductive bias implementation.
    Standardized with LayerNorm for fair benchmarking.
    """
    def __init__(self, n_particles=5, input_dim=6, hidden_dim=64):
        super().__init__()
        self.n_particles = n_particles
        self.node_enc = nn.Linear(input_dim, hidden_dim)
        self.edge_enc = nn.Linear(input_dim, hidden_dim)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        B, S, N, D = x.shape
        x_flat = x.reshape(B*S, N, D)
        nodes = self.node_enc(x_flat) 
        
        x_i = x_flat.unsqueeze(2).expand(-1, -1, N, -1)
        x_j = x_flat.unsqueeze(1).expand(-1, N, -1, -1)
        rel_x = x_j - x_i
        edges = self.edge_enc(rel_x)
        
        n_i = nodes.unsqueeze(2).expand(-1, -1, N, -1)
        n_j = nodes.unsqueeze(1).expand(-1, N, -1, -1)
        
        edge_input = torch.cat([n_i, n_j, edges], dim=-1)
        messages = self.edge_mlp(edge_input).sum(dim=2)
        
        next_state = x_flat + self.node_mlp(torch.cat([nodes, messages], dim=-1))
        return next_state.reshape(B, S, N, D)

class HamiltonianNN(nn.Module):
    def __init__(self, n_particles=5, input_dim=6, hidden_dim=128):
        super().__init__()
        self.n_particles = n_particles
        self.state_dim = n_particles * 6
        self.h_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, dt=0.01):
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N * D)
        with torch.enable_grad():
            x_grad = x_flat.detach().requires_grad_(True)
            energy = self.h_net(x_grad)
            grads = torch.autograd.grad(energy.sum(), x_grad, create_graph=True)[0]
        
        grads = grads.reshape(B*S, N, 6)
        dot_q, dot_p = grads[..., 3:], -grads[..., :3]
        next_state = x_flat.reshape(B*S, N, 6) + torch.cat([dot_q, dot_p], dim=-1) * dt
        return next_state.reshape(B, S, N, D)

class HamiltonianVersorNN(nn.Module):
    """
    Hamiltonian Neural Network with Versor (Geometric Algebra) backbone.
    Hybrid model combining symplectic integration with geometric algebra representations.
    """
    def __init__(self, n_particles=5, input_dim=6, embed_dim=8, hidden_dim=32):
        super().__init__()
        self.n_particles = n_particles
        self.embed_dim = embed_dim
        
        # Project (pos, vel) or (q, p) to Versor space
        # We input 6 scalars per particle -> embed_dim multivectors
        self.input_proj = nn.Linear(input_dim, embed_dim * 32)
        
        # Versor Backbone to approximate the Hamiltonian H(q, p)
        self.v_net = nn.Sequential(
            VersorLinear(embed_dim, hidden_dim),
            VersorActivation(),
            VersorLinear(hidden_dim, hidden_dim),
            VersorActivation(),
            VersorLinear(hidden_dim, 1) # Output 1 multivector per particle (we will use scalar part)
        )
        
    def forward(self, x, dt=0.01):
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N, D)
        
        # Enable gradient calculation for H wrt x
        with torch.enable_grad():
            x_grad = x_flat.detach().requires_grad_(True) # (BS, N, 6)
            
            # 1. Project to Versor Space
            # (BS, N, 6) -> (BS, N, embed_dim * 32)
            h_emb = self.input_proj(x_grad)
            h_emb = h_emb.view(-1, N, self.embed_dim, 32) # match VersorLinear shape (B, S, In, 32) where S=N here
            
            # 2. Compute per-particle Energy-like terms
            # v_net expects (B, S, In, 32). We pass (BS, N, embed_dim, 32).
            # Output is (BS, N, 1, 32)
            h_out = self.v_net(h_emb)
            
            # 3. Aggregate to Global Scalar Energy
            # We take the scalar part (index 0) of the output multivector
            # h_out[..., 0] is (BS, N, 1)
            # Sum over particles to get total energy H for the system
            energy = h_out[..., 0].sum()
            
            # 4. Compute gradients (Hamilton's equations)
            grads = torch.autograd.grad(energy, x_grad, create_graph=True)[0]
            
        grads = grads.reshape(B*S, N, 6)
        # Symplectic update: dot_q = dH/dp, dot_p = -dH/dq
        # x is (q, p) = (pos, vel) if mass=1
        # grads is (dH/dq, dH/dp)
        dq_dt = grads[..., 3:]
        dp_dt = -grads[..., :3]
        
        next_state = x_flat + torch.cat([dq_dt, dp_dt], dim=-1) * dt
        return next_state.reshape(B, S, N, D)

class MambaSimulator(nn.Module):
    """
    Selective State Space Model baseline.
    Vectorized where possible for fair comparison.
    """
    def __init__(self, n_particles=5, input_dim=6, d_model=256, d_state=64):
        super().__init__()
        self.n_particles = n_particles
        self.d_model = d_model
        self.d_state = d_state
        self.input_size = n_particles * input_dim
        
        self.embedding = nn.Linear(self.input_size, d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.A = nn.Parameter(-torch.exp(torch.arange(1, d_state + 1).float() * 0.1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.input_size)
        
    def forward(self, x):
        B, S, N, D = x.shape
        x_flat = x.reshape(B, S, -1)
        u = self.embedding(x_flat)
        
        # Pre-compute all selective parameters for the entire sequence at once
        sel = self.x_proj(u) # (B, S, d_state * 2 + 1)
        deltas = F.softplus(sel[..., 0:1]) * 0.01
        Bs = sel[..., 1:1+self.d_state]
        Cs = sel[..., 1+self.d_state:]
        
        h = torch.zeros(B, self.d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(S):
            u_t = u[:, t]
            delta_t = deltas[:, t]
            B_t = Bs[:, t]
            C_t = Cs[:, t]
            
            A_bar = torch.exp(delta_t * self.A)
            B_bar = delta_t * B_t
            
            h = h * A_bar.unsqueeze(1) + B_bar.unsqueeze(1) * u_t.unsqueeze(2)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + u_t * self.D
            outputs.append(y_t)
            
        pred = self.out_proj(self.norm(torch.stack(outputs, dim=1)))
        return pred.reshape(B, S, N, D)

class VersorPhysicsTransformer(nn.Module):
    """
    Geometric Transformer adapted for N-Body physics.
    Uses VersorBlocks (with GPA) to process particle interactions.
    """
    def __init__(self, n_particles=5, input_dim=6, embed_dim=16, n_heads=4, n_layers=2):
        super().__init__()
        self.n_particles = n_particles
        self.embed_dim = embed_dim
        
        # Project each particle's (pos, vel) to a multivector state
        self.input_proj = nn.Linear(input_dim, embed_dim * 32)
        
        self.blocks = nn.ModuleList([
            VersorBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])
        
        # Project back to (pos, vel) deltas
        self.output_proj = nn.Linear(embed_dim * 32, input_dim)
        
    def forward(self, x, return_attention=False):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N, D)
        
        # Project to MV space
        h = self.input_proj(x_flat).view(B * S, N, self.embed_dim, 32)
        
        all_attn_scalar = []
        all_attn_bivector = []
        for block in self.blocks:
            if return_attention:
                h, (attn, biv) = block(h, return_attention=True)
                all_attn_scalar.append(attn)
                all_attn_bivector.append(biv)
            else:
                h = block(h)
                
        # Project back
        out_flat = h.view(B * S, N, -1)
        delta = self.output_proj(out_flat)
        
        # Residual update
        next_state = x_flat + delta
        
        if return_attention:
            return next_state.reshape(B, S, N, D), (all_attn_scalar, all_attn_bivector)
        return next_state.reshape(B, S, N, D)

class MultiChannelVersor(nn.Module):
    """
    Multi-Channel Geometrically Equivariant Transformer.
    Processes K parallel geometric channels, allowing for higher capacity (D_model = K * 32)
    while maintaining partial equivariance within channels.
    """
    def __init__(self, n_particles=5, input_dim=6, n_channels=4, n_heads=4, n_layers=2):
        super().__init__()
        self.n_particles = n_particles
        self.n_channels = n_channels
        self.d_mv = 32
        
        # Input Projection: Project (pos, vel) into K * 32 dimensions
        self.input_proj = nn.Linear(input_dim, n_channels * self.d_mv)
        
        # Parallel Geometric Blocks
        # Treat (B, S, N, K, 32) as (B*K, S, N, 32) for the block.
        self.blocks = nn.ModuleList([
            VersorBlock(n_channels, n_heads) for _ in range(n_layers)
        ])
        
        # Mixing Layer: Equivariant inter-channel communication
        self.mixing = nn.Sequential(
            VersorLinear(n_channels, n_channels),
            VersorActivation(),
            VersorLinear(n_channels, n_channels)
        )
        self.output_proj = nn.Linear(n_channels * 32, input_dim)
        
    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, _ = x.shape
        x_flat = x.reshape(B * S, N, -1)
        
        # (B*S, N, K*32) -> (B*S, N, K, 32)
        h = self.input_proj(x_flat).reshape(B * S, N, self.n_channels, 32)
        
        # Fold K into Batch for parallel processing
        # (B*S*K, N, 32, 1) if VersorBlock expects (B, N, D, 32)? 
        # Checking VersorBlock: expects (..., N, D, 32)
        # So we pass (B*S, N, K, 32) directly if dim matches.
        # VersorBlock uses geometric_linear which handles (..., K, 32)
        
        h_in = h # (Batch, N, Channels, 32)
        
        for block in self.blocks:
            h_in = block(h_in)
            
        # Mixing: Clifford-covariant channel interaction
        h_mixed = h_in + self.mixing(h_in)
        
        # Projection: Map multivectors back to state space
        delta = self.output_proj(h_mixed.reshape(B * S, N, -1))
        next_state = x_flat + delta
        
        return next_state.reshape(B, S, N, -1)

class EquivariantGNN(nn.Module):
    """
    EGNN (Equivariant Graph Neural Network) baseline.
    Standard implementation for comparison with GA-based models.
    """
    def __init__(self, n_particles=5, input_dim=6, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )
        self.emb = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: (B, S, N, 6)
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N, D)
        pos = x_flat[..., :3]
        vel = x_flat[..., 3:]
        
        h = self.emb(x_flat)
        
        # Edge update
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        dist_sq = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1))**2, dim=-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h_i, h_j, dist_sq], dim=-1))
        m_i = m_ij.sum(dim=2)
        
        # Pos update
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        pos_out = pos + (rel_pos * self.coord_mlp(m_ij)).sum(dim=2) / N
        
        # Node update
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        vel_out = vel + self.final_mlp(h)[..., 3:]
        
        out = torch.cat([pos_out, vel_out], dim=-1)
        return out.reshape(B, S, N, D)
