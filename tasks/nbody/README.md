# Geometric Algebra for Physic-Informed Neural Networks

A demonstration of **Geometric Algebra** for learning physical laws from data.
This project compares a **Standard Transformer** against a **Versor Rotor RNN** on an N-body gravity simulation.

## The Problem
Predicting the motion of N stars interacting via gravity ($F = G \frac{m_1 m_2}{r^2}$) requires respecting conservation laws (Energy, Momentum) and Symmetries (Rotation, Translation). Standard Neural Networks often violate these, causing planets to spiral into the sun or drift away.

## The Solution: Versor
By embedding the state into a **Geometric Algebra (Cl(4,1))** multivector and using **Rotor-based updates**, the model learns to respect geometric constraints (like rotations being structure-preserving) naturally.

## Contents
- `data_gen.py`: Generates synthetic N-body gravity data (positions, velocities).
- `models.py`:
    - **StandardTransformer**: A baseline `nn.TransformerEncoder`.
    - **VersorRotorRNN**: A Recurrent Network using Geometric Linear Layers (`algebra.py`).
- `train.py`: Trains both models and compares them on **Energy Drift** and **MSE**.
- `algebra.py`: The Geometric Algebra kernel (from Versor).

## Comparative Benchmarking Results
Analysis of the **Standard Transformer**, **Graph Network Simulator (GNS)**, **Hamiltonian Neural Network (HNN)**, and **Versor** architectures over a 100-step autoregressive rollout.

| Model | MSE (Motion) | Energy Drift (Physics) | Notes |
|-------|--------------|------------------------|-------|
| Standard Transformer | 18.63 | 214.31 | Unstable |
| GNS (Relational) | 23.83 | 1261.01 | Suffers from coordinate drift |
| HNN (Energy-based) | 11.12 | **61.86** | Great physics, average motion |
| **Versor (Rotor RNN)** | **5.45** | **66.13** | **Best performance on balance** |

### Comprehensive Architectural Analysis

#### 1. Standard Transformer (Baseline Architecture)
*   **Performance**: High MSE, high Energy Drift.
*   **Why**: Transformers are universal function approximators but have no concept of 3D space. They treat $(x, y, z)$ as independent numbers rather than a vector. In autoregressive rollouts, small errors in coordinate prediction compound into "non-physical" states (e.g., a planet suddenly gaining mass/energy), causing the system to explode.

#### 2. GNS - Graph Network Simulator (Relational Inductive Bias)
*   **Performance**: Worst MSE/Drift in this specific task.
*   **Why**: GNS is designed for local interactions (like sand or fluid). For a global system like 5-body gravity, the graph is fully connected. GNS update steps are purely additive; without a global conservation prior, the "message-passing" errors aggregate across the particles, leading to massive energy drift (1200+) where the system loses all structure.

#### 3. HNN - Hamiltonian Neural Network (Energy-Conserving Architecture)
*   **Performance**: Best Energy Drift, Moderate MSE.
*   **Why**: HNNs are mathematically forced to conserve energy because they predict a scalar potential $H$ and derive motion from it. This is why its drift is so low (61.8). However, the mapping from raw coordinates $(q, p)$ to a stable Hamiltonian surface is difficult to learn precisely from small data, leading to "accurate-ish" motion but slightly higher coordinate error than Versor.

#### 4. Versor - Rotor RNN (Proposed Architecture)
*   **Performance**: Best MSE, near-perfect Energy Stability.
*   **Why**:
    *   **Rotational Invariance**: Because it operates in $Cl(4,1)$ Geometric Algebra, a rotation of the solar system is just a change in the rotor state, not a new pattern to learn.
    *   **Rotor Accumulator**: Its recurrent state uses Geometric Linear layers that effectively learn the *Lie Algebra* of the motion (the "screw motions" of physics).
    *   **Manifold Normalization**: Similar to how HNNs force conservation via a scalar field, Versor forces stability by projecting hidden states back onto the Geometric Manifold. This provides "Physical Regularization" without the rigid constraints of a Hamiltonian.

## Requirements
- PyTorch
- NumPy

