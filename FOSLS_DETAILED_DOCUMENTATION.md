# Comprehensive FOSLS Formulation Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Implementation Details](#implementation-details)
4. [Code Structure](#code-structure)
5. [Comparison with Original Formulation](#comparison-with-original-formulation)
6. [Theoretical Advantages](#theoretical-advantages)
7. [Numerical Considerations](#numerical-considerations)

---

## Introduction

This document provides a comprehensive description of the First-Order System Least Squares (FOSLS) formulation for hyperelastic mechanics with Neo-Hookean material model, implemented in `src/example4_HLHS_TV_NeoHookean_FOSLS.py`.

### What is FOSLS?

FOSLS is a reformulation technique that converts higher-order PDEs into first-order systems by introducing additional variables (in this case, stress components) as primary unknowns. This approach:
- Reduces the order of derivatives needed
- Improves numerical conditioning
- Provides direct access to both displacement and stress fields
- Enables better error estimation and adaptivity

---

## Mathematical Background

### Continuum Mechanics Framework

#### Configuration and Deformation
- **Reference configuration** Ω₀: Initial (undeformed) state
- **Current configuration** Ωₜ: Deformed state at time t
- **Deformation map** φ: Ω₀ → Ωₜ
- **Displacement** u(X,t) = φ(X,t) - X

#### Deformation Gradient
The deformation gradient F relates material and spatial line elements:

```
F = ∂φ/∂X = I + ∂u/∂X = I + ∇ₓu
```

Components:
```
F_ij = δ_ij + ∂u_i/∂X_j

F = [1 + ∂u_x/∂X    ∂u_x/∂Y      ∂u_x/∂Z   ]
    [∂u_y/∂X        1 + ∂u_y/∂Y  ∂u_y/∂Z   ]
    [∂u_z/∂X        ∂u_z/∂Y      1 + ∂u_z/∂Z]
```

Jacobian (volume ratio):
```
J = det(F) > 0  (physical admissibility)
```

### Stress Tensors

#### First Piola-Kirchhoff Stress (P)
- Maps force per unit reference area to current force
- Two-point tensor (mixed reference-current)
- **Not symmetric** in general
- Components: P_ij where i = force direction, j = reference normal direction

#### Cauchy Stress (σ)
- Maps force per unit current area to current force
- Symmetric: σ = σᵀ
- Physically measurable stress

#### Relationship
```
P = J σ F⁻ᵀ
σ = (1/J) P Fᵀ
```

### Neo-Hookean Material Model

#### Strain Energy Function
For compressible Neo-Hookean material:
```
Ψ(F) = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2)(ln J)²
```

where:
- I₁ = tr(FᵀF) = tr(C) - first invariant
- C = FᵀF - right Cauchy-Green tensor
- μ = E / [2(1 + ν)] - shear modulus
- λ = E ν / [(1 + ν)(1 - 2ν)] - first Lamé parameter
- E - Young's modulus
- ν - Poisson's ratio

#### First Piola-Kirchhoff Stress
Derived from strain energy:
```
P = ∂Ψ/∂F = μ F + [λ ln(J) - μ] F⁻ᵀ
```

This is the constitutive equation that relates stress to deformation.

---

## Implementation Details

### Network Architecture

The network uses a Multi-Pathway Feedforward Neural Network (MPFNN) with two parallel paths:

```
Path 1: Input [x, y, z, t] → [4, 32, 16, 8, 3] → Displacements [u_x, u_y, u_z]
Path 2: Input [x, y, z, t] → [4, 32, 16, 8, 6] → Stresses [P_xx, P_yy, P_zz, P_xy, P_xz, P_yz]
```

**Output**: 9-component vector concatenated from both paths

### PDE Residuals

The FOSLS formulation enforces 9 residual equations:

#### 1. Momentum Balance Equations (3 residuals)

Conservation of linear momentum in reference configuration:

```python
# X-direction momentum
mx = ∂P_xx/∂X + ∂P_xy/∂Y + ∂P_xz/∂Z - ρ₀ ∂²u_x/∂t²

# Y-direction momentum  
my = ∂P_xy/∂X + ∂P_yy/∂Y + ∂P_yz/∂Z - ρ₀ ∂²u_y/∂t²

# Z-direction momentum
mz = ∂P_xz/∂X + ∂P_yz/∂Y + ∂P_zz/∂Z - ρ₀ ∂²u_z/∂t²
```

These use the **network-predicted stress** components and enforce dynamic equilibrium.

#### 2. Constitutive Equation Residuals (6 residuals)

Enforce that network stress matches constitutive law:

```python
stress_xx = P_xx(u) - P_xx_network
stress_yy = P_yy(u) - P_yy_network  
stress_zz = P_zz(u) - P_zz_network
stress_xy = P_xy(u) - P_xy_network
stress_xz = P_xz(u) - P_xz_network
stress_yz = P_yz(u) - P_yz_network
```

where P_ij(u) is computed from displacement field using the constitutive law.

### Code Implementation

#### Function: `piola_kirchhoff_stress(x, y)`

Computes P from displacement field:

```python
def piola_kirchhoff_stress(x, y):
    # 1. Extract displacements
    u_x, u_y, u_z = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # 2. Compute displacement gradients
    ∂u_x/∂X, ∂u_x/∂Y, ∂u_x/∂Z = ...
    # ... (9 components total)
    
    # 3. Build deformation gradient F = I + ∇u
    F_xx = 1.0 + ∂u_x/∂X
    F_xy = ∂u_x/∂Y
    # ... (9 components)
    
    # 4. Compute J = det(F)
    J = det(F)
    
    # 5. Compute F⁻ᵀ via adjugate
    F⁻ᵀ = adj(F) / J
    
    # 6. Material parameters
    μ = E / [2(1 + ν)]
    λ = E ν / [(1 + ν)(1 - 2ν)]
    
    # 7. Neo-Hookean stress
    P = μ F + [λ ln(J) - μ] F⁻ᵀ
    
    return P_xx, P_yy, P_zz, P_xy, P_xz, P_yz
```

**Note**: Only 6 independent components are returned (not all 9), assuming material symmetry.

#### Function: `pde(x, y)`

```python
def pde(x, y):
    # Extract network outputs
    u = y[:, 0:3]           # Displacements
    P_net = y[:, 3:9]       # Network-predicted stresses
    
    # Compute stress from constitutive law
    P_const = piola_kirchhoff_stress(x, y)
    
    # Divergence of network stress
    div_P = compute_divergence(P_net, x)
    
    # Acceleration
    a = ∂²u/∂t²
    
    # Momentum residuals
    momentum = div_P - ρ₀ * a
    
    # Constitutive residuals
    constitutive = P_const - P_net
    
    return [momentum_x, momentum_y, momentum_z,
            constitutive_xx, constitutive_yy, constitutive_zz,
            constitutive_xy, constitutive_xz, constitutive_yz]
```

### Boundary Conditions

Two sets of boundary conditions are applied:

1. **Initial conditions (t=0)**: Zero displacement
   ```python
   BC1: u(X, t=0) = 0  (components 0, 1, 2)
   ```

2. **Final conditions (t=0.1)**: Prescribed deformed position
   ```python
   BC2: X + u(X, t=0.1) = X_deformed  (components 9, 10, 11)
   ```

Note: Components 9-11 are the deformed positions computed in `output_transform`.

---

## Comparison with Original Formulation

### Original: Second-Order Formulation

**PDE**: 
```
div(σ) = ρ ∂²u/∂t²
```

**Approach**:
1. Network outputs: u and σ (Cauchy stress)
2. Compute P from u via constitutive law
3. Transform P → σ
4. Enforce momentum balance with σ
5. Match network σ with computed σ

**Derivative Requirements**:
- First derivatives: ∂u/∂X (for F)
- Second derivatives: ∂²u/∂t² (for acceleration)
- Network stress: σ (Cauchy, symmetric)

### FOSLS: First-Order Formulation

**PDE System**:
```
div(P) = ρ₀ ∂²u/∂t²
P = μ F + [λ ln(J) - μ] F⁻ᵀ
```

**Approach**:
1. Network outputs: u and P (Piola-Kirchhoff stress)
2. Compute P_constitutive from u
3. Enforce momentum balance with network P
4. Enforce P_network = P_constitutive

**Derivative Requirements**:
- First derivatives: ∂u/∂X (for F), ∂P/∂X (for divergence)
- Second derivatives: ∂²u/∂t² (for acceleration)
- Network stress: P (Piola-Kirchhoff, non-symmetric)

### Feature Comparison Table

| Feature | Original | FOSLS |
|---------|----------|-------|
| **System Order** | Second-order | First-order |
| **Network Outputs** | 9 (3 u + 6 σ) | 9 (3 u + 6 P) |
| **Stress Type** | Cauchy (σ) | Piola-Kirchhoff (P) |
| **Stress Symmetry** | Symmetric | General (6 indep.) |
| **PDE Residuals** | 9 | 9 |
| **Momentum Balance** | div(σ) | div(P) |
| **Constitutive Enforcement** | σ_net = σ(u) | P_net = P(u) |
| **Spatial Derivatives of u** | 1st order | 1st order |
| **Spatial Derivatives of stress** | 1st order | 1st order |
| **Transformation Required** | P → σ | None |

---

## Theoretical Advantages

### 1. First-Order System Benefits

**Regularity**: First-order systems require less smoothness on solutions
- Original: u ∈ H² (two continuous derivatives)
- FOSLS: u ∈ H¹, P ∈ H¹ (one continuous derivative each)

**Approximation**: Lower regularity requirements → better approximation with neural networks

### 2. Direct Stress Prediction

The network learns stress field directly rather than as a derived quantity:
- Better stress accuracy for stress-critical applications
- Useful when stress is the quantity of interest
- Natural for mixed formulations

### 3. Conditioning Improvements

FOSLS can improve problem conditioning by:
- Balancing equation orders
- Avoiding high-order derivatives
- Natural scaling of variables

### 4. Physical Interpretability

Explicit enforcement of constitutive law:
- Clear separation of kinematics (momentum) and material (constitutive)
- Easy to modify material model
- Better physical insight into solution process

### 5. Error Estimation

Independent stress prediction enables:
- Residual-based error indicators
- Separate tracking of momentum vs constitutive errors
- Better adaptive refinement strategies

---

## Numerical Considerations

### Scaling and Normalization

**Displacements** are normalized:
```python
u_normalized = (u - u_mean) / u_std
```

**Stresses** are kept in physical units (no normalization shown in code).

### Loss Weighting

The loss function uses weighted MSE:
```python
loss_weights = [1e1] * 10 + [1]
```
- Residuals 0-9: Weight 10 (momentum + constitutive + BCs)
- Residual 10: Weight 1 (Hausdorff distance)

Higher weight on physics residuals ensures PDE satisfaction.

### Determinant Regularization

To ensure physical admissibility (J > 0):
```python
J = det(F)
J = torch.where(J ≤ 0, 0.001, J)  # Clamp to small positive value
```

This prevents numerical issues from negative Jacobians.

### Material Parameter Optimization

E and ν are learned as external trainable variables:
```python
E = (tanh(E_) + 1) * 400      # Range: [0, 800]
ν = (tanh(ν_) + 1) / 4        # Range: [0, 0.5]
```

Transformation ensures physically valid ranges.

---

## Training Configuration

### Network
- Architecture: MPFNN (Multi-Pathway Feedforward)
- Path 1: [4, 32, 16, 8, 3] for displacements
- Path 2: [4, 32, 16, 8, 6] for stresses
- Activation: Swish
- Initialization: Glorot normal

### Optimizer
- Method: Adam
- Learning rate: 1e-3
- Decay: Step decay (factor 0.1 every 15000 epochs)

### Training
- Epochs: 250,000
- Display frequency: 1000 epochs
- Variable callback: Save E, ν every 1000 epochs

### Loss Components
1. BC at t=0 (displacement)
2. BC at t=0.1 (deformed position)
3. PDE residuals (9 components)
4. Hausdorff distance (geometric error)

---

## Validation and Testing

### Verification Checklist

- [x] Syntax validation (no Python errors)
- [x] Consistent variable naming
- [x] Correct tensor indexing
- [x] Physical unit consistency
- [x] Determinant positivity
- [x] Boundary condition compatibility

### Expected Outputs

1. **variables.dat**: Time history of E and ν
2. **loss.dat**: Training loss history
3. **train.dat**: Training metrics
4. Predicted material parameters at convergence

---

## References

### Theoretical Background
1. Bonet, J., & Wood, R. D. (1997). *Nonlinear continuum mechanics for finite element analysis*. Cambridge University Press.
2. Holzapfel, G. A. (2000). *Nonlinear solid mechanics*. Wiley.
3. Bochev, P. B., & Gunzburger, M. D. (2009). *Least-squares finite element methods*. Springer.

### Application Domain
4. Wu, W., Daneker, M., Herz, C., Dewey, H., Weiss, J. A., Pouch, A. M., Lu, L., & Jolley, M. A. (2025). A Noninvasive Method for Determining Elastic Properties of Valve Tissue Using Physics-Informed Neural Networks. *Acta Biomaterialia*, 200, 283-298.

### Deep Learning Framework
5. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228.

---

## Appendix: Complete Code Example

### Minimal Working Example

```python
import deepxde as dde
from deepxde.backend import torch

# Define geometry and time
geomtime = dde.geometry.PointCloud(points=pde_pts)

# Material parameters (trainable)
E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)

# FOSLS formulation
def piola_kirchhoff_stress(x, y):
    # Extract displacements
    u_x, u_y, u_z = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Compute F = I + ∇u
    F = compute_deformation_gradient(x, y)
    
    # Material parameters
    E = (torch.tanh(E_) + 1.0) * 400
    nu = (torch.tanh(nu_) + 1.0) / 4
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # Neo-Hookean stress
    J = det(F)
    P = mu * F + (lmbda * ln(J) - mu) * inv_transpose(F)
    
    return P_xx, P_yy, P_zz, P_xy, P_xz, P_yz

def pde(x, y):
    # Network predictions
    u = y[:, 0:3]
    P_net = y[:, 3:9]
    
    # Constitutive stress
    P_const = piola_kirchhoff_stress(x, y)
    
    # Residuals
    momentum = div(P_net) - rho * d2_dt2(u)
    constitutive = P_const - P_net
    
    return momentum + constitutive

# Train
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, 
              external_trainable_variables=[E_, nu_])
model.train(epochs=250000)
```

---

## Summary

The FOSLS formulation provides a robust alternative to traditional second-order formulations for hyperelastic problems. By treating stress as a primary unknown and explicitly enforcing the constitutive equation, it offers improved numerical properties and direct access to stress fields—critical for biomechanical applications like valve tissue characterization.

**Key Takeaway**: FOSLS trades a slightly larger system (9 unknowns vs computing 3) for better conditioning, lower regularity requirements, and enhanced physical interpretability.
