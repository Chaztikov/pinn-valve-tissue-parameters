# FOSLS Formulation for Neo-Hookean Material

This document describes the First Order System Least Squares (FOSLS) formulation implemented in `example4_HLHS_TV_NeoHookean_FOSLS.py`.

## Overview

The FOSLS formulation reformulates the second-order hyperelastic momentum equation into a first-order system by treating both displacement and stress as primary unknowns.

## Key Differences from Original Formulation

### Original Formulation (`example4_HLHS_TV_NeoHookean.py`)
- Network outputs: 3 displacements + 6 Cauchy stress components
- PDE: Second-order momentum balance using Cauchy stress divergence
- Stress function: Computes Cauchy stress from Piola-Kirchhoff stress

### FOSLS Formulation (`example4_HLHS_TV_NeoHookean_FOSLS.py`)
- Network outputs: 3 displacements + 6 First Piola-Kirchhoff stress components
- PDE: First-order system with two sets of residuals
  1. Momentum balance: div(P) = ρ * d²u/dt²
  2. Constitutive equation: P_network - P_computed = 0
- Stress function: Directly computes First Piola-Kirchhoff stress

## Network Outputs

The network outputs 9 components (indices 0-8):
- `y[:, 0:3]`: Displacement components (u_x, u_y, u_z)
- `y[:, 3:9]`: First Piola-Kirchhoff stress components (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz)

Note: Only 6 stress components are needed due to symmetry in material response (though P itself is not symmetric in general).

## PDE Formulation

The FOSLS formulation enforces two sets of equations:

### 1. Momentum Balance
```
∂P_xx/∂X + ∂P_xy/∂Y + ∂P_xz/∂Z = ρ * ∂²u_x/∂t²
∂P_xy/∂X + ∂P_yy/∂Y + ∂P_yz/∂Z = ρ * ∂²u_y/∂t²
∂P_xz/∂X + ∂P_yz/∂Y + ∂P_zz/∂Z = ρ * ∂²u_z/∂t²
```

### 2. Constitutive Equation (Neo-Hookean Material)
```
P = μ*F + [λ * ln(det(F)) - μ] * F^(-T)
```

Where:
- P: First Piola-Kirchhoff stress tensor
- F: Deformation gradient = I + ∇u
- μ, λ: Lamé parameters derived from Young's modulus E and Poisson's ratio ν
- F^(-T): Inverse transpose of deformation gradient

The residuals enforce:
```
P_xx_network - P_xx_computed = 0
P_yy_network - P_yy_computed = 0
P_zz_network - P_zz_computed = 0
P_xy_network - P_xy_computed = 0
P_xz_network - P_xz_computed = 0
P_yz_network - P_yz_computed = 0
```

## Functions

### `piola_kirchhoff_stress(x, y)`
Computes the First Piola-Kirchhoff stress tensor from the displacement field using the Neo-Hookean constitutive law.

**Returns**: P_xx, P_yy, P_zz, P_xy, P_xz, P_yz

### `pde(x, y)`
Implements the FOSLS formulation residuals.

**Returns**: 9 residuals
- Residuals 0-2: Momentum balance in x, y, z directions
- Residuals 3-8: Constitutive equation for 6 stress components

### `output_transform(x, y)`
Transforms network outputs by:
1. Denormalizing displacements using mean and std
2. Keeping Piola-Kirchhoff stress components as-is
3. Computing deformed positions (x + u)

**Returns**: 12 components
- Components 0-8: Original network outputs (3 displacements + 6 stresses)
- Components 9-11: Deformed positions (used for boundary conditions)

## Advantages of FOSLS Formulation

1. **First-order system**: Avoids second-order spatial derivatives of displacement
2. **Direct stress prediction**: Network learns stress field directly
3. **Better conditioning**: FOSLS can lead to better-conditioned systems
4. **Physical insight**: Explicitly enforces constitutive relation as a residual

## Usage

Run the code the same way as the original:
```bash
cd src
python example4_HLHS_TV_NeoHookean_FOSLS.py
```

The code will:
1. Load HLHS tricuspid valve data
2. Train a PINN to learn both displacement and stress fields
3. Infer material parameters E (Young's modulus) and ν (Poisson's ratio)
4. Save training history and predicted parameters

## Dependencies

- DeepXDE v1.12.1
- PyTorch
- NumPy

## References

- Javier Bonet, Richard D. Wood. *Nonlinear continuum mechanics for finite element analysis*. Cambridge University Press, 1997.
- W. Wu, M. Daneker, et al. *A Noninvasive Method for Determining Elastic Properties of Valve Tissue Using Physics-Informed Neural Networks*, Acta Biomaterialia, 200, 283-298, 2025.
