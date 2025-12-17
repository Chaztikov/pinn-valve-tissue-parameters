# Code Comparison: Original vs FOSLS Formulation

This document provides a detailed side-by-side comparison of the implementation differences between the original second-order formulation and the FOSLS first-order formulation.

---

## File Names

- **Original**: `src/example4_HLHS_TV_NeoHookean.py`
- **FOSLS**: `src/example4_HLHS_TV_NeoHookean_FOSLS.py`

---

## 1. Stress Computation Function

### Original: `stress(x, y)`

```python
def stress(x, y):
    """Computes Cauchy stress from displacement field"""
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Compute displacement gradients (9 components)
    duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
    # ... (8 more)
    
    # Build deformation gradient F
    Fxx = duxdx + 1.0
    # ... (8 more components)
    
    # Compute det(F) and F^(-T)
    detF = Fxx*(Fyy*Fzz - Fyz*Fzy) - ...
    invFxx = adjFxx / detF
    # ... (8 more inverse components)
    
    # Material parameters
    E = (torch.tanh(E_) + 1.0) * 400
    nu = (torch.tanh(nu_) + 1.0) / 4
    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    # First Piola-Kirchhoff stress
    lnF = torch.log(detF)
    Pxx = mu * Fxx + (lmbd * lnF - mu) * invFxx
    Pxy = mu * Fxy + (lmbd * lnF - mu) * invFyx
    # ... (7 more P components - all 9)
    
    # Transform to Cauchy stress: σ = (1/J) P F^T
    sxx = invFxx * Pxx + invFxy * Pyx + invFxz * Pzx
    sxy = invFxx * Pxy + invFxy * Pyy + invFxz * Pzy
    sxz = invFxx * Pxz + invFxy * Pyz + invFxz * Pzz
    syy = invFyx * Pxy + invFyy * Pyy + invFyz * Pzy
    syz = invFyx * Pxz + invFyy * Pyz + invFyz * Pzz
    szz = invFzx * Pxz + invFzy * Pyz + invFzz * Pzz
    
    return sxx, sxy, sxz, syy, syz, szz  # 6 Cauchy stress components
```

### FOSLS: `piola_kirchhoff_stress(x, y)`

```python
def piola_kirchhoff_stress(x, y):
    """
    Compute the First Piola-Kirchhoff stress tensor from displacement field.
    Returns the six independent components: P_xx, P_yy, P_zz, P_xy, P_xz, P_yz
    """
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Compute displacement gradients (9 components)
    duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
    # ... (8 more)
    
    # Deformation gradient F = I + grad(u)
    Fxx = duxdx + 1.0
    # ... (8 more components)
    
    # Determinant of F
    detF = Fxx*(Fyy*Fzz - Fyz*Fzy) - ...
    detF = torch.where(torch.le(detF, 0), 0.001, detF)
    
    # Compute F^(-T) via adjugate
    invFxx = adjFxx / detF
    # ... (8 more inverse components)
    
    # Material parameters
    E = (torch.tanh(E_) + 1.0) * 400
    nu = (torch.tanh(nu_) + 1.0) / 4
    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    # Compressible Neo-Hookean 1st Piola-Kirchhoff Stress: 
    # P = mu*F + [lambda * log(detF) - mu] * F^(-T)
    lnF = torch.log(detF)
    
    Pxx = mu * Fxx + (lmbd * lnF - mu) * invFxx
    Pyy = mu * Fyy + (lmbd * lnF - mu) * invFyy
    Pzz = mu * Fzz + (lmbd * lnF - mu) * invFzz
    Pxy = mu * Fxy + (lmbd * lnF - mu) * invFyx
    Pxz = mu * Fxz + (lmbd * lnF - mu) * invFzx
    Pyz = mu * Fyz + (lmbd * lnF - mu) * invFzy
    
    # NO transformation to Cauchy stress
    
    return Pxx, Pyy, Pzz, Pxy, Pxz, Pyz  # 6 P-K stress components
```

**Key Differences**:
1. ❌ Original computes all 9 components of P, then transforms to 6 Cauchy stress
2. ✅ FOSLS computes only 6 independent P components, no transformation
3. ✅ FOSLS returns P directly (what network learns)
4. ❌ Original returns σ (requires transformation)

---

## 2. PDE Residual Function

### Original: `pde(x, y)`

```python
def pde(x, y):
    # Extract network outputs
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4], y[:, 4:5], y[:, 5:6],
        y[:, 6:7], y[:, 7:8], y[:, 8:9]
    )
    
    # Compute Cauchy stress from constitutive law
    sxx, sxy, sxz, syy, syz, szz = stress(x, y)
    
    # Divergence of NETWORK Cauchy stress
    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxz_z = dde.grad.jacobian(Nsxz, x, i=0, j=2)
    # ... (6 more stress gradients)
    
    # Inertial terms
    rho = 1e-6
    d2x_dt2 = dde.grad.hessian(Nux, x, i=3, j=3)
    d2y_dt2 = dde.grad.hessian(Nuy, x, i=3, j=3)
    d2z_dt2 = dde.grad.hessian(Nuz, x, i=3, j=3)
    
    # Momentum balance: div(σ) = ρ * a
    mx = sxx_x + sxy_y + sxz_z - rho * d2x_dt2
    my = sxy_x + syy_y + syz_z - rho * d2y_dt2
    mz = sxz_x + syz_y + szz_z - rho * d2z_dt2
    
    # Constitutive matching: σ_network - σ_computed = 0
    stress_xx = sxx - Nsxx
    stress_yy = syy - Nsyy
    stress_zz = szz - Nszz
    stress_xy = sxy - Nsxy
    stress_xz = sxz - Nsxz
    stress_yz = syz - Nsyz
    
    return [mx, my, mz, stress_xx, stress_yy, stress_zz,
            stress_xy, stress_xz, stress_yz]
```

### FOSLS: `pde(x, y)`

```python
def pde(x, y):
    """
    FOSLS formulation of the hyperelastic momentum equation.
    The PDE enforces:
    1. Momentum balance: div(P) = rho * d^2u/dt^2
    2. Constitutive equation: P_network = P_computed (from constitutive law)
    
    Network outputs:
    - y[:, 0:3]: displacements (u_x, u_y, u_z)
    - y[:, 3:9]: First Piola-Kirchhoff stress components (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz)
    """
    # Extract network outputs
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    NPxx, NPyy, NPzz, NPxy, NPxz, NPyz = (
        y[:, 3:4], y[:, 4:5], y[:, 5:6],
        y[:, 6:7], y[:, 7:8], y[:, 8:9]
    )
    
    # Compute Piola-Kirchhoff stress from constitutive law
    Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = piola_kirchhoff_stress(x, y)
    
    # Divergence of NETWORK Piola-Kirchhoff stress
    # div(P) for momentum balance: dP_ij/dX_j
    dPxx_dx = dde.grad.jacobian(NPxx, x, i=0, j=0)
    dPxy_dy = dde.grad.jacobian(NPxy, x, i=0, j=1)
    dPxz_dz = dde.grad.jacobian(NPxz, x, i=0, j=2)
    # ... (6 more stress gradients)
    
    # Inertial terms
    rho = 1e-6
    d2x_dt2 = dde.grad.hessian(Nux, x, i=3, j=3)
    d2y_dt2 = dde.grad.hessian(Nuy, x, i=3, j=3)
    d2z_dt2 = dde.grad.hessian(Nuz, x, i=3, j=3)
    
    # Momentum balance residuals: div(P) - rho * d^2u/dt^2 = 0
    mx = dPxx_dx + dPxy_dy + dPxz_dz - rho * d2x_dt2
    my = dPxy_dx + dPyy_dy + dPyz_dz - rho * d2y_dt2
    mz = dPxz_dx + dPyz_dy + dPzz_dz - rho * d2z_dt2
    
    # Constitutive equation residuals: P_network - P_computed = 0
    stress_xx = Pxx - NPxx
    stress_yy = Pyy - NPyy
    stress_zz = Pzz - NPzz
    stress_xy = Pxy - NPxy
    stress_xz = Pxz - NPxz
    stress_yz = Pyz - NPyz
    
    return [mx, my, mz, stress_xx, stress_yy, stress_zz,
            stress_xy, stress_xz, stress_yz]
```

**Key Differences**:
1. ✅ FOSLS uses div(P) in momentum balance (reference configuration)
2. ❌ Original uses div(σ) in momentum balance (current configuration)
3. ✅ FOSLS: Network learns P, uses P gradients
4. ❌ Original: Network learns σ, uses σ gradients
5. Both enforce constitutive relation, but with different stress measures

---

## 3. Output Transform Function

### Original: `output_transform(x, y)`

```python
def output_transform(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Denormalize displacements
    Nux = Nux * ux_std + ux_mean
    Nuy = Nuy * uy_std + uy_mean
    Nuz = Nuz * uz_std + uz_mean
    
    # Extract network Cauchy stress
    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4], y[:, 4:5], y[:, 5:6],
        y[:, 6:7], y[:, 7:8], y[:, 8:9]
    )
    
    # Compute deformed positions
    Nux_new = Nux + x[:, 0:1]
    Nuy_new = Nuy + x[:, 1:2]
    Nuz_new = Nuz + x[:, 2:3]
    
    return torch.concat(
        [Nux, Nuy, Nuz, Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz, 
         Nux_new, Nuy_new, Nuz_new],
        axis=1
    )
```

### FOSLS: `output_transform(x, y)`

```python
def output_transform(x, y):
    """
    Transform network outputs.
    Network outputs 9 components:
    - 3 displacements (u_x, u_y, u_z)
    - 6 Piola-Kirchhoff stress components (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz)
    
    Returns 12 components:
    - Original 9 outputs
    - 3 transformed positions (x + u)
    """
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Denormalize displacements
    Nux = Nux * ux_std + ux_mean
    Nuy = Nuy * uy_std + uy_mean
    Nuz = Nuz * uz_std + uz_mean
    
    # Extract network Piola-Kirchhoff stress
    NPxx, NPyy, NPzz, NPxy, NPxz, NPyz = (
        y[:, 3:4], y[:, 4:5], y[:, 5:6],
        y[:, 6:7], y[:, 7:8], y[:, 8:9]
    )
    
    # Compute deformed positions
    Nux_new = Nux + x[:, 0:1]
    Nuy_new = Nuy + x[:, 1:2]
    Nuz_new = Nuz + x[:, 2:3]
    
    return torch.cat(
        [Nux, Nuy, Nuz, NPxx, NPyy, NPzz, NPxy, NPxz, NPyz,
         Nux_new, Nuy_new, Nuz_new],
        dim=1
    )
```

**Key Differences**:
1. Variable naming: `Nsxx, Nsxy, ...` → `NPxx, NPyy, ...`
2. Stress ordering: Original uses (xx, xy, xz, yy, yz, zz), FOSLS uses (xx, yy, zz, xy, xz, yz)
3. Function call: `torch.concat(..., axis=1)` → `torch.cat(..., dim=1)` (better compatibility)

---

## 4. Component Indexing Summary

### Network Output Indices

| Index | Original | FOSLS |
|-------|----------|-------|
| 0 | u_x | u_x |
| 1 | u_y | u_y |
| 2 | u_z | u_z |
| 3 | σ_xx | P_xx |
| 4 | σ_xy | P_yy |
| 5 | σ_xz | P_zz |
| 6 | σ_yy | P_xy |
| 7 | σ_yz | P_xz |
| 8 | σ_zz | P_yz |
| 9 | X + u_x | X + u_x |
| 10 | Y + u_y | Y + u_y |
| 11 | Z + u_z | Z + u_z |

**Note**: Ordering difference in stress components (indices 3-8)

---

## 5. Complete Side-by-Side PDE Flow

### Original Flow

```
1. Network outputs: [u, σ_network]
                         ↓
2. Compute from u:  F → J → F^(-1) → P → σ_computed
                         ↓
3. Use σ_network:   ∂σ/∂x (divergence)
                         ↓
4. Residuals:       div(σ_network) - ρa = 0  (momentum)
                    σ_computed - σ_network = 0  (constitutive)
```

### FOSLS Flow

```
1. Network outputs: [u, P_network]
                         ↓
2. Compute from u:  F → J → F^(-1) → P_computed
                         ↓
3. Use P_network:   ∂P/∂X (divergence)
                         ↓
4. Residuals:       div(P_network) - ρ₀a = 0  (momentum)
                    P_computed - P_network = 0  (constitutive)
```

**Key Difference**: 
- Original: Extra step P → σ transformation
- FOSLS: Direct use of P (no transformation)

---

## 6. Mathematical Formulation Comparison

### Original: Second-Order in Spatial Domain

**Strong Form**:
```
div(σ(u)) = ρ ∂²u/∂t²    in Ωₜ (current config)
```

**Weak Form** (implemented):
```
Find u ∈ V and σ ∈ S such that:
  div(σ) - ρ ∂²u/∂t² = 0
  σ - σ(u) = 0
```

where σ(u) is computed via P → σ transformation.

### FOSLS: First-Order System

**Strong Form**:
```
div(P) = ρ₀ ∂²u/∂t²       in Ω₀ (reference config)
P = P(F(u))                in Ω₀
```

**Weak Form** (implemented):
```
Find u ∈ V and P ∈ T such that:
  div(P) - ρ₀ ∂²u/∂t² = 0
  P - P(F(u)) = 0
```

where P(F) is directly from Neo-Hookean model.

---

## 7. Advantages Summary

| Aspect | Original | FOSLS |
|--------|----------|-------|
| **Conceptual Simplicity** | ✅ Uses familiar Cauchy stress | ❌ Requires P-K stress understanding |
| **Computational Cost** | ❌ Extra P→σ transformation | ✅ Direct P computation |
| **Reference Frame** | Current (Ωₜ) | Reference (Ω₀) |
| **Stress Symmetry** | Enforced (σ symmetric) | Not required (P general) |
| **Large Deformations** | ⚠️ May need careful handling | ✅ Natural in reference frame |
| **First-Order System** | ❌ Still second-order in u | ✅ True first-order system |
| **Direct Stress Learning** | ❌ Learns transformed stress | ✅ Learns constitutive stress |

---

## Conclusion

The FOSLS formulation is a more direct implementation that:
- Avoids the P → σ transformation
- Works directly in reference configuration
- Treats stress as a true primary variable (first-order system)
- May offer better conditioning for large deformation problems

Both formulations solve the same physical problem but with different mathematical strategies.
