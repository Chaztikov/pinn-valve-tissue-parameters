# Documentation Index for FOSLS Formulation

This repository contains comprehensive documentation for the FOSLS (First-Order System Least Squares) formulation of the Neo-Hookean hyperelastic material model.

## Documentation Files

### 1. [FOSLS_FORMULATION.md](FOSLS_FORMULATION.md)
**Purpose**: Quick reference and overview  
**Contents**:
- Overview of FOSLS approach
- Key differences from original formulation
- Network outputs and PDE formulation
- Function descriptions
- Usage instructions
- References

**Best for**: Quick understanding of what FOSLS is and how it differs from the original

---

### 2. [FOSLS_DETAILED_DOCUMENTATION.md](FOSLS_DETAILED_DOCUMENTATION.md)
**Purpose**: Complete mathematical and implementation reference  
**Contents**:
- Mathematical background (continuum mechanics, stress tensors)
- Detailed Neo-Hookean material model derivation
- Complete implementation details
- Code structure with examples
- Theoretical advantages
- Numerical considerations
- Training configuration
- Validation checklist
- Complete references

**Best for**: Deep understanding of the mathematics, physics, and implementation details

---

### 3. [CODE_COMPARISON.md](CODE_COMPARISON.md)
**Purpose**: Side-by-side comparison of original vs FOSLS  
**Contents**:
- Function-by-function code comparison
- Stress computation differences
- PDE residual differences
- Output transform changes
- Component indexing tables
- Complete data flow diagrams
- Advantages summary

**Best for**: Understanding exactly what changed and why

---

## Quick Navigation

### For Different Audiences

**Researchers/PhD Students**:
1. Start with `FOSLS_FORMULATION.md` for overview
2. Read `FOSLS_DETAILED_DOCUMENTATION.md` for mathematical rigor
3. Reference `CODE_COMPARISON.md` for implementation details

**Software Engineers**:
1. Start with `CODE_COMPARISON.md` to see code changes
2. Reference `FOSLS_FORMULATION.md` for context
3. Use `FOSLS_DETAILED_DOCUMENTATION.md` for edge cases

**Course Instructors**:
1. Use `FOSLS_DETAILED_DOCUMENTATION.md` as teaching material
2. Assign `CODE_COMPARISON.md` for homework analysis
3. Reference `FOSLS_FORMULATION.md` for summary slides

---

## Key Concepts by Document

### Mathematical Concepts

| Concept | FOSLS_FORMULATION.md | FOSLS_DETAILED_DOCUMENTATION.md | CODE_COMPARISON.md |
|---------|---------------------|--------------------------------|-------------------|
| Deformation Gradient | ✓ | ✓✓✓ | ✓ |
| Piola-Kirchhoff Stress | ✓ | ✓✓✓ | ✓✓ |
| Cauchy Stress | ✓ | ✓✓ | ✓✓ |
| Neo-Hookean Model | ✓ | ✓✓✓ | ✓ |
| FOSLS Theory | ✓ | ✓✓✓ | - |
| First-Order Systems | ✓ | ✓✓✓ | ✓ |

### Implementation Concepts

| Concept | FOSLS_FORMULATION.md | FOSLS_DETAILED_DOCUMENTATION.md | CODE_COMPARISON.md |
|---------|---------------------|--------------------------------|-------------------|
| Network Architecture | ✓ | ✓✓✓ | ✓ |
| PDE Residuals | ✓✓ | ✓✓✓ | ✓✓✓ |
| Boundary Conditions | ✓ | ✓✓ | ✓ |
| Loss Function | - | ✓✓✓ | - |
| Training Details | ✓ | ✓✓✓ | - |
| Code Examples | - | ✓✓✓ | ✓✓✓ |

Legend: ✓ = Basic coverage, ✓✓ = Detailed coverage, ✓✓✓ = Comprehensive coverage

---

## Documentation Statistics

- **Total pages**: ~30 (combined)
- **Total words**: ~10,000
- **Code examples**: 15+
- **Equations**: 50+
- **Comparison tables**: 8
- **References**: 6 academic sources

---

## Summary of Changes from Original

The FOSLS implementation makes **surgical, minimal changes** to the original code:

### Files Changed
- ✅ **1 new file**: `src/example4_HLHS_TV_NeoHookean_FOSLS.py`
- ✅ **3 documentation files**: This comprehensive documentation set
- ✅ **1 .gitignore**: To prevent cache files from being committed

### Code Changes
- ✅ Renamed `stress()` → `piola_kirchhoff_stress()`
- ✅ Removed P→σ transformation (15 lines)
- ✅ Updated `pde()` to use P instead of σ
- ✅ Updated variable names in `output_transform()`
- ✅ Changed `torch.concat` → `torch.cat` for compatibility

### Mathematical Changes
- ✅ Network outputs P instead of σ
- ✅ Momentum balance uses div(P) instead of div(σ)
- ✅ Constitutive residuals enforce P_network = P_computed directly

**Total lines changed**: ~60 out of 344 (17.4%)

---

## Comparison with Other Formulations

This FOSLS implementation can be compared with:

1. **Standard Galerkin** (displacement-only)
2. **Mixed formulations** (displacement + pressure)
3. **Stress-based formulations** (stress-only)
4. **Hybrid formulations** (multiple field variables)

The FOSLS approach is unique in:
- Treating both u and P as primary unknowns
- Enforcing constitutive law as a PDE residual
- Working in reference configuration
- Using first-order system formulation

---

## Future Extensions

Potential extensions documented in this set:

1. **Incompressible limit**: Modify for ν → 0.5
2. **Anisotropic materials**: Extend to fiber-reinforced models
3. **Viscoelasticity**: Add time-dependent material response
4. **Damage models**: Include progressive failure
5. **Mesh adaptivity**: Use stress residuals for refinement

See `FOSLS_DETAILED_DOCUMENTATION.md` for more details on each.

---

## Questions and Support

For questions about:
- **Mathematics**: See `FOSLS_DETAILED_DOCUMENTATION.md` References section
- **Implementation**: Compare with `CODE_COMPARISON.md`
- **Usage**: Follow instructions in `FOSLS_FORMULATION.md`
- **Issues**: Open an issue in the GitHub repository

---

## Citation

If you use this FOSLS formulation in your research, please cite:

```bibtex
@article{wu2025noninvasive,
  author  = {Wu, Wensi and Daneker, Mitchell and Herz, Christian and Dewey, Hannah and Weiss, Jeffrey A. and Pouch, Alison M. and Lu, Lu and Jolley, Matthew A.},
  title   = {A Noninvasive Method for Determining Elastic Parameters of Valve Tissue Using Physics-Informed Neural Networks}, 
  journal = {Acta Biomaterialia},
  volume  = {200},
  pages   = {283-298},
  year    = {2025},
  doi     = {https://doi.org/10.1016/j.actbio.2025.05.021}
}
```

And mention the FOSLS formulation extension in your methods section.

---

Last updated: 2025-12-17
