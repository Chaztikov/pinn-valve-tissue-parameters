"""
FOSLS (First Order System Least Squares) formulation of the HLHS tricuspid valve 
with Neo-Hookean material model.

This version uses a first-order system formulation where the network outputs both:
1. Displacement components (u_x, u_y, u_z) - primitive variables
2. Six independent components of the First Piola-Kirchhoff stress tensor 
   (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz)

The PDE residuals enforce:
1. Momentum balance: div(P) = rho * d^2u/dt^2
2. Constitutive equation: P_network = P_computed (from constitutive law)
"""

import os

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
from deepxde.backend import torch
from deepxde.nn import activations

dde.config.set_random_seed(2024)

if torch.cuda.is_available():
    torch.cuda.set_device(0)


class MPFNN(dde.nn.PFNN):
    def __init__(self, layer_sizes, second_layer_sizes, activation, kernel_initializer):
        super(MPFNN, self).__init__(layer_sizes, activation, kernel_initializer)
        self.first_layer_sizes = layer_sizes
        self.second_layer_sizes = second_layer_sizes
        self.activation = activations.get(activation)

        self.firstFNN = dde.nn.PFNN(
            self.first_layer_sizes, self.activation, kernel_initializer
        )
        self.secondFNN = dde.nn.PFNN(
            self.second_layer_sizes, self.activation, kernel_initializer
        )

    def forward(self, inputs):
        x = inputs

        if self._input_transform is not None:
            x = self._input_transform(x)

        x_firstFNN = self.firstFNN(x)
        x_secondFNN = self.secondFNN(x)

        x = torch.cat((x_firstFNN, x_secondFNN), dim=1)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return x


# Read and organize input data
data = np.load("../data/HLHS_TV_data.npy", allow_pickle="TRUE")
coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]

coors_t0 = np.hstack((coors, np.zeros((len(coors), 1))))
coors_t1 = np.hstack((coors, 0.1 * np.ones((len(coors), 1))))

ux_mean, uy_mean, uz_mean = (
    np.mean(gt_disp[:, 0]),
    np.mean(gt_disp[:, 1]),
    np.mean(gt_disp[:, 2]),
)
ux_std, uy_std, uz_std = (
    np.std(gt_disp[:, 0]),
    np.std(gt_disp[:, 1]),
    np.std(gt_disp[:, 2]),
)

# Extract sampling points
idx1 = np.random.choice(np.where(coors_t0)[0], 2500, replace=False)
idx2 = np.random.choice(np.where(coors_t1)[0], 2500, replace=False)
pde_pts = np.vstack((coors_t0[idx1, :], coors_t1[idx2, :]))
pde_pts_disp = np.vstack((np.zeros((len(coors_t0[idx1, :]), 3)), gt_disp[idx2, :]))

geomtime = dde.geometry.PointCloud(points=pde_pts)

loss = [
    dde.PointSetBC(
        coors_t0[idx1, :], np.zeros((len(coors[idx1, :]), 3)), component=[0, 1, 2]
    ),
    dde.PointSetBC(
        coors_t1[idx2, :], coors_t1[idx2, :3] + gt_disp[idx2, :], component=[9, 10, 11]
    ),
]

# Model variables
E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)


def piola_kirchhoff_stress(x, y):
    """
    Compute the First Piola-Kirchhoff stress tensor from displacement field.
    Returns the six independent components: P_xx, P_yy, P_zz, P_xy, P_xz, P_yz
    """
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
    duxdy = dde.grad.jacobian(Nux, x, i=0, j=1)
    duxdz = dde.grad.jacobian(Nux, x, i=0, j=2)

    duydx = dde.grad.jacobian(Nuy, x, i=0, j=0)
    duydy = dde.grad.jacobian(Nuy, x, i=0, j=1)
    duydz = dde.grad.jacobian(Nuy, x, i=0, j=2)

    duzdx = dde.grad.jacobian(Nuz, x, i=0, j=0)
    duzdy = dde.grad.jacobian(Nuz, x, i=0, j=1)
    duzdz = dde.grad.jacobian(Nuz, x, i=0, j=2)

    # Deformation gradient F = I + grad(u)
    Fxx = duxdx + 1.0
    Fxy = duxdy
    Fxz = duxdz

    Fyx = duydx
    Fyy = duydy + 1.0
    Fyz = duydz

    Fzx = duzdx
    Fzy = duzdy
    Fzz = duzdz + 1.0

    # Determinant of F
    detF = (
        Fxx * (Fyy * Fzz - Fyz * Fzy)
        - Fxy * (Fyx * Fzz - Fyz * Fzx)
        + Fxz * (Fyx * Fzy - Fyy * Fzx)
    )

    detF = torch.where(torch.le(detF, 0), 0.001, detF)

    # Compute F^(-T) via adjugate
    adjFxx = Fyy * Fzz - Fyz * Fzy
    adjFxy = -(Fxy * Fzz - Fxz * Fzy)
    adjFxz = Fxy * Fyz - Fxz * Fyy

    adjFyx = -(Fyx * Fzz - Fyz * Fzx)
    adjFyy = Fxx * Fzz - Fxz * Fzx
    adjFyz = -(Fxx * Fyz - Fxz * Fyz)

    adjFzx = Fyx * Fzy - Fzx * Fyy
    adjFzy = -(Fxx * Fzy - Fxy * Fzx)
    adjFzz = Fxx * Fyy - Fxy * Fyx

    invFxx = adjFxx / detF
    invFxy = adjFxy / detF
    invFxz = adjFxz / detF

    invFyx = adjFyx / detF
    invFyy = adjFyy / detF
    invFyz = adjFyz / detF

    invFzx = adjFzx / detF
    invFzy = adjFzy / detF
    invFzz = adjFzz / detF

    # Material parameters
    E = (torch.tanh(E_) + 1.0) * 400
    nu = (torch.tanh(nu_) + 1.0) / 4

    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # Compressible Neo-Hookean 1st Piola-Kirchhoff Stress: 
    # P = mu*F + [lambda * log(detF) - mu] * F^(-T)
    # Derivation based on strain energy function from
    # Javier Bonet, Richard D. Wood. Nonlinear continuum mechanics for
    # finite element analysis. Cambridge University Press, 1997.

    lnF = torch.log(detF)

    Pxx = mu * Fxx + (lmbd * lnF - mu) * invFxx
    Pyy = mu * Fyy + (lmbd * lnF - mu) * invFyy
    Pzz = mu * Fzz + (lmbd * lnF - mu) * invFzz
    Pxy = mu * Fxy + (lmbd * lnF - mu) * invFyx
    Pxz = mu * Fxz + (lmbd * lnF - mu) * invFzx
    Pyz = mu * Fyz + (lmbd * lnF - mu) * invFzy

    return Pxx, Pyy, Pzz, Pxy, Pxz, Pyz


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
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    NPxx, NPyy, NPzz, NPxy, NPxz, NPyz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    # Compute Piola-Kirchhoff stress from constitutive law
    Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = piola_kirchhoff_stress(x, y)

    # Compute divergence of network Piola-Kirchhoff stress
    # div(P) for momentum balance: dP_ij/dX_j
    dPxx_dx = dde.grad.jacobian(NPxx, x, i=0, j=0)
    dPxy_dy = dde.grad.jacobian(NPxy, x, i=0, j=1)
    dPxz_dz = dde.grad.jacobian(NPxz, x, i=0, j=2)

    dPxy_dx = dde.grad.jacobian(NPxy, x, i=0, j=0)
    dPyy_dy = dde.grad.jacobian(NPyy, x, i=0, j=1)
    dPyz_dz = dde.grad.jacobian(NPyz, x, i=0, j=2)

    dPxz_dx = dde.grad.jacobian(NPxz, x, i=0, j=0)
    dPyz_dy = dde.grad.jacobian(NPyz, x, i=0, j=1)
    dPzz_dz = dde.grad.jacobian(NPzz, x, i=0, j=2)

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

    return [
        mx,
        my,
        mz,
        stress_xx,
        stress_yy,
        stress_zz,
        stress_xy,
        stress_xz,
        stress_yz,
    ]


net = MPFNN([4, 32, 16, 8, 3], [4, 32, 16, 8, 6], "swish", "Glorot normal")


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

    Nux = Nux * ux_std + ux_mean
    Nuy = Nuy * uy_std + uy_mean
    Nuz = Nuz * uz_std + uz_mean

    NPxx, NPyy, NPzz, NPxy, NPxz, NPyz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    Nux_new = Nux + x[:, 0:1]
    Nuy_new = Nuy + x[:, 1:2]
    Nuz_new = Nuz + x[:, 2:3]

    return torch.concat(
        [Nux, Nuy, Nuz, NPxx, NPyy, NPzz, NPxy, NPxz, NPyz, Nux_new, Nuy_new, Nuz_new],
        axis=1,
    )


def hausdorff_distance(y_true, y_pred):
    distances = torch.cdist(y_pred[:, :3], y_true[:, :3], p=2)
    avg_distances_1 = torch.mean(
        torch.min(distances, dim=1).values
    )  # Max of min distances from 1 to 2
    avg_distances_2 = torch.mean(
        torch.min(distances, dim=0).values
    )  # Max of min distances from 2 to 1
    error = 0.5 * (avg_distances_1 + avg_distances_2)

    return error


net.apply_output_transform(output_transform)
data = dde.data.PDE(geomtime, pde, loss, anchors=pde_pts)

model = dde.Model(data, net)
loss_type = ["MSE"] * 10 + [hausdorff_distance]
model = dde.Model(data, net)
external_trainable_variables = [E_, nu_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

model.compile(
    "adam",
    loss=loss_type,
    lr=1e-3,
    decay=["step", 15000, 0.10],
    loss_weights=[1e1] * 10 + [1] * 1,
    external_trainable_variables=external_trainable_variables,
)

losshistory, train_state = model.train(
    epochs=250000, display_every=1000, callbacks=[variables]
)
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

import re

lines = open("variables.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = vkinfer.shape
E_pred = (np.tanh(vkinfer[:, 0]) + 1) * 400
nu_pred = (np.tanh(vkinfer[:, 1]) + 1) / 4

print("E prediction: ", E_pred[-1])
print("nu prediction: ", nu_pred[-1])
