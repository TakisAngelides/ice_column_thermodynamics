#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 01:43:41 2023

@author: dmoreno
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import special
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import mpmath
mpmath.mp.dps = 15  # Set the desired decimal precision


# Paths
path_fig = "/home/dmoreno/figures/fourier_analytical/"
save_fig = False

# Options.
#calculate_lmbda = True

# Plot options.
stat_profiles = True
stat_simple   = False
full_profiles = False
energy        = False
plot_lmbd     = False
heat_map      = False
numerical     = False
eigenvalues   = False

# Seconds in a year.
#const = 60.0 * 60.0 * 24.0 * 365.0
const = 31556926.0  # EISMINT.

# Physical constants
K = 2.0                 # Thermal conductivity [W / (mºC)], 2.0
G = 0.05                # Geothermal heat flux [W] 0.05
kappa = 1.4e-6          # Thermal diffusivity [m^2/s]


# Temperatures.
T_L = -20.0           # Ice surface temperature [ºC]
T_sl = -10.0           # Ice basal temperature [ºC]
T_air = -30.0           # Air temperature [ºC] -20.0


# Boundary conditions.
beta = 50.0            # Insulating parameter [m]. 50.0, 100.0

# Vertical advection
# Numerical computation of eigenvalue is extremly sensitive to w_0.
# Make sure that all eigenvalues are captured if w_0 is changed.
w_0 = 0.2 / const     # Surface vertical advection. [m/yr] --> [m/s] 0.2.


# Define the non-dimensional variables.
n_t = 100  # 50, 100, 10 for stationary with multiplots.
n = 100  # 50, 100
L = 1000
t = np.linspace(0.0, 1.1e5, n_t) * const    # Time in seconds. 1.0e5
xi = np.linspace(0.0, 1.0, n)
#tau = t * (kappa / L**2)

# For visualizsation purposes.
n_exp = 2  # 2
tau_max = 1.05  # 1.0,

tau = np.linspace(0.0, 1.0, n_t)  # (0.0, 27.0, n_t)
tau = tau**n_exp
tau = tau_max * tau


# NON-DIMENSIONALIZATION.
# Vertical advection.
w_0_bar = w_0 * L / kappa    # Adimensional surface vertical advection. []
w_0_bar = 5.0                # 5.0
w_bar = w_0_bar * xi       # Linear dependency from w_0 to 0 at the base.
#print("w_0_bar = ", w_0_bar)

# STRAIN HEAT TERM.
strain = 1.0e-3  # 0.15, 5.0e-1
br = strain * L**2 / (abs(T_air + 273.15) * const * kappa)
strain = br
#print('Br = ', br)

# Boundary conditions.
#G_k = G * L / ( K * T_air )
G_k = -1.0

# Insulating parameter
beta_L = beta / L


# Function to calculate the dimensionless parameter from physical magnitudes.
def params(L, T_air, kappa, w_0, K, beta, G, Q_fric, u, dtheta, strain_eff):
    
    # Difussivity.
    kappa_yr = kappa * const       # [m^/yr]

    # Dimensionless definitions.
    beta_L = beta / L

    # Peclét.
    Pe = w_0 * L / kappa_yr

    # Dimensionless geothermal heat flow.
    gamma = - ( G + Q_fric ) * L / ( K * T_air )

    # Dimensionless strain heat dissipation: Br.
    Br = - strain_eff * L**2 / ( kappa_yr * T_air ) 
    
    # Dimensionless lateral advection: \Lambda.
    # Horizontal advection.
    adv_h    = u * dtheta 
    Lambda_h = adv_h * L**2 / ( kappa_yr * T_air ) 
    

    return [Pe, beta_L, gamma, Br, Lambda_h]


# Ice thickness.
L   = 2.0e3        # [m]

# Vertical advection at the ice surface.
w_0 = 0.3          # [m/yr]

# Geothermal heat flow.
G   = 0.05        # [W/m²]

# Frictional heat (referred to G).
# Four tau_b = 100kPa, u_b=10-20 m/yr already matches G value.
Q_fric = 10.0 * G

# Atmospheric temperature.
T_air = 223.15        # 223.15 K

# Effective strain rate. Meyer and Minchsew (2018): 0.1 [1/yr].
strain_eff = 1.0e-2  # 0.2, 5.5. From Nix: du/dx ~ 10^-4 to 10^-2 yr^-1.

# Surface insulation.
beta = 500.0 # [m]

# Horizontal advection.
u      = 300.0       # [m/yr]
dtheta = 0.2e-3       # 0.2 degrees per km in horizontal. [K/m]

# Thermal conductivity [W / (mºC)].
K = 2.0                 

# Difussivity.
rho = 910.0  # Km/m³
c_p = 2009.0  # J / (kg K)
kappa = K / (rho * c_p)     # [m²/s]


# Call function.
Pe, beta_L, gamma, Br, Lambda_h = params(L, T_air, kappa, w_0, K, beta, G, Q_fric, u, dtheta, strain_eff)

print('Pe       = ', np.round(Pe, 3))
print('beta_L   = ', np.round(beta_L, 3))
print('gamma    = ', np.round(gamma, 3))
print('Br       = ', np.round(Br, 3))
print('Lambda_h = ', np.round(Lambda_h, 3))




###################################################
# We now create variable mesh.
XI, TAU = np.meshgrid(xi, tau)

# Stationary solution.
# Auxiliar variables.
a = (0.5 * w_0_bar)**0.5

# Truncation order.
order = 50  # 30


def f_adv(x, w_0, a):
    y = - w_0 * x**a

    return y


def f_hyp2f2(b_1, b_2, b_3, b_4, zeta):

    # Shape.
    s = np.shape(zeta)
    l_s = len(s)
    #s = len(zeta)

    if l_s == 0:
        result = mpmath.hyp2f2(b_1, b_2, b_3, b_4, zeta)

    elif l_s == 1:
        # Initialize an empty array to store the results
        result = np.empty(s)

        # Iterate over each element in the z_array
        for i in range(s[0]):
            result[i] = mpmath.hyp2f2(b_1, b_2, b_3, b_4, zeta[i])

    elif l_s == 2:
        # Initialize an empty array to store the results
        result = np.empty((s[0], s[1]))

        # Iterate over each element in the z_array
        for i in range(s[0]):
            for j in range(s[1]):
                result[i, j] = mpmath.hyp2f2(b_1, b_2, b_3, b_4, zeta[i, j])

    # Convert the results list to a NumPy array
    #results_array = np.array(result)

    return result


# STATIONARY SOLUTIONS. Clarke et al. (1977).
# FUNCTION THAT TAKES A GIVEN ADVECTION VALUE AND RETURNS STATIONARY PROFILES.
# Eq. 11 (Clarke, 1977).
def stat_sol(w_s, G_k, beta_L, strain, m, adv, xi, n):

    # Analytical with my coordinate system.
    erf_fact = (0.5 * np.pi / w_s)**0.5

    # Solution without strain heating term.
    # Downwards.
    B_down = 1.0 - G_k * (beta_L * np.exp(- 0.5 * w_s) + erf_fact *
                          special.erf((0.5 * w_s)**0.5))

    u_down_an = G_k * erf_fact * special.erf((0.5 * w_s)**0.5 * xi) + B_down

    # Upwards vertical advection.
    B_up = 1.0 - G_k * (beta_L * np.exp(0.5 * w_s) + erf_fact *
                        special.erfi((0.5 * w_s)**0.5))

    u_up_an = G_k * erf_fact * special.erfi((0.5 * w_s)**0.5 * xi) + B_up

    # Linear advection profile and inhomogeneous correction (strain or vertically-averaged horizontal
    # advection).
    if adv == 'linear':
        # Generalised hypergeometric parameters.
        b_1 = 1.0
        b_2 = 1.0
        b_3 = 3.0/2.0
        b_4 = 2.0
        zeta = -0.5 * w_s * xi**2

        # Handy definitions.
        a = (0.5 * w_s)**0.5  # (0.5 * w_s)**0.5
        hyp2f2 = f_hyp2f2(b_1, b_2, b_3, b_4, zeta)
        hyp2f2_plus = f_hyp2f2(b_1+1, b_2+1, b_3+1, b_4+1, zeta)
        fact_hyp = 1.0 / 3.0

        # Inhomogeneous surface boundary condition.  hyp2f2[n-1]
        B_down = 1.0 - G_k * (beta_L * np.exp(-a**2) + erf_fact * special.erf(a)) - \
            strain * ((beta_L + 0.5) * hyp2f2[n-1] -
                      beta_L * a**2 * fact_hyp * hyp2f2_plus[n-1])

        u_down_an = B_down + G_k * erf_fact * special.erf(a * xi) + \
            0.5 * strain * xi**2 * hyp2f2

        # Opposite vertical advection sign.
        zeta = 0.5 * w_s * xi**2
        hyp2f2 = f_hyp2f2(b_1, b_2, b_3, b_4, zeta)
        hyp2f2_plus = f_hyp2f2(b_1+1, b_2+1, b_3+1, b_4+1, zeta)

        # Inhomogeneous surface boundary condition.
        B_up = 1.0 - G_k * (beta_L * np.exp(a**2) + erf_fact * special.erfi(a)) + \
            - strain * ((beta_L + 0.5) * hyp2f2[n-1] +
                        beta_L * a**2 * fact_hyp * hyp2f2_plus[n-1])

        u_up_an = B_up + G_k * erf_fact * special.erfi(a * xi) + \
            0.5 * strain * xi**2 * hyp2f2

    # General power-law advective profile. No strain heating contribution (i.e., strain = 0)
    # to express in terms of the incomplete Gamma function.
    # w(x) = w_0 x^m.
    elif adv == 'power-law':

        m_plus = m + 1.0
        p = 1.0 / m_plus
        p_w = p*w_s
        pre = p * G_k / p_w**p

        # Upper incomplete gamma function.
        # There is a mismath at the base here compared to EISMINT Fig. 3.
        B_down = 1.0 + pre * (m_plus * beta_L * p_w**p *
                              np.exp(-p_w) - special.gammaincc(p, p_w))

        u_down_an = B_down - pre * special.gammaincc(p, p_w * xi**m_plus)

        print('m_plus = ',  m_plus)
        print('B_down = ',  B_down)

        # Numerical solution for comparison.
        """integrand_down = lambda x: np.exp( -p * w_s * x**m_plus ) * \
                                    ( G_k - ( p**(1.0-p) * x * strain * \
                                                    special.gammaincc(p, p * w_s * x**m_plus ) ) / w_s )"""

    return [u_up_an, u_down_an]


# Find as many roots of the eigenvalue equation as truncation order.
def zeros_eigen(order, w_0_bar, beta_L, lambd_0, z, b_1, advection):

    def eigenequation(lambd_eigen, z, advection):
        # Update parameter hypergeometric function.
        if advection == 'up':
            a_1 = - 0.5 * lambd_eigen / w_0_bar
            y_F = special.hyp1f1(a_1, b_1, z)
            tol = 1.0e-1  # 1.0e-2, 5.0e-2
            pre = 0.25  # 1.0

        elif advection == 'down':
            a_1 = 0.5 - 0.5 * lambd_eigen / w_0_bar
            y_F = np.exp(-z) * special.hyp1f1(a_1, b_1, z)
            tol = 1.0e-2  # 1.0e-2
            pre = 1.0  # 2.0

        # Confluent hypergeometric functions.
        dz = 1.0 / n
        y_F_der = np.gradient(y_F, dz)

        # Eigenvalue equation for v(xi,tau).
        eigenval = beta_L * y_F_der + y_F

        # Factor to avoid numerical issues.
        eigenval = pre * eigenval

        return eigenval[n-1]

    # Store eigenvalues.
    lmbd = np.empty(order)
    y_lmbd = np.empty(order)

    # Initial guess
    c = 0
    lambd_eigen = lambd_0  # 0.1
    delta_lambd = 0.05

    while c < order:

        """
        # Update parameter hypergeometric function.
        if advection == 'up':
            a_1 = - 0.5 * lambd_eigen / w_0_bar 
            y_F = special.hyp1f1(a_1, b_1, z)
            tol = 1.0e-1 # 1.0e-2, 5.0e-2
            pre = 0.25 # 1.0

        elif advection == 'down':
            a_1 = 0.5 - 0.5 * lambd_eigen / w_0_bar 
            y_F = np.exp(-z) * special.hyp1f1(a_1, b_1, z)
            tol = 1.0e-2 # 1.0e-2
            pre = 1.0 # 2.0

        # Confluent hypergeometric functions.
        dz = 1.0 / n
        y_F_der = np.gradient(y_F, dz)

        # Eigenvalue equation for v(xi,tau).
        eigenval = beta_L * y_F_der + y_F

        # Factor to avoid numerical issues.
        eigenval = pre * eigenval"""

        eigenval = eigenequation(lambd_eigen, z, advection)
        eigenval_delta = eigenequation(lambd_eigen+delta_lambd, z, advection)

        # Boundary condition at xi = 1.
        # if abs(eigenval[n-1]) < tol:
        if np.sign(eigenval) != np.sign(eigenval_delta):

            #print('Eigenvalue order = ', c)
            #lmbd[c]   = lambd_eigen
            # y_lmbd[c] = eigenval[n-1]
            lmbd[c] = lambd_eigen + 0.5 * delta_lambd
            y_lmbd[c] = 0.5 * (eigenval + eigenval_delta)

            # We use the gap between two previous zeros since spacing increases with lambda (plot).
            if c == 0:
                lambd_eigen += 5.0  # 2.0, 5.0
            else:
                lambd_eigen += lmbd[c] - lmbd[c-1]

            c += 1

        # Update eigenvalue.
        #lambd_eigen += 0.1
        lambd_eigen += delta_lambd

    return [lmbd, y_lmbd]


def full_sol(plot_lmbd, order, w_0_bar, XI, G_k, strain, beta_L, theta_0, n, n_t):

    # Hypegeometric function variable.
    b_1 = 0.5
    z = 0.5 * w_0_bar * xi**2
    #zeta = 0.5 * w_0_bar * XI**2

    # Eigenvalues to plot.
    lambd_0 = 1.0e-6  # 0.1

    # Calculate zeros in eigenvalue equation.
    lmbd_up, y_lmbd_up = zeros_eigen(
        order, w_0_bar, beta_L, lambd_0, z, b_1, 'up')
    lmbd_down, y_lmbd_down = zeros_eigen(
        order, w_0_bar, beta_L, lambd_0, z, b_1, 'down')

    # EIGENVALUE PLOTS
    if plot_lmbd == True:

        # Calculate eigenvalue of the problem.
        # Define eigenvalue equation to make plots.
        def f_eigen(z, lambd_eigen, advection):

            # Update parameter hypergeometric function.
            if advection == 'up':
                a_1 = - 0.5 * lambd_eigen / w_0_bar
                y_F = special.hyp1f1(a_1, b_1, z)

            elif advection == 'down':
                a_1 = 0.5 - 0.5 * lambd_eigen / w_0_bar
                y_F = np.exp(-z) * special.hyp1f1(a_1, b_1, z)

            # Confluent hypergeometric functions.
            # Spacing in our values.
            dz = 1.0 / n
            y_F_der = np.gradient(y_F, dz)

            # Eigenvalue equation.
            eigen = beta_L * y_F_der + y_F

            return eigen[n-1]

        # Array with lambda values.
        lambda_s = np.linspace(lambd_0, 1000.0, 1000)
        l_s = len(lambda_s)

        # Plot BC at xi = 1 for all lambda values.
        y_eigen_up = np.empty(l_s)
        y_eigen_down = np.empty(l_s)

        for i in range(l_s):
            y_eigen_up[i] = f_eigen(z, lambda_s[i], 'up')
            y_eigen_down[i] = f_eigen(z, lambda_s[i], 'down')

        fig = plt.figure(dpi=600, figsize=(8, 6))
        plt.rcParams['text.usetex'] = True
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # Eigenvalues plot.
        ax.plot(lambda_s, np.zeros(len(lambda_s)), linestyle=':', color='grey', marker='None',
                linewidth=2.0, alpha=1.0, label=r'$ y = 0 $')

        ax.plot(lambda_s, y_eigen_down, linestyle='-', color='darkgreen', marker='None',
                linewidth=2.0, alpha=1.0, label=r'$  \beta X_{\xi}^{-} +  X^{-} = 0 $')

        ax.plot(lmbd_down, y_lmbd_down, linestyle='None', color='darkgreen', marker='o', markersize=7.0,
                markerfacecolor='yellow', linewidth=2.0, alpha=1.0, label=r'$ \lambda^{-}_n $')

        # Eigenvalues plot down.
        ax2.plot(lambda_s, np.zeros(len(lambda_s)), linestyle=':', color='grey', marker='None',
                 linewidth=2.0, alpha=1.0, label=r'$ y = 0 $')

        ax2.plot(lambda_s, y_eigen_up, linestyle='-', color='purple', marker='None',
                 linewidth=2.0, alpha=1.0, label=r'$ \beta X_{\xi}^{+} +  X^{+} = 0 $')

        ax2.plot(lmbd_up, y_lmbd_up, linestyle='None', color='purple', marker='o', markersize=7.0,
                 markerfacecolor='lightblue', linewidth=2.0,
                 alpha=1.0, label=r'$ \lambda^{+}_n $')

        ax2.plot(lambda_s, y_eigen_down, linestyle='-', color='darkgreen', marker='None',
                 linewidth=2.0, alpha=0.5)

        ax2.plot(lmbd_up, y_lmbd_down, linestyle='None', color='darkgreen', marker='o', markersize=7.0,
                 markerfacecolor='yellow', linewidth=2.0, alpha=0.5)

        # Add labels and title to plot
        ax.set_ylabel(r'$ y $', fontsize=20)
        ax.grid(axis='x', which='major', alpha=0.85)

        ax.set_xlim(0.0, 400)

        ax2.set_ylabel(r'$ y $', fontsize=20)
        ax2.set_xlabel(r'$ \lambda $', fontsize=20)
        ax2.grid(axis='x', which='major', alpha=0.85)

        ax2.set_xlim(0.0, 400)

        ax.legend(loc='lower right', ncol=1, frameon=True, framealpha=1.0,
                  fontsize=12, fancybox=True)

        ax2.legend(loc='lower right', ncol=1, frameon=True, framealpha=1.0,
                   fontsize=12, fancybox=True)

        # Save figure.
        if save_fig == True:
            name_fig = 'eigenvalues'
            plt.savefig(path_fig+name_fig+'.png',
                        format="png", bbox_inches='tight')
            plt.savefig(path_fig+name_fig+'.pdf',
                        format="pdf", bbox_inches="tight")

        # Display plot
        plt.show()
        plt.close(fig)

    # Fourier coefficients.
    C_n_up = np.empty(order)
    C_n_down = np.empty(order)

    # Intialize transitory solution for two advection cases.
    v_up = np.zeros([n_t, n])
    v_down = np.zeros([n_t, n])

    # Handy definitions
    erf_fact = (0.5 * np.pi / w_0_bar)**0.5
    A = G_k * erf_fact
    a = (0.5 * w_0_bar)**0.5
    fact_hyp = 1.0 / 3.0

    # Generalised hypergeometric parameters.
    b_1 = 1.0
    b_2 = 1.0
    b_3 = 3.0/2.0
    b_4 = 2.0
    zeta_up = 0.5 * w_0_bar * XI**2
    zeta_down = - 0.5 * w_0_bar * XI**2

    # Function definition.
    hyp2f2_up = f_hyp2f2(b_1, b_2, b_3, b_4, zeta_up)
    hyp2f2_down = f_hyp2f2(b_1, b_2, b_3, b_4, zeta_down)
    hyp2f2_up_plus = f_hyp2f2(b_1+1, b_2+1, b_3+1, b_4+1, zeta_up)
    hyp2f2_down_plus = f_hyp2f2(b_1+1, b_2+1, b_3+1, b_4+1, zeta_down)

    # Analytical solutions for the stationary case.
    # Constant from boundary condition.
    B_up = 1.0 - G_k * (beta_L * np.exp(a**2) + erf_fact * special.erfi(a)) \
        - strain * ((beta_L + 0.5) * hyp2f2_up[n-1, n_t-1]
                    + beta_L * a**2 * fact_hyp * hyp2f2_up_plus[n-1, n_t-1])

    # Constant from boundary condition.
    B_down = 1.0 - G_k * (beta_L * np.exp(-a**2) + erf_fact * special.erf(a)) \
        - strain * ((beta_L + 0.5) * hyp2f2_down[n-1, n_t-1]
                    - beta_L * a**2 * fact_hyp * hyp2f2_down_plus[n-1, n_t-1])

    # Stationary solutions.
    u_up_an = B_up + A * \
        special.erfi(a * XI) + 0.5 * strain * XI**2 * hyp2f2_up
    u_down_an = B_down + A * \
        special.erf(a * XI) + 0.5 * strain * XI**2 * hyp2f2_down

    # Confluent hypergeometric function parameters.
    delta = 0.5
    a_1_up = - 0.5 * lmbd_up / w_0_bar
    a_1_down = 0.5 - 0.5 * lmbd_down / w_0_bar  # 0.5 - 0.5 * lmbd_down / w_0_bar

    # Fourier series.
    for i in range(0, order, 1):

        # The inital condition is on v(xi,0) = theta(xi,0) - u(xi)
        # Upwards.
        def phi_up_an(x): return (theta_0 - A * special.erfi(a*x) - B_up
                                  - 0.5 * strain * x**2 * mpmath.hyp2f2(b_1, b_2, b_3, b_4, (a*x)**2)) \
            * special.hyp1f1(a_1_up[i], delta, (a*x)**2) \
            * np.exp(-(a*x)**2)

        # Downwards. 1.5
        # BE CAREFUL WITH np.exp(-(a*x)**2). I do not fully understand it.
        def phi_down_an(x): return (theta_0 - A * special.erf(a*x) - B_down
                                    - 0.5 * strain * x**2 * mpmath.hyp2f2(b_1, b_2, b_3, b_4, -(a*x)**2)) \
            * special.hyp1f1(a_1_down[i], delta, (a*x)**2)

        # Normalization coefficient.

        def phi_norm_up(x): return special.hyp1f1(a_1_up[i], delta, (a*x)**2)**2 \
            * np.exp(-(a*x)**2)

        def phi_norm_down(x): return special.hyp1f1(a_1_down[i], delta, (a*x)**2)**2 \
            * np.exp(-(a*x)**2)

        # Normalization factor.
        norm_up   = quad(phi_norm_up, 0.0, 1.0)[0]
        norm_down = quad(phi_norm_down, 0.0, 1.0)[0]

        # Coefficients from orthogonality.
        C_n_up[i]   = quad(phi_up_an, 0.0, 1.0)[0] / norm_up
        C_n_down[i] = quad(phi_down_an, 0.0, 1.0)[0] / norm_down

        # Time-dependent solution.
        v_up += C_n_up[i] * special.hyp1f1(a_1_up[i],
                                           delta, zeta_up) * np.exp(-lmbd_up[i] * TAU)

        v_down += C_n_down[i] * special.hyp1f1(a_1_down[i], delta, zeta_up) * np.exp(-lmbd_down[i] * TAU) \
            * np.exp(zeta_down)

    # Full solutions: stationary + transitory.
    theta_up = u_up_an + v_up
    theta_down = u_down_an + v_down

    return [theta_up, theta_down, u_up_an, u_down_an]


# Function to obtain the energy content from a temperature profile.
def Q_int(sol, n_energy, tau, theta_0, n_w):

    Q_up = np.empty([n_w, n_Q])
    Q_down = np.empty([n_w, n_Q])
    tau_Q = np.empty(n_Q)

    for i in range(n_w):

        # Calculate full problem solutions.
        theta_up = sol[i][0][:, :]
        theta_down = sol[i][1][:, :] - sol[i][1][n_t-1, :]

        c_Q = 0

        # Integration to obtain energy content.
        for j in n_energy:

            # Energy content in the temperature profile for a given time frame j.
            Q_up[i, c_Q] = np.trapz(theta_up[j, :], xi)
            Q_down[i, c_Q] = np.trapz(theta_down[j, :], xi)

            tau_Q[c_Q] = tau[j]

            c_Q += 1

    return [Q_up, Q_down, tau_Q]


#########################################################################################
#########################################################################################
# MULTIPLOT STATIONARY SOLUTIONS.

if stat_profiles == True:

    # No beta dependency at the surface (fixed temperature here).
    n_w = 6
    w_0_bar_min = 0.1
    w_0_bar_max = 8.0
    #w_0_bar_s   = np.linspace(w_0_bar_min, w_0_bar_max, n_w)

    # Colours array.
    col_w = np.linspace(0.0, 1.0, n_w)

    # Power-law formulation of velocity vertical profiles.
    m = 1.0
    adv = 'linear'

    # Array values for comparison.
    # negative values of strain_s imply a sink from a horizontal advection contribution.

    # Old values for dimensionless plots.
    """w_0_bar_s = np.array([0.01, 1.0, 3.0, 5.0, 7.0, 9.0])
    G_k_s = - np.array([0.125, 0.25, 0.5, 1.0, 1.5, 2.0])
    #beta_L_s  = np.array([0.0, 0.025, 0.05, 0.075, 0.10, 0.125])
    beta_L_s = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    strain_s = - np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5]
    strain_H_s = np.array([0.0, 1.5, 3.0, 4.5, 6.0, 7.5])"""

    # New values for dimensional plots..
    # negative values of strain_s imply a sink from a horizontal advection contribution.
    w_0_bar_s = np.array([0.01, 1.0, 3.0, 5.0, 7.0, 9.0])
    G_k_s = - np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
    #beta_L_s  = np.array([0.0, 0.025, 0.05, 0.075, 0.10, 0.125])
    beta_L_s = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    strain_s = - np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75])
    # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5]
    strain_H_s = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    # Calculate stationary solutions for each value of w_0_bar and G_k_s.
    stat      = []
    stat_G    = []
    stat_beta = []
    stat_strn = []
    stat_hor  = []

    for i in range(len(w_0_bar_s)):
        stat.append(stat_sol(w_0_bar_s[i], G_k_s[3],
                    beta_L_s[0], strain_s[0], m, adv, xi, n))

    for i in range(len(G_k_s)):
        stat_G.append(stat_sol(w_0_bar_s[4], G_k_s[i], beta_L_s[0], strain_s[0], m, adv, xi, n))

    for i in range(len(beta_L_s)):
        stat_beta.append(
            stat_sol(w_0_bar_s[4], G_k_s[3], beta_L_s[i], strain_s[0], m, adv, xi, n))

    for i in range(len(strain_s)):
        stat_strn.append(
            stat_sol(w_0_bar_s[4], G_k_s[1], beta_L_s[0], strain_s[i], m, adv, xi, n))

    for i in range(len(strain_H_s)):
        print('Calculating solution # = ', i)
        stat_hor.append(stat_sol(
            w_0_bar_s[4],  G_k_s[5], beta_L_s[0], strain_H_s[i], m, adv, xi, n))  # -3.0

    
    # Convert to dimensional magnitudes express in ºC.
    T_air = 223.15 # [K]
    stat      = T_air * np.array(stat) - 273.15
    stat_G    = T_air * np.array(stat_G) - 273.15
    stat_beta = T_air * np.array(stat_beta) - 273.15
    stat_strn = T_air * np.array(stat_strn) - 273.15
    stat_hor  = 263.15 * np.array(stat_hor) - 273.15
    
    # MULTIPLOT STATIONARY UPWARDS.
    fig = plt.figure(dpi=600, figsize=(14, 14))
    plt.rcParams['text.usetex'] = True

    ax = fig.add_subplot(241)
    ax2 = fig.add_subplot(243)
    ax3 = fig.add_subplot(242)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)

    # Plot stationary solutions
    for i in range(len(w_0_bar_s)):

        # STATIONARY SOLUTION FOR DIFFERENT PECLÉT NUMBERS.
        # Stationary solution imposing non-zero beta.
        # stat[i][1][0,:]
        ax.plot(stat[i][0][:], xi, linestyle='-', color=[col_w[n_w-1-i], 0.5*col_w[n_w-1-i], 0.0],
                marker='None', linewidth=2.75, alpha=1.0,
                label=r'$ \mathrm{Pe} = '+str(np.round(abs(w_0_bar_s[i]), 2))+' $')

        # Stationary solution imposing non-zero beta.
        ax2.plot(stat_G[i][0][:], xi, linestyle='-', color=[col_w[n_w-1-i], 0.0, 0.5*col_w[n_w-1-i]],
                 marker='None', linewidth=2.75, alpha=1.0,
                 label=r'$ \gamma = '+str(np.round(G_k_s[i], 3))+' $')

        # Stationary solution imposing non-zero beta.
        ax3.plot(stat_beta[i][0][:], xi, linestyle='-', color=[0.0, 0.5*col_w[n_w-1-i], col_w[n_w-1-i]],
                 marker='None', linewidth=2.75, alpha=1.0,
                 label=r'$ \beta = '+str(np.round(beta_L_s[i], 3))+' $')

        # Stationary solution imposing non-zero beta.
        # ax4.plot(stat_strn[i][2][n_t-1,:], xi, linestyle='-', color=[0.5*col_w[n_w-1-i],0.0,col_w[n_w-1-i]], \
        #                marker='None', linewidth=2.75, alpha=1.0, \
        #                    label=r'$ \mathrm{Br} = '+str(np.round(abs(strain_s[i]),2))+' $')
        ax4.plot(stat_strn[i][0][:], xi, linestyle='-', color=[0.5*col_w[n_w-1-i], 0.0, col_w[n_w-1-i]],
                 marker='None', linewidth=2.75, alpha=1.0,
                 label=r'$ \mathrm{Br} = '+str(np.round(strain_s[i], 2))+' $')

        # DOWNWARDS.
        # Stationary solution imposing non-zero beta.
        ax5.plot(stat[i][1][:], xi, linestyle='-', color=[col_w[n_w-1-i], 0.5*col_w[n_w-1-i], 0.0], marker='None',
                 linewidth=2.75, alpha=1.0, label=r'$ \mathrm{Pe} = '+str(np.round(abs(w_0_bar_s[i]), 2))+' $')

        # Stationary solution imposing non-zero beta.
        ax7.plot(stat_G[i][1][:], xi, linestyle='-', color=[col_w[n_w-1-i], 0.0, 0.5*col_w[n_w-1-i]], marker='None',
                 linewidth=2.75, alpha=1.0, label=r'$ \gamma = '+str(np.round(G_k_s[i], 3))+' $')

        # Stationary solution imposing non-zero beta.
        ax6.plot(stat_beta[i][1][:], xi, linestyle='-', color=[0.0, 0.5*col_w[n_w-1-i], col_w[n_w-1-i]], marker='None',
                 linewidth=2.75, alpha=1.0, label=r'$ \beta = '+str(np.round(beta_L_s[i], 3))+' $')

        # Stationary solution imposing non-zero beta.
        # ax8.plot(stat_strn[i][1][n_t-1,:], xi, linestyle='-', color=[0.5*col_w[n_w-1-i],0.0,col_w[n_w-1-i]], \
        #                marker='None', linewidth=2.75, alpha=1.0, \
        #                    label=r'$ \mathrm{Br} = '+str(np.round(abs(strain_s[i]),2))+' $')
        ax8.plot(stat_strn[i][1][:], xi, linestyle='-', color=[0.5*col_w[n_w-1-i], 0.0, col_w[n_w-1-i]],
                 marker='None', linewidth=2.75, alpha=1.0,
                 label=r'$ \mathrm{Br} = '+str(np.round(strain_s[i], 2))+' $')

    # Add labels and title to plot
    ax.set_xlabel(r'$ \vartheta^{+}(\xi)$', fontsize=20)
    ax2.set_xlabel(r'$ \vartheta^{+}(\xi) $', fontsize=20)
    ax3.set_xlabel(r'$ \vartheta^{+}(\xi) $', fontsize=20)
    ax4.set_xlabel(r'$ \vartheta^{+}(\xi) $', fontsize=20)

    ax5.set_xlabel(r'$ \vartheta^{-}(\xi)$', fontsize=20)
    ax6.set_xlabel(r'$ \vartheta^{-}(\xi) $', fontsize=20)
    ax7.set_xlabel(r'$ \vartheta^{-}(\xi) $', fontsize=20)
    ax8.set_xlabel(r'$ \vartheta^{-}(\xi) $', fontsize=20)

    ax.set_ylabel(r'$ \xi \ $', fontsize=20)
    ax5.set_ylabel(r'$ \xi \ $', fontsize=20)

    ax.grid(axis='y', which='major', alpha=0.85)
    ax2.grid(axis='y', which='major', alpha=0.85)
    ax3.grid(axis='y', which='major', alpha=0.85)
    ax4.grid(axis='y', which='major', alpha=0.85)
    ax5.grid(axis='y', which='major', alpha=0.85)
    ax6.grid(axis='y', which='major', alpha=0.85)
    ax7.grid(axis='y', which='major', alpha=0.85)
    ax8.grid(axis='y', which='major', alpha=0.85)

    # Ticks.
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])
    ax6.set_yticklabels([])
    ax7.set_yticklabels([])
    ax8.set_yticklabels([])

    ax2.tick_params(axis='y', which='major', length=0, colors='black')
    ax3.tick_params(axis='y', which='major', length=0, colors='black')
    ax4.tick_params(axis='y', which='major', length=0, colors='black')
    ax6.tick_params(axis='y', which='major', length=0, colors='black')
    ax7.tick_params(axis='y', which='major', length=0, colors='black')
    ax8.tick_params(axis='y', which='major', length=0, colors='black')

    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['$0$', '$0.2$', '$0.4$', '$0.6$',
                       '$0.8$', '$1.0$'], fontsize=16)

    ax.set_xticks([0, 6.0, 12.0, 18.0])
    ax.set_xticklabels(['$0$', '$6$', '$12$', '$18$'], fontsize=16)

    ax2.set_xticks([0, 6.0, 12.0, 18.0])
    ax2.set_xticklabels(['$0$', '$6$', '$12$', '$18$'], fontsize=16)

    ax3.set_xticks([0, 6.0, 12.0, 18.0])
    ax3.set_xticklabels(['$0$', '$6$', '$12$', '$18$'], fontsize=16)

    ax4.set_xticks([0, 6.0, 12.0, 18.0])
    ax4.set_xticklabels(['$0$', '$6$', '$12$', '$18$'], fontsize=16)

    ax5.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.set_yticklabels(
        ['$0.0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$'], fontsize=16)

    ax5.set_xticks([1.0, 1.4, 1.8, 2.2])
    ax5.set_xticklabels(['$1.0$', '$1.4$', '$1.8$', '$2.2$'], fontsize=16)

    ax6.set_xticks([1.0, 1.4, 1.8, 2.2])
    ax6.set_xticklabels(['$1.0$', '$1.4$', '$1.8$', '$2.2$'], fontsize=16)

    ax7.set_xticks([1.0, 1.4, 1.8, 2.2])
    ax7.set_xticklabels(['$1.0$', '$1.4$', '$1.8$', '$2.2$'], fontsize=16)

    ax8.set_xticks([1.0, 1.4, 1.8, 2.2])
    ax8.set_xticklabels(['$1.0$', '$1.4$', '$1.8$', '$2.2$'], fontsize=16)

    # Limits
    ax.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)

    ax5.set_ylim(0.0, 1.0)
    ax6.set_ylim(0.0, 1.0)
    ax7.set_ylim(0.0, 1.0)
    ax8.set_ylim(0.0, 1.0)

    ax.set_xlim(0.0, 18.0)
    ax2.set_xlim(0.0, 18.0)
    ax3.set_xlim(0.0, 18.0)
    ax4.set_xlim(0.0, 18.0)

    ax5.set_xlim(1.0, 2.2)
    ax6.set_xlim(1.0, 2.2)
    ax7.set_xlim(1.0, 2.2)
    ax8.set_xlim(1.0, 2.2)

    ax5.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=13, fancybox=True)

    ax6.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=13, fancybox=True)

    ax7.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=13, fancybox=True)

    ax8.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=13, fancybox=True)

    # Title.
    ax.set_title(r'$ \mathrm{(a)} $', fontsize=20, loc='left', pad=10)
    ax2.set_title(r'$ \mathrm{(b)} $', fontsize=20, loc='left', pad=10)
    ax3.set_title(r'$ \mathrm{(c)} $', fontsize=20, loc='left', pad=10)
    ax4.set_title(r'$ \mathrm{(d)} $', fontsize=20, loc='left', pad=10)
    ax5.set_title(r'$ \mathrm{(e)} $', fontsize=20, loc='left', pad=10)
    ax6.set_title(r'$ \mathrm{(f)} $', fontsize=20, loc='left', pad=10)
    ax7.set_title(r'$ \mathrm{(g)} $', fontsize=20, loc='left', pad=10)
    ax8.set_title(r'$ \mathrm{(h)} $', fontsize=20, loc='left', pad=10)

    # Save figure.
    if save_fig == True:
        name_fig = 'stationary'
        plt.savefig(path_fig+name_fig+'.png',
                    format="png", bbox_inches='tight')
        plt.savefig(path_fig+name_fig+'.pdf',
                    format="pdf", bbox_inches="tight")

    # Display plot
    plt.show()
    plt.close(fig)

    # MULTIPLOT STATIONARY DOWNWARDS.
    fig = plt.figure(dpi=600, figsize=(18, 8))

    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(151)
    ax2 = fig.add_subplot(152)
    ax3 = fig.add_subplot(153)
    ax4 = fig.add_subplot(154)
    ax5 = fig.add_subplot(155)

    # Plot stationary solutions
    for i in range(len(w_0_bar_s)):

        # Stationary solution imposing non-zero beta.
        ax.plot(stat[i][1], xi, linestyle='-', color=[col_w[n_w-1-i], 0.5*col_w[n_w-1-i], 0.0], marker='None',
                linewidth=2.75, alpha=1.0, label=r'$ \mathrm{Pe} = '+str(np.round(abs(w_0_bar_s[i]), 2))+' $')

        # Stationary solution imposing non-zero beta.
        ax2.plot(stat_beta[i][1], xi, linestyle='-', color=[0.5*col_w[n_w-1-i], 0.0, col_w[n_w-1-i]], marker='None',
                 linewidth=2.75, alpha=1.0, label=r'$ \beta = '+str(np.round(beta_L_s[i], 3))+' $')

        # Stationary solution imposing non-zero beta.
        ax3.plot(stat_G[i][1], xi, linestyle='-', color=[col_w[n_w-1-i], 0.0, 0.0], marker='None',
                 linewidth=2.75, alpha=1.0, label=r'$ \gamma = '+str(np.round(G_k_s[i], 3))+' $')

        # Stationary solution imposing non-zero beta.
        ax4.plot(stat_strn[i][1][:], xi, linestyle='-', color=[0.0, 0.5*col_w[n_w-1-i], col_w[n_w-1-i]],
                 marker='None', linewidth=2.75, alpha=1.0,
                 label=r'$ \mathrm{Br} = '+str(np.round(strain_s[i], 2))+' $')

        # Stationary solution imposing non-zero beta.
        ax5.plot(stat_hor[i][1][:], xi, linestyle='-', color=[col_w[n_w-1-i], 0.0, col_w[n_w-1-i]],
                 marker='None', linewidth=2.75, alpha=1.0,
                 label=r'$ \Lambda = '+str(np.round(strain_H_s[i], 2))+' $')

    # Add labels and title to plot
    ax.set_xlabel(r'$ \vartheta(^{\circ} \mathrm{C})$', fontsize=22)
    ax2.set_xlabel(r'$ \vartheta(^{\circ} \mathrm{C}) $', fontsize=22)
    ax3.set_xlabel(r'$ \vartheta(^{\circ} \mathrm{C}) $', fontsize=22)
    ax4.set_xlabel(r'$ \vartheta(^{\circ} \mathrm{C}) $', fontsize=22)
    ax5.set_xlabel(r'$ \vartheta(^{\circ} \mathrm{C}) $', fontsize=22)

    ax.set_ylabel(r'$ \xi \ $', fontsize=22)

    ax.grid(axis='y', which='major', alpha=0.85)
    ax2.grid(axis='y', which='major', alpha=0.85)
    ax3.grid(axis='y', which='major', alpha=0.85)
    ax4.grid(axis='y', which='major', alpha=0.85)
    ax5.grid(axis='y', which='major', alpha=0.85)

    # Ticks.
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])
    ax2.tick_params(axis='y', which='major', length=0, colors='black')
    ax3.tick_params(axis='y', which='major', length=0, colors='black')
    ax4.tick_params(axis='y', which='major', length=0, colors='black')
    ax5.tick_params(axis='y', which='major', length=0, colors='black')

    """ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['$0.0$', '$0.2$', '$0.4$', '$0.6$',
                       '$0.8$', '$1.0$'], fontsize=18)

    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xticklabels(['$1.0$', '$1.5$', '$2.0$'], fontsize=18)

    ax2.set_xticks([1.0, 1.5, 2.0])
    ax2.set_xticklabels(['$1.0$', '$1.5$', '$2.0$'], fontsize=18)

    ax3.set_xticks([1.0, 1.5, 2.0])
    ax3.set_xticklabels(['$1.0$', '$1.5$', '$2.0$'], fontsize=18)

    ax4.set_xticks([1.0, 1.5, 2.0])
    ax4.set_xticklabels(['$1.0$', '$1.5$', '$2.0$'], fontsize=18)

    ax5.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax5.set_xticklabels(['$0.0$', '$1.0$', '$2.0$', '$3.0$'], fontsize=18)

    # Limits
    ax.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)
    ax5.set_ylim(0.0, 1.0)

    ax.set_xlim(1.0, 2.0)
    ax2.set_xlim(1.0, 2.0)
    ax3.set_xlim(1.0, 2.0)
    ax4.set_xlim(1.0, 2.0)
    ax5.set_xlim(0.0, 3.0)"""

    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['$0.0$', '$0.2$', '$0.4$', '$0.6$',
                       '$0.8$', '$1.0$'], fontsize=18)

    ax.set_xticks([-50, -25, 0])
    ax.set_xticklabels(['$-50$', '$-25$', '$0$'], fontsize=18)

    ax2.set_xticks([-50, -25, 0])
    ax2.set_xticklabels(['$-50$', '$-25$', '$0$'], fontsize=18)

    ax3.set_xticks([-50, -25, 0])
    ax3.set_xticklabels(['$-50$', '$-25$', '$0$'], fontsize=18)

    ax4.set_xticks([-50, -25, 0])
    ax4.set_xticklabels(['$-50$', '$-25$', '$0$'], fontsize=18)

    ax5.set_xticks([-50, -25, 0])
    ax5.set_xticklabels(['$-50$', '$-25$', '$0$'], fontsize=18)

    # Limits
    ax.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)
    ax5.set_ylim(0.0, 1.0)

    ax.set_xlim(-50, 0)
    ax2.set_xlim(-50, 0)
    ax3.set_xlim(-50, 0)
    ax4.set_xlim(-50, 0)
    ax5.set_xlim(-50, 0)

    ax.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
              fontsize=14, fancybox=True)

    ax2.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=14, fancybox=True)

    ax3.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=14, fancybox=True)

    ax4.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=14, fancybox=True)

    ax5.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=14, fancybox=True)

    # Title.
    ax.set_title(r'$ \mathrm{(a)} $', fontsize=22, loc='center', pad=10)
    ax2.set_title(r'$ \mathrm{(b)} $', fontsize=22, loc='center', pad=10)
    ax3.set_title(r'$ \mathrm{(c)} $', fontsize=22, loc='center', pad=10)
    ax4.set_title(r'$ \mathrm{(d)} $', fontsize=22, loc='center', pad=10)
    ax5.set_title(r'$ \mathrm{(e)} $', fontsize=22, loc='center', pad=10)

    # Save figure.
    if save_fig == True:
        plt.savefig(path_fig+'stat_comparison_w_0_bar_'+str(np.round(w_0_bar, 2)
                                                            )+'_beta_'+str(beta)+'.png', bbox_inches='tight')

    # Display plot
    plt.show()
    plt.close(fig)


#########################################################################################
#########################################################################################
# STATIONARY MULTIPLOT SIMPLE.

if stat_simple == True:

    # Different exponent for the vertical dependency of vertical advection
    # 1.0 is Robin and 2.0 is Raymond and we explore some values in between.
    m_s = np.array([1.0, 5.0/4.0, 6.0/4.0, 7.0/4.0, 2.0])
    m_lab = np.array([r'$1$', r'$5/4$', r'$6/4$', r'$7/4$', r'$2$'])
    n_m = len(m_s)

    # Obtain stationary solution with EISMINT parameters.
    y_adv = []
    sol_s = []

    adv = 'power-law'

    # Solutions.
    for i in range(n_m):
        y_adv.append(f_adv(xi, w_0, m_s[i]))
        sol_s.append(stat_sol(w_0_bar, G_k, beta_L,
                     strain, m_s[i], adv, xi, n))

    # Colours array.s
    col_w = np.linspace(0.0, 1.0, n_m)

    # Non-zero beta here. Surface temperature can evolve in time.
    fig = plt.figure(dpi=600, figsize=(8, 6))
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i in range(n_m):

        # We transform back to Kelvin.
        # Analytical.
        sol_down = T_air * sol_s[i][3][n_t-1, :]

        # Numerical
        #sol_down = T_air * sol_s[i][2][:]

        # Convert to homologous temperature to compare to eismint.
        sol_down = sol_down + beta_melt * L * xi[::-1]

        # Plot stationary solutions
        ax.plot(y_adv[i], xi, linestyle='-', color=[col_w[n_m-1-i], 0.25*col_w[n_m-1-i], 0.5], marker='None',
                linewidth=2.0, alpha=1.0, label=r'$ m = '+str(np.round(m_s[i], 2))+' $')

        ax2.plot(sol_down, xi, linestyle='-', color=[col_w[n_m-1-i], 0.25*col_w[n_m-1-i], 0.5], marker='None',
                 linewidth=2.0, alpha=1.0, label=r'$ m = '+str(np.round(m_s[i], 2))+' $')

    # Add labels and title to plot
    ax.set_xlabel(r'$ w \ (\mathrm{m/yr})$', fontsize=20)
    ax.set_ylabel(r'$ \xi \ $', fontsize=20)

    ax2.set_xlabel(r'$ T_{h} \ (^{\circ} \mathrm{C}) $', fontsize=20)

    ax.grid(axis='y', which='major', alpha=0.85)
    ax2.grid(axis='y', which='major', alpha=0.85)

    # Ticks.
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', which='major', length=0, colors='black')

    # Ticks.
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', which='major', length=0, colors='black')

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['$0.0$', '$0.2$', '$0.4$', '$0.6$',
                       '$0.8$', '$1.0$'], fontsize=15)

    ax.set_xticks([-0.3, -0.2, -0.1, 0.0])
    ax.set_xticklabels(['$-0.3$', '$-0.2$', '$-0.1$', '$0.0$'], fontsize=15)

    #ax2.set_xticks([-40, -30, -20, -10, 0])
    #ax2.set_xticklabels(['$-40$', '$-30$', '$-20$','$-10$', '$0$'], fontsize=15)

    # Limits
    ax.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.3, 0.0)

    ax2.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
               fontsize=15, fancybox=True)

    # Save figure.
    if save_fig == True:
        name_fig = 'eismint_stat_w_0_bar_' + \
            str(np.round(w_0_bar, 2))+'_beta_' + \
            str(beta)+'_Br_'+str(np.round(strain, 2))
        plt.savefig(path_fig+name_fig+'.png',
                    format="png", bbox_inches='tight')
        plt.savefig(path_fig+name_fig+'.pdf',
                    format="pdf", bbox_inches="tight")

    # Display plot
    plt.show()
    plt.close(fig)


#########################################################################################
#########################################################################################
# FULL SOLUTIONS WITH INITIAL CONDITIONS.
# THETA = U + V (STATIONARY PLUS TRANSITORY).

if full_profiles == True:

    # Create colour map.
    """col_a = np.ones(int(0.10*n_t))
    col_b = np.linspace(1.0, 0.0, int(0.90*n_t))
    col_1 = np.concatenate([col_a, col_b])
    col_1 = col_1**2.0

    col_a = np.zeros(int(0.2*n_t))
    col_b = np.linspace(0.0, 1.0, int(0.8*n_t))
    col_2 = np.concatenate([col_a, col_b])
    col_2 = col_2**0.5

    col_a = np.linspace(1.0, 0.5, int(0.1*n_t))
    col_b = np.linspace(0.5, 0.0, int(0.3*n_t))
    col_c = np.linspace(0.0, 1.0, int(0.6*n_t)) # 0.75
    col_3 = np.concatenate([col_a, col_b, col_c])"""

    # Create colour map.
    col_a = np.ones(int(0.20*n_t))
    col_b = np.linspace(1.0, 0.0, int(0.80*n_t))
    col_1 = np.concatenate([col_a, col_b])
    col_1 = col_1**2.0

    col_a = np.zeros(int(0.3*n_t))
    col_b = np.linspace(0.0, 1.0, int(0.7*n_t))
    col_2 = np.concatenate([col_a, col_b])
    col_2 = col_2**0.5

    col_a = np.linspace(1.0, 0.5, int(0.2*n_t))
    col_b = np.linspace(0.5, 0.0, int(0.2*n_t))
    col_c = np.linspace(0.0, 1.0, int(0.6*n_t))  # 0.75
    col_3 = np.concatenate([col_a, col_b, col_c])

    # Create a colormap from the three arrays
    colors = np.column_stack((col_1, col_3, col_2))
    cmap = ListedColormap(colors)

    # Dimensionless parameters of the solution.
    """w_0_bar_1 = 5.0  # 5.0
    w_0_bar_2 = 5.0
    G_k_1     = -3.0
    G_k_2     = -3.0  # 2.5
    beta_L_1  = 0.5  # 0.0
    beta_L_2  = 0.0
    strain_1  = 0.0
    strain_2  = 7.25"""

    # Dimensionless parameters of the solution for dimensional plots.
    w_0_bar_1 = 5.0  # 5.0
    w_0_bar_2 = 5.0
    G_k_1     = -0.35
    G_k_2     = -0.35  # 2.5
    beta_L_1  = 0.5  # 0.0
    beta_L_2  = 0.0
    strain_1  = 0.0
    strain_2  = 1.0

    # Initial temperature. Constant for simplicity.
    theta_0_1 = 0.98
    theta_0_2 = 1.05

    # Calculate full problem solutions.
    sol_1 = full_sol(plot_lmbd, order, w_0_bar_1, XI, G_k_1,
                     strain_1, beta_L_1, theta_0_1, n, n_t)
    sol_2 = full_sol(plot_lmbd, order, w_0_bar_2, XI, G_k_2,
                     strain_2, beta_L_2, theta_0_2, n, n_t)

    
    # Transform back to dimensional variables.
    T_air = 223.15
    sol_1 = T_air * np.array(sol_1) - 273.15
    sol_2 = 248.15 * np.array(sol_2) - 273.15

    theta_0_1 = T_air * theta_0_1 - 273.15
    theta_0_2 = 248.15 * theta_0_2 - 273.15

    # Rewrite soluions.
    theta_down_1 = sol_1[1]
    theta_down_2 = sol_2[1]
    u_down_1 = sol_1[3]
    u_down_2 = sol_2[3]

    # Synthetic initial condition.
    theta_0_1_xi = np.full(n, theta_0_1)
    theta_0_2_xi = np.full(n, theta_0_2)

    # PLOTS
    fig = plt.figure(dpi=600, figsize=(8, 6))
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Label counter.
    c_lab = 0
    c_ind = 0

    # DOWNWARDS SOLUTION.
    for i in range(4, n_t, 5):
        ax.plot(theta_down_1[i, :], XI[i, :], linestyle='-', color=[col_1[i], col_3[i], col_2[i]], marker='None',
                linewidth=2.0, alpha=1.0)

    for i in range(2, n_t, 4): # (2, n_t, 4)
        ax2.plot(theta_down_2[i, :], XI[i, :], linestyle='-', color=[col_1[i], col_3[i], col_2[i]], marker='None',
                 linewidth=2.0, alpha=1.0, label=r'$ \tau_{ '+str(c_ind)+' } $')

    # Initial condition.
    ax.plot(theta_0_1_xi, xi, linestyle=':', color='darkgreen', marker='None',
            linewidth=2.0, alpha=1.0, label=r'$ \theta_{0} $')

    ax2.plot(theta_0_2_xi, xi, linestyle=':', color='darkgreen', marker='None',
             linewidth=2.0, alpha=1.0, label=r'$ \theta_{0} $')

    # Stationary solutions.
    ax.plot(u_down_1[0, :], xi, linestyle='--', color='black', marker='None',
            linewidth=2.0, alpha=1.0, label=r'$ \vartheta(\xi) $')

    ax2.plot(u_down_2[0, :], xi, linestyle='--', color='black', marker='None',
             linewidth=2.0, alpha=1.0, label=r'$ \vartheta(\xi) $')

    # Add a colorbar based on the colormap
    cbar_ax = fig.add_axes([0.95, 0.125, 0.04, 0.755])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)

    #tau_norm = np.linspace(0.0, 1.0, n_t)**n_exp

    # Given an array of values, we find those indexes with the closest values of tau_norm
    # to the former array (values).
    #values = np.array([0.0, 0.4, 0.6, 0.8, 1.0])

    #n_exp = 2

    # We create a linearly-spaced vector for the ticks but the labels consider tau_max.
    values = np.linspace(0.0, 1.0, 5)
    ticks_plot  = values**n_exp  # indexes
    ticks_time  = tau_max * ticks_plot  # indexes

    # Transform dimensionless time into yr.
    tick_time   = ticks_time * (L**2 / (kappa * const))

    # Round all decimals after expressing into kyr.
    tick_time   = np.round(1.0e-3 * tick_time, 0)
    tick_labels = [r'$'+str(int(tick))+'$' for tick in tick_time]
    #tick_labels = [r'$0.0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1.0$']

    # Set the modified ticks and tick labels
    cb.set_ticks(ticks_plot)
    cb.set_ticklabels(tick_labels)

    cb.set_label(r'$ \mathrm{Time \ (kyr)}  $', rotation=90, labelpad=8, fontsize=25)

    # Font properties of the tick labels
    cb.ax.tick_params(labelsize=15)

    # Add labels and title to plot
    ax.set_xlabel(r'$\theta \ (^{\circ} \mathrm{C})$', fontsize=20)
    ax.set_ylabel(r'$ \xi \ $', fontsize=20)

    ax2.set_xlabel(r'$ \theta \ (^{\circ} \mathrm{C}) $', fontsize=20)

    ax.grid(axis='y', which='major', alpha=0.85)
    ax2.grid(axis='y', which='major', alpha=0.85)

    # Ticks.
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', which='major', length=0, colors='black')

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['$0$', '$0.2$', '$0.4$', '$0.6$',
                       '$0.8$', '$1.0$'], fontsize=15)

    ax.set_xticks([-60, -40, -20, 0])
    ax.set_xticklabels(['$-60$', '$-40$', '$-20$', '$0$'], fontsize=15)

    ax2.set_xticks([-60, -40, -20, 0])
    ax2.set_xticklabels(['$-60$', '$-40$', '$-20$', '$0$'], fontsize=15)

    # Title.
    ax.set_title(r'$ \mathrm{(a)} $', fontsize=18, loc='center', pad=10)
    ax2.set_title(r'$ \mathrm{(b)} $', fontsize=18, loc='center', pad=10)

    # Limits
    ax.set_xlim(-60, 0)
    ax2.set_xlim(-60, 0)

    ax.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)

    # Legend.
    ax.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
              fontsize=16, fancybox=True)

    # Save figure.
    if save_fig == True:
        name_fig = 'theta_w_0_bar_' + \
            str(np.round(w_0_bar, 2))+'_beta_' + \
            str(beta)+'_Br_'+str(np.round(strain, 2))
        plt.savefig(path_fig+name_fig+'.png',
                    format="png", bbox_inches='tight')
        plt.savefig(path_fig+name_fig+'.pdf',
                    format="pdf", bbox_inches="tight")

    # Display plot
    plt.show()
    plt.close(fig)


#########################################################################################
#########################################################################################
# ENERGY CONTENT BY INTEGRATING TEMPERATURE PROFILES.

if energy == True:

    compute = True
    plot = True

    if compute == True:

        # Array values for comparison.
        w_0_bar_s = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        G_k_s = np.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        beta_L_s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        strain_s = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])

        n_w = len(w_0_bar_s)

        # Prepare variables for integration.
        # Number of time points where energy integration is performed.
        n_Q = 100
        Q_up = np.empty([n_w, n_Q])
        Q_down = np.empty([n_w, n_Q])
        tau_Q = np.empty(n_Q)

        # Corresponding indexes for the given number of integration points.
        n_energy = np.linspace(0, n_t-1, n_Q, dtype=int)

        # Colours array.
        col_w = np.linspace(0.0, 1.0, n_w)

        # Initial profile.
        theta_0 = 1.75

        sol_adv = []
        sol_G = []
        sol_beta = []
        sol_strn = []

        for i in range(len(strain_s)):
            print('Calculating strain_s solution # = ', i)
            sol_strn.append(full_sol(plot_lmbd, order, w_0_bar_s[3], XI,  G_k_s[2],
                                     strain_s[i], 0.0, theta_0, n, n_t))

        for i in range(len(G_k_s)):
            print('Calculating G_s solution # = ', i)
            sol_G.append(full_sol(plot_lmbd, order, w_0_bar_s[4], XI, G_k_s[i],
                                  strain_s[0], 0.0, theta_0, n, n_t))

        for i in range(len(beta_L_s)):
            print('Calculating beta_L_s solution # = ', i)
            sol_beta.append(full_sol(plot_lmbd, order, w_0_bar_s[4], XI, -0.5,
                                     strain_s[0], beta_L_s[i], theta_0, n, n_t))

        for i in range(n_w):
            print('Calculating w_0_bar_s solution # = ', i)
            sol_adv.append(full_sol(plot_lmbd, order, w_0_bar_s[i], XI, -0.5,
                                    strain_s[0], 0.0, theta_0, n, n_t))

        # Integration of temperature profiles.
        Q_up_adv, Q_down_adv, tau_Q = Q_int(
            sol_adv, n_energy, tau, theta_0, n_w)
        Q_up_G, Q_down_G, tau_Q = Q_int(sol_G, n_energy, tau, theta_0, n_w)
        Q_up_beta, Q_down_beta, tau_Q = Q_int(
            sol_beta, n_energy, tau, theta_0, n_w)
        Q_up_strn, Q_down_strn, tau_Q = Q_int(
            sol_strn, n_energy, tau, theta_0, n_w)

        # Plot derivaive instead:
        Q_down_adv = np.gradient(Q_down_adv, axis=1)
        Q_down_G = np.gradient(Q_down_G, axis=1)
        Q_down_beta = np.gradient(Q_down_beta, axis=1)
        Q_down_strn = np.gradient(Q_down_strn, axis=1)

    if plot == True:

        # Figure.
        fig = plt.figure(dpi=800, figsize=(14, 15))
        plt.rcParams['text.usetex'] = True
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # Plots.
        for i in range(n_w):

            # Time series.
            ax.plot(tau_Q, Q_down_adv[i, :], linestyle='-', color=[col_w[n_w-1-i], 0.5*col_w[n_w-1-i], 0.0],
                    marker='None', linewidth=4.0, alpha=1.0,
                    label=r'$ \mathrm{Pe} = '+str(np.round(abs(w_0_bar_s[i]), 2))+' $')

            # Time series.
            ax2.plot(tau_Q, Q_down_strn[i, :], linestyle='-', color=[0.5*col_w[n_w-1-i], 0.0, col_w[n_w-1-i]], marker='None',
                     linewidth=4.0, alpha=1.0, label=r'$ \Omega = '+str(np.round(strain_s[i], 2))+' $')

            # Time series.
            ax3.plot(tau_Q, Q_down_beta[i, :], linestyle='-', color=[0.0, 0.5*col_w[n_w-1-i], col_w[n_w-1-i]], marker='None',
                     linewidth=4.0, alpha=1.0, label=r'$ \beta = '+str(np.round(beta_L_s[i], 2))+' $')

            # Time series.
            ax4.plot(tau_Q, Q_down_G[i, :], linestyle='-', color=[col_w[n_w-1-i], 0.0, 0.5*col_w[n_w-1-i]], marker='None',
                     linewidth=4.0, alpha=1.0, label=r'$ \gamma = '+str(np.round(G_k_s[i], 2))+' $')

        # Zero lines.
        ax.plot(np.zeros(n), linestyle=':', color='black', marker='None',
                linewidth=4.0, alpha=0.5, zorder=2)

        ax2.plot(np.zeros(n), linestyle=':', color='black', marker='None',
                 linewidth=4.0, alpha=0.5, zorder=2)

        ax3.plot(np.zeros(n), linestyle=':', color='black', marker='None',
                 linewidth=4.0, alpha=0.5, zorder=2)

        ax4.plot(np.zeros(n), linestyle=':', color='black', marker='None',
                 linewidth=4.0, alpha=0.5, zorder=2)

        # Add labels and title to plot
        ax3.set_xlabel(r'$\tau$', fontsize=30)
        ax4.set_xlabel(r'$\tau$', fontsize=30)
        ax.set_ylabel(r'$ Q $', fontsize=30, labelpad=15)
        ax3.set_ylabel(r'$ Q $', fontsize=30, labelpad=15)

        ax.grid(axis='x', which='major', alpha=0.85)
        ax2.grid(axis='x', which='major', alpha=0.85)
        ax3.grid(axis='x', which='major', alpha=0.85)
        ax4.grid(axis='x', which='major', alpha=0.85)

        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])

        ax.tick_params(axis='x', which='major', length=4,
                       colors='black', labelsize=5)
        ax2.tick_params(axis='x', which='major', length=4,
                        colors='black', labelsize=5)
        ax3.tick_params(axis='x', which='major', length=4,
                        colors='black', labelsize=5)
        ax4.tick_params(axis='x', which='major', length=4,
                        colors='black', labelsize=20)

        ax.tick_params(axis='y', which='major', length=4,
                       colors='black', labelsize=20)
        ax2.tick_params(axis='y', which='major', length=4,
                        colors='black', labelsize=20)
        ax3.tick_params(axis='y', which='major', length=4,
                        colors='black', labelsize=20)
        ax4.tick_params(axis='y', which='major', length=4,
                        colors='black', labelsize=20)

        # Title.
        ax.set_title(r'$ \mathrm{(a)} $', fontsize=30, loc='center', pad=12)
        ax2.set_title(r'$ \mathrm{(b)} $', fontsize=30, loc='center', pad=12)
        ax3.set_title(r'$ \mathrm{(c)} $', fontsize=30, loc='center', pad=12)
        ax4.set_title(r'$ \mathrm{(d)} $', fontsize=30, loc='center', pad=12)

        # Ticks.
        """ax2.set_yticks([1.0, 1.5, 2.0, 2.5])
        ax4.set_yticks([1.0, 1.5, 2.0, 2.5])
        ax2.set_yticklabels([])
        ax4.set_yticklabels([])"""

        ax3.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax3.set_xticklabels(['$0.0$', '$0.25$', '$0.50$',
                            '$0.75$', '$1.0$'], fontsize=25)

        ax4.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax4.set_xticklabels(['$0.0$', '$0.25$', '$0.50$',
                            '$0.75$', '$1.0$'], fontsize=25)

        """ax.set_yticks([1.0, 1.5, 2.0, 2.5])
        ax.set_yticklabels(['$1.0$', '$1.5$', '$2.0$', '$2.5$'], fontsize=25)

        ax3.set_yticks([1.0, 1.5, 2.0, 2.5])
        ax3.set_yticklabels(['$1.0$', '$1.5$', '$2.0$', '$2.5$'], fontsize=25)"""

        # Limits
        y_min = 1.0
        y_max = 2.5
        """ax.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        ax3.set_ylim(y_min, y_max)
        ax4.set_ylim(y_min, y_max)"""

        x_max = 0.75
        ax.set_xlim(0.0, x_max)
        ax2.set_xlim(0.0, x_max)
        ax3.set_xlim(0.0, x_max)
        ax4.set_xlim(0.0, x_max)

        plt.tight_layout()

        # Legend.
        ax.legend(loc='best', ncol=2, frameon=True, framealpha=1.0,
                  fontsize=22, fancybox=True)
        ax2.legend(loc='best', ncol=2, frameon=True, framealpha=1.0,
                   fontsize=22, fancybox=True)
        ax3.legend(loc='best', ncol=2, frameon=True, framealpha=1.0,
                   fontsize=22, fancybox=True)
        ax4.legend(loc='best', ncol=2, frameon=True, framealpha=1.0,
                   fontsize=22, fancybox=True)

        # Save figure.
        if save_fig == True:
            name_fig = 'Q_time_series'
            plt.savefig(path_fig+name_fig+'.png',
                        format="png", bbox_inches='tight')
            plt.savefig(path_fig+name_fig+'.pdf',
                        format="pdf", bbox_inches="tight")

        # Display plot
        plt.show()
        plt.close(fig)


#########################################################################################
#########################################################################################
# EIGENVALUE PLOTS
if eigenvalues == True:

    calculate = False
    plot = True

    if calculate == True:
        # Order.
        order = 4
        x_plot = np.linspace(0, order-1, order)

        # Hypergeometric function variable.
        """w_0_bar_s = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        beta_L_s  = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])"""

        n_plot = 50  # 50
        w_0_bar_s = np.linspace(1.0, 5.0, n_plot)
        beta_L_s = np.linspace(0.0, 1.0, n_plot)

        l_w = len(w_0_bar_s)
        l_beta = len(beta_L_s)

        b_1 = 0.5
        z = 0.5 * w_0_bar * xi**2

        # Eigenvalues to plot.
        lambd_0 = 1.0e-6  # 0.1
        lmbd = np.empty([l_w, l_beta, order])

        for i in range(l_w):
            for j in range(l_beta):
                print('w_0_bar = ', w_0_bar_s[i])
                print('beta_L  = ', beta_L_s[j])
                lmbd[i, j, :] = zeros_eigen(
                    order, w_0_bar_s[i], beta_L_s[j], lambd_0, z, b_1, 'down')[0]

    if plot == True:

        # Colours.
        col_w = np.linspace(0.0, 1.0, l_w)
        col_beta = np.linspace(0.0, 1.0, l_beta)

        # Labels for legend.
        beta_plot = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        w_0_plot = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Convert dimensionless axis into time.
        y_ticks = np.array([1.0e-2, 1.0e-1, 1.0])
        time = (L**2 / (kappa * const)) * y_ticks

        # FIGURE.
        fig = plt.figure(dpi=400, figsize=(12, 5))  # (8,5)

        plt.rcParams['text.usetex'] = True

        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax3 = ax.twinx()

        # Adjust the separation between the two subplots
        fig.subplots_adjust(wspace=0.6)

        c = 0

        # Eigenvalues plot.
        for i in range(0, l_w, int(n_plot/len(w_0_plot))):

            """inverse = 1.0 / lmbd[i,0,:]
            ax.plot(x_plot, inverse, linestyle='-', color=[col_w[l_w-1-i],0.5*col_w[l_w-1-i],0.0], marker='None', \
                                linewidth=2.0, alpha=1.0, label=r'$ \mathrm{Pe} = '+str(w_0_plot[c])+' $')"""

            ax.plot(np.nan, linestyle='-', color=[col_w[l_w-1-i], 0.5*col_w[l_w-1-i], 0.0], marker='o', markersize=5,
                    linewidth=2.0, alpha=1.0, label=r'$ \mathrm{Pe} = '+str(w_0_plot[c])+' $')

            """inverse = 1.0 / lmbd[3,i,:]
            ax.plot(x_plot, inverse, linestyle='-', color=[col_beta[i], 0.0, col_beta[i]], marker='None', \
                                linewidth=2.0, alpha=1.0, label=r'$ \beta = '+str(beta_plot[c])+' $')"""

            c += 1

        for i in range(int(0.3*n_plot/len(w_0_plot)), l_w, int(0.4*n_plot/len(w_0_plot))):

            inverse = 1.0 / lmbd[i, 0, :]
            ax.plot(x_plot, inverse, linestyle='-', color=[col_w[l_w-1-i], 0.5*col_w[l_w-1-i], 0.0], marker='o',
                    markersize=5.0, linewidth=1.5, alpha=1.0)

        ax3.plot(np.nan, linestyle='-', color='white', marker='o',
                 markersize=5.0, linewidth=1.5, alpha=0.0)

        ax.set_yscale('log')
        ax3.set_yscale('log')

        # HEAT MAP.
        inverse = 1.0e-3 * (1.0 / lmbd[:, :, 0]) * (L**2 / (kappa * const))

        inverse_min = 0.0
        #inverse_max = np.nanmax(inverse)
        inverse_max = 40.0

        im = ax2.imshow(np.rot90(inverse), cmap='gist_rainbow',
                        vmin=inverse_min, vmax=inverse_max, aspect='auto', interpolation='bicubic')  # 'RdYlBu', 'gist_rainbow', nipy_spectral

        cbar_ax = fig.add_axes([0.93, 0.2, 0.025, 0.6])  # 1.025
        cb = fig.colorbar(im, cax=cbar_ax, extend='neither')

        cb.set_label(
            r'$ 1 / \lambda \ (\mathrm{kyr}) $', rotation=90, labelpad=15, fontsize=20)
        cb.ax.tick_params(labelsize=15)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['$1$', '$2$', '$3$', '$4$'], fontsize=15)

        ax.set_yticks([1.0e-2, 1.0e-1, 1.0])
        ax.set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=15)

        ax3.set_yticks([1.0e-2, 1.0e-1, 1.0])
        ax3.set_yticklabels(['$0.25$', '$2.5$', '$25$'], fontsize=15)

        heat_ticks_x = np.linspace(0.0, n_plot-1, 5)
        heat_ticks_y = np.linspace(0.0, n_plot-1, 6)

        ax2.set_xticks(heat_ticks_x)
        ax2.set_xticklabels(['$1$', '$2$', '$3$', '$4$', '$5$'], fontsize=16)

        ax2.set_yticks(heat_ticks_y[::-1])
        ax2.set_yticklabels(
            ['$0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$'], fontsize=16)

        # Add labels and title to plot
        ax.set_xlabel(r'$ \mathrm{Order} $', fontsize=20)
        ax.set_ylabel(r'$ 1 / \lambda $', fontsize=20)

        ax3.set_ylabel(r'$ \mathrm{Decay \ time} \ (\mathrm{kyr}) $', fontsize=20)

        ax.grid(axis='x', which='major', alpha=0.85)
        ax.grid(axis='y', which='major', alpha=0.85)
        ax.grid(axis='y', which='minor', alpha=0.25)

        ax.set_xlim(-0.075, 3.075)
        #ax.set_ylim(0.0, 0.8)
        ax.set_ylim(0.5e-2, 1.0)
        ax3.set_ylim(0.5e-2, 1.0)

        ax2.set_xlabel(r'$ \mathrm{Pe} $', fontsize=20)
        ax2.set_ylabel(r'$ \beta $', fontsize=22)

        ax.legend(loc='best', ncol=1, frameon=True, framealpha=1.0,
                  fontsize=13, fancybox=True)

        # Title.
        ax.set_title(r'$ \mathrm{(a)} $', fontsize=20, loc='center', pad=10)
        ax2.set_title(r'$ \mathrm{(b)} $', fontsize=20, loc='center', pad=10)

        # Save figure.
        if save_fig == True:
            name_fig = 'decay_time'
            plt.savefig(path_fig+name_fig+'.png',
                        format="png", bbox_inches='tight')
            plt.savefig(path_fig+name_fig+'.pdf',
                        format="pdf", bbox_inches="tight")

        # Display plot
        plt.show()
        plt.close(fig)


#########################################################################################
#########################################################################################
# HEAT MAPS PLOT.
if heat_map == True:

    # Initial temperature.
    theta_0 = 1.02

    # Colours array.

    # Solutions parameters.
    w_0_bar_1 = 7.0
    w_0_bar_2 = 2.0
    w_0_bar_3 = 1.0e-3 # 1.0e-3
    w_0_bar_4 = 1.0e-3  # 5.0

    # Old parameters for dimensionless plots.
    """G_k_1 = - 6.0
    G_k_2 = - 0.5
    G_k_3 = - 6.0
    G_k_4 = - 1.5
    
    # Positive values mean horizontal advection of cold ice.
    # Negative values imply heat dissipation due to strain deformation.
    strain_1 = 0.0
    strain_2 = - 6.0
    strain_3 = 9.0
    strain_4 = 0.0
    """

    # New values for dimensional plots.
    G_k_1 = - 0.5
    G_k_2 = - 0.025
    G_k_3 = - 0.7 # 0.6
    G_k_4 = - 0.15

    # Positive values mean horizontal advection of cold ice.
    # Negative values imply heat dissipation due to strain deformation.
    strain_1 = 0.0
    strain_2 = - 1.2
    strain_3 = 1.0
    strain_4 = 0.0

    beta_L = 0.0

    # Full solutions.
    sol_1 = full_sol(plot_lmbd, order, w_0_bar_1, XI, G_k_1,
                     strain_1, beta_L, theta_0, n, n_t)[1]
    sol_2 = full_sol(plot_lmbd, order, w_0_bar_2, XI, G_k_2,
                     strain_2, beta_L, theta_0, n, n_t)[1]
    sol_3 = full_sol(plot_lmbd, order, w_0_bar_3, XI, G_k_3,
                     strain_3, 0.11, theta_0, n, n_t)[1]
    sol_4 = full_sol(plot_lmbd, order, w_0_bar_4, XI, G_k_4,
                     strain_4, 0.0, theta_0, n, n_t)[1]

    # Ensure initial condition is satisfied.
    """sol_1[0,:] = theta_0
    sol_2[0,:] = theta_0
    sol_3[0,:] = theta_0
    sol_4[0,:] = theta_0"""

    # Transform back into dimensional units.
    T_air = 223.15
    sol_1 = T_air * np.array(sol_1) - 273.15
    sol_2 = T_air * np.array(sol_2) - 273.15
    sol_3 = T_air * np.array(sol_3) - 273.15
    sol_4 = T_air * np.array(sol_4) - 273.15

    # Theta limits.
    theta_min = -50.0  # 0.5
    theta_max = 0.0  # 3.5

    # FIGURE.
    fig = plt.figure(dpi=400, figsize=(14, 15))
    plt.rcParams['text.usetex'] = True

    ax = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    ax_2 = ax.twinx()
    ax_4 = ax2.twinx()
    ax_6 = ax3.twinx()
    ax_8 = ax4.twinx()

    # Define the two colormaps
    cmap1 = plt.cm.Blues  # Blues
    cmap2 = plt.cm.YlOrRd
    cmap1 = cmap1.reversed()

    # Define the value ranges for each section of the colormap
    vmin1 = theta_min
    vmax1 = 0.5 * (theta_max + theta_min)
    vmin2 = 0.5 * (theta_max + theta_min)
    vmax2 = theta_max

    # Generate a list of colors for each section of the colormap
    colors = []
    num_colors = 256  # Number of colors in the resulting colormap

    for i in range(num_colors):
        if i < num_colors * (vmax1 - vmin1) / (vmax1 - vmin1 + vmax2 - vmin2):
            color = cmap1(i / (num_colors * (vmax1 - vmin1) /
                          (vmax1 - vmin1 + vmax2 - vmin2)))
        else:
            color = cmap2((i - num_colors * (vmax1 - vmin1) / (vmax1 - vmin1 + vmax2 - vmin2)) /
                          (num_colors * (vmax2 - vmin2) / (vmax1 - vmin1 + vmax2 - vmin2)))
        colors.append(color)

    # Create the combined colormap
    combined_cmap = ListedColormap(colors)

    # Get the 'gist_rainbow' colormap
    cmap = plt.get_cmap('PiYG')  # PiYG
    cmap_down = plt.get_cmap('PRGn')  # PRGn

    # Reverse the list of colors in the colormap
    reversed_cmap = cmap_down.reversed()
    cmap = cmap.reversed()

    # Flip theta matrix so that the plot is not upside down. 'plasma', 'jet', 'gist_rainbow', 'rainbow'
    ax.imshow(np.rot90(sol_1), cmap=combined_cmap,
              vmin=theta_min, vmax=theta_max, aspect='auto', interpolation='bilinear')

    im = ax2.imshow(np.rot90(sol_2), cmap=combined_cmap,
                    vmin=theta_min, vmax=theta_max, aspect='auto', interpolation='bilinear')

    im2 = ax3.imshow(np.rot90(sol_3), cmap=combined_cmap,
                     vmin=theta_min, vmax=theta_max, aspect='auto', interpolation='bilinear')

    ax4.imshow(np.rot90(sol_4), cmap=combined_cmap,
               vmin=theta_min, vmax=theta_max, aspect='auto', interpolation='bilinear')

    ax.set_ylabel(r'$ \xi $', fontsize=40, labelpad=10)
    ax2.set_ylabel(r'$ \xi $', fontsize=40, labelpad=10)
    ax3.set_ylabel(r'$ \xi $', fontsize=40, labelpad=10)
    ax4.set_ylabel(r'$ \xi $', fontsize=40, labelpad=10)

    ax4.set_xlabel(r'$ \mathrm{Time \ (kyr)}  $', fontsize=40)

    #divider = make_axes_locatable(ax2)
    cbar_ax = fig.add_axes([1.02, 0.15, 0.04, 0.75])  # 1.025
    cb = fig.colorbar(im, cax=cbar_ax, extend='neither')

    cb.set_label(r'$ \theta \ (^{\circ} \mathrm{C}) $', rotation=90, labelpad=15, fontsize=45)
    cb.ax.tick_params(labelsize=30, length=7)

    # Title.
    ax.set_title(r'$ \mathrm{(a)} \ \mathrm{Pe}=7, \ \gamma=-0.5, \ \mathrm{Br}=0,\ \Lambda=0 $',
                 fontsize=30, loc='center', pad=14)
    ax2.set_title(r'$ \mathrm{(b)} \ \mathrm{Pe}=2, \ \gamma=-0.025, \ \mathrm{Br}=-1.2,\ \Lambda=0 $',
                  fontsize=30, loc='center', pad=14)
    ax3.set_title(r'$ \mathrm{(c)} \ \mathrm{Pe}=0, \ \gamma=-0.7, \ \mathrm{Br}=0,\ \Lambda=1.0 $',
                  fontsize=30, loc='center', pad=14)
    ax4.set_title(r'$ \mathrm{(d)} \ \mathrm{Pe}=0, \ \gamma=-0.15, \ \mathrm{Br}=0,\ \Lambda=0 $',
                  fontsize=30   , loc='center', pad=14)

    # Number of ticks.
    n_ticks_x = 5
    n_ticks_y = 3

    # Ticks.
    values = np.linspace(0.0, 1.0, n_ticks_x)
    values_uneven = values**(1.0/n_exp)
    x_ticks = n_t * values_uneven
    
    # Adjust to avoid blank spaces at the borders.
    x_ticks[0] = x_ticks[0] - 0.5
    x_ticks[n_ticks_x-1] = x_ticks[n_ticks_x-1] - 0.5

    # Unevenly-spaced tau vector. values**n_exp
    ticks_time = tau_max * values**n_exp

    # Transform dimensionless time into yr.
    tick_time = ticks_time * (L**2 / (kappa * const))

    # Round all decimals after expressing into kyr.
    tick_time   = np.round(1.0e-3 * tick_time, 0)
    tick_labels = [r'$'+str(int(tick))+'$' for tick in tick_time]

    y_ticks  = np.linspace(0.0, n-0.5, n_ticks_y)
    y_labels = [r'$0.0$', r'$0.5$', r'$1.0$']

    ax.tick_params(axis='y', which='major', length=5, colors='black')
    ax2.tick_params(axis='y', which='major', length=5, colors='black')
    ax3.tick_params(axis='y', which='major', length=5, colors='black')
    ax4.tick_params(axis='y', which='major', length=5, colors='black')

    ax.tick_params(axis='x', which='major', length=7, colors='black')
    ax2.tick_params(axis='x', which='major', length=7, colors='black')
    ax3.tick_params(axis='x', which='major', length=7, colors='black')
    ax4.tick_params(axis='x', which='major', length=7, colors='black', labelsize=22)

    ax.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    ax_2.set_yticklabels([])
    ax_4.set_yticklabels([])
    ax_6.set_yticklabels([])
    ax_8.set_yticklabels([])

    ax_2.tick_params(axis='y', which='major', length=0, colors='black')
    ax_4.tick_params(axis='y', which='major', length=0, colors='black')
    ax_6.tick_params(axis='y', which='major', length=0, colors='black')
    ax_8.tick_params(axis='y', which='major', length=0, colors='black')

    # ax8.set_xticklabels([])
    ax.set_xticks(x_ticks)
    ax2.set_xticks(x_ticks)
    ax3.set_xticks(x_ticks)
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(tick_labels, fontsize=35)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(list(y_labels[::-1]), fontsize=30)

    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(list(y_labels[::-1]), fontsize=30)

    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(list(y_labels[::-1]), fontsize=30)

    ax4.set_yticks(y_ticks)
    ax4.set_yticklabels(list(y_labels[::-1]), fontsize=30)

    plt.tight_layout()

    if save_fig == True:
        name_fig = 'heat_maps'
        plt.savefig(path_fig+name_fig+'.png',
                    format="png", bbox_inches='tight')
        plt.savefig(path_fig+name_fig+'.pdf',
                    format="pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)

#########################################################################################
#########################################################################################
# NUMERICAL SOLUTION. f_num(theta_0, w_bar, strain, G_k, tau_f, dtau, n_s[i], n_out)


def f_num(theta, w_bar, strain, beta_L, G_k, tau_f, dtau, n,
          n_out, theta_zz_meth, bc, theta_z_w, uneven, dxi):

    # Handy definitions.
    dz = 1.0 / n
    dz_inv = (1.0 / dz)
    dz_2_inv = dz_inv**2
    dxi_inv = 1.0 / dxi

    # Prepare variables.
    theta_now = np.empty(n, dtype=np.float64)
    theta_all = np.empty([n, n_out], dtype=np.float64)

    # Without the loop.
    """
    theta_p1 = np.copy(theta)
    theta_p2 = np.copy(theta)
    theta_m1 = np.copy(theta)
    theta_m2 = np.copy(theta)
    """

    # Output.
    out = np.linspace(0.0, 1.0, n_out)
    out = out**3
    out = out * tau_f
    tau_plot = np.empty(n_out)

    c = 0
    tau = 0.0

    # Numerical solution with different discretization schemes.
    while (tau < tau_f):

        if uneven == True:
            for i in range(2, n-2, 1):

                # Temperature second order vertical derivatives.
                # Three-point second derivative.
                if theta_zz_meth == '3p':
                    h_1 = dxi[i-1]
                    h_2 = dxi[i]

                    theta_zz = 2.0 * (h_1 * theta[i+1] - (h_2 + h_1) * theta[i]
                                      + h_2 * theta[i-1]) / (h_2 * h_1 * (h_2 + h_1))

                elif theta_zz_meth == '4p':

                    # Eq. 30
                    h_1 = dxi[i-2]
                    h_2 = dxi[i-1]
                    h_3 = dxi[i]

                    H_1 = h_1 + h_2 + h_3

                    # Eq. 18c.
                    a_1 = 2.0 * (h_3 - h_2) / (h_1 * (h_1 + h_2) * H_1)

                    a_2 = 2.0 * (H_1 - 2.0 * h_3) / (h_1 * h_2 * (h_1 + h_3))

                    a_3 = 2.0 * (h_3 - 2.0 * h_2 - h_1) / \
                        ((h_1 + h_2) * h_2 * h_3)

                    a_4 = 2.0 * (h_1 + 2.0 * h_2) / (H_1 + (h_2 + h_3) * h_3)

                    theta_zz = a_1 * theta[i-2] + a_2 * theta[i-1] + a_3 * theta[i] \
                        + a_4 * theta[i+1]

                # Five-point second derivative.
                elif theta_zz_meth == '5p':

                    #print('theta_zz_3p = ', theta_zz)

                    # Eq. 30
                    h_1 = dxi[i-2]
                    h_2 = dxi[i-1]
                    h_3 = dxi[i]
                    h_4 = dxi[i+1]

                    """h_1 = dxi[i+1]
                    h_2 = dxi[i]
                    h_3 = dxi[i-1]
                    h_4 = dxi[i-2]"""

                    H_2 = h_1 + h_2 + h_3 + h_4

                    a_1 = (- 2.0 * h_2 * (2.0 * h_3 + h_4) + 2.0 * h_3 * (h_3 + h_4)) \
                        / (h_1 * (h_1 + h_2) * (h_1 + h_2 + h_3) * H_2)

                    a_2 = (2.0 * (h_1 + h_2) * (2.0 * h_3 + h_4) - 2.0 * h_3 * (h_3 + h_4)) \
                        / (h_1 * h_2 * (h_1 + h_3) * (h_2 + h_3 + h_4))

                    a_3 = (2.0 * h_2 * (h_1 + h_2) - 2.0 * (h_1 + 2.0 * h_2) * (2.0 * h_3 + h_4)
                           + 2.0 * h_3 * (h_3 + h_4)) / ((h_1 + h_2) * h_2 * h_3 * (h_3 + h_4))

                    a_4 = (2.0 * (h_1 + 2.0 * h_2) * (h_3 + h_4) - 2.0 * h_2 * (h_1 + h_2)) \
                        / ((h_1 + h_2 + h_3) * (h_2 + h_3) * h_3 * h_4)

                    a_5 = (2.0 * (h_1 + h_2) * h_2 - 2.0 * (h_1 + 2.0 * h_2) * h_3) \
                        / (H_2 * (h_2 + h_3 + h_4) * (h_3 + h_4) * h_4)

                    theta_zz = a_1 * theta[i-2] + a_2 * theta[i-1] + a_3 * theta[i] \
                        + a_4 * theta[i+1] + a_5 * theta[i+2]

                    #print('theta_zz_5p = ', theta_zz)

                # Two-point first derivative.
                if theta_z_w == 'backward_2p':
                    theta_z = dxi_inv[i] * (theta[i] - theta[i+1])

                # Three-point first derivative.
                elif theta_z_w == 'backward_3p':

                    a_1 = (2.0 * dxi[i-1] + dxi[i]) / \
                        (dxi[i-1] * (dxi[i-1] + dxi[i]))
                    a_2 = - (dxi[i-1] + dxi[i]) / (dxi[i-1] * dxi[i])
                    a_3 = dxi[i-1] / (dxi[i] * (dxi[i-1] + dxi[i]))

                    theta_z = a_1 * theta[i] + a_2 * \
                        theta[i+1] + a_3 * theta[i+2]

                # Two-point symmetric first derivative.
                elif theta_z_w == 'symmetric_2p':

                    theta_z = (theta[i-1] - theta[i+1]) / (dxi[i] + dxi[i-1])

                # Integrate temperature in time.
                theta_now[i] = theta[i] + dtau * \
                    (theta_zz + theta_z * w_bar[i] - strain)

            # Temperature at 1 for three-point schemes.
            theta_zz = 2.0 * (dxi[0] * theta[2] - (dxi[1] + dxi[0]) * theta[1]
                              + dxi[1] * theta[0]) / (dxi[1] * dxi[0] * (dxi[1] + dxi[0]))

            theta_z = dxi_inv[0] * (theta[1] - theta[2])

            theta_now[1] = theta[1] + dtau * \
                (theta_zz + theta_z * w_bar[1] - strain)

            # Temperature at n-2 for three-point schemes.
            theta_zz = 2.0 * (dxi[n-3] * theta[n-1] - (dxi[n-2] + dxi[n-3]) * theta[n-2]
                              + dxi[n-2] * theta[n-3]) / (dxi[n-2] * dxi[n-3] * (dxi[n-2] + dxi[n-3]))

            theta_z = dxi_inv[n-2] * (theta[n-2] - theta[n-1])

            theta_now[n-2] = theta[n-2] + dtau * \
                (theta_zz + theta_z * w_bar[n-2] - strain)

            # Boundary conditions.
            # Base.
            if bc == '2p':
                theta_now[0] = theta_now[1] + dxi[0] * (- G_k)

            elif bc == '3p':
                h_1 = dxi[0]
                h_2 = dxi[1]
                H_1 = h_1 + h_2

                a_1 = (2.0 * h_1 + h_2) / (h_1 * H_1)
                a_2 = - H_1 / (h_1 * h_2)
                a_3 = h_1 / (h_2 * H_1)

                theta_now[0] = (- G_k - a_2 * theta_now[1] -
                                a_3 * theta_now[2]) / a_1

            elif bc == '4p':
                # Eq. 15a.
                h_1 = dxi[0]
                h_2 = dxi[1]
                h_3 = dxi[2]

                H_1 = h_1 + h_2 + h_3

                a_1 = ((2.0 * h_1 + h_2) * H_1 + h_1 * (h_1 + h_2)) \
                    / (h_1 * (h_1 + h_2) * H_1)

                a_2 = - (h_1 + h_2) * H_1 / (h_1 * h_2 * (h_1 + h_3))

                a_3 = h_1 * H_1 / ((h_1 + h_2) * h_2 * h_3)

                a_4 = - h_1 * (h_1 * h_2) / (H_1 * (h_2 + h_3) * h_3)

                theta_now[0] = (- G_k - a_2 * theta_now[1] -
                                a_3 * theta_now[2] - a_4 * theta_now[3]) / a_1

            # Surface.
            theta_now[n-1] = (beta_L * theta_now[n-2] + 1.0) / \
                                (1.0 + (beta_L / dxi[n-2]))

        else:

            # Five-point factor.
            fact = 1.0 / 12.0

            # Avoid loop. Only for three-point vertical derivative.
            theta_p1[0:(n-2)] = theta[1:(n-1)]
            theta_p2[0:(n-3)] = theta[2:(n-1)]
            theta_m1[1:(n-1)] = theta[0:(n-2)]
            theta_m2[2:(n-1)] = theta[0:(n-3)]

            # VERTICAL ADVECTION DISCRETISATION.
            # Vertical velocities are negative (ice moving from surface to base).
            if theta_z_w == 'forward_2p':
                theta_z = theta_m1 - theta

            elif theta_z_w == 'forward_3p':
                theta_z = 0.5 * (- 3.0 * theta + 4.0 * theta_m1 - theta_m2)

            elif theta_z_w == 'backward_2p':
                theta_z = theta - theta_p1

            elif theta_z_w == 'backward_3p':
                theta_z = 0.5 * (3.0 * theta - 4.0 * theta_p1 + theta_p2)

            elif theta_z_w == 'backward_4p':
                theta_z = 0.5 * (2.0 * theta_m1 + 3.0 *
                                 theta - 6.0 * theta_p1 + theta_p2)

            elif theta_z_w == 'symmetric_2p':
                theta_z = 0.5 * (theta_m1 - theta_p1)

            elif theta_z_w == 'symmetric_4p':
                theta_z = (- theta_m2 + 8.0 * theta_m1 -
                           8.0 * theta_p1 + theta_p2) / 12.0

            # DIFFUSION DISCRETISATION.
            # Three-point vertical derivative.
            if theta_zz == '3p':
                theta_now = theta + dtau * (dz_2_inv * (theta_p1 - 2.0 * theta + theta_m1) +
                                            theta_z * w_bar * dz_inv - strain)

            # Five-point vertical derivative.
            elif theta_zz == '5p':

                # 2p in vertical advection.
                theta_now = theta + dtau * (fact * dz_2_inv * (- theta_m2 + 16.0 * theta_m1 - 30.0 * theta +
                                                               16.0 * theta_p1 - theta_p2) +
                                            theta_z * w_bar * dz_inv - strain)

                # Points near the border require a three-point discretisation instead.
                theta_now[1] = theta[1] + dtau * (dz_2_inv * (theta[2] - 2.0 * theta[1] + theta[0]) +
                                                  dz_inv * theta_z[1] * w_bar[1] - strain)

                theta_now[n-2] = theta[n-2] + dtau * (dz_2_inv * (theta[n-1] - 2.0 * theta[n-2] + theta[n-3]) +
                                                      dz_inv * theta_z[n-2] * w_bar[n-2] - strain)

            # BOUNDARY CONDITIONS. Geothermal heat flow at the base.
            # Two-point discretization.
            if bc == '2p':
                theta_now[0] = theta_now[1] + dz * (- G_k)

            # Three-point discretization.
            # du/dz = 0.5 * ( - 3.0 * u(0) + 4.0 * u(1) - u(2) )
            elif bc == '3p':
                theta_now[0] = (4.0 * theta_now[1] -
                                theta_now[2] + 2.0 * dz * (- G_k)) / 3.0

            # Four-point discretization.
            # du/dz = ( + 2 * f[i-3] - 9 * f[i-2] + 18 * f[i-1] - 11 * f[i+0] ) / (6*h)
            elif bc == '4p':
                theta_now[0] = (18.0 * theta_now[1] - 9.0 * theta_now[2] +
                                2.0 * theta_now[3] + 6.0 * dz * (- G_k)) / 11.0

            # Surface. beta * ( ( theta_now[n-1] - theta_now[n-2] ) / dz ) + theta_now[n-1] = T_air
            #theta_now[n-1] = ( beta_L * theta_now[n-2] + 1.0 ) / ( 1.0 + beta_L  )
            theta_now[n-1] = (beta_L * theta_now[n-2] + 1.0) / (1.0 + (beta_L / dz))

        if tau > out[c]:
            print('tau = ', np.round(tau, 2))

            tau_plot[c] = tau
            theta_all[:, c] = theta_now
            c += 1

        # Update variable.
        theta = theta_now

        # Update time.
        tau += dtau

    # Save last solution.
    tau_plot[n_out-1] = tau
    theta_all[:, n_out-1] = theta_now

    return [theta_all, tau_plot]


# Function that get numerical and analytical solution and error between them.
def f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
          G_k, tau_f, dtau, n_s, n_out, m, adv, xi, theta_zz, bc, theta_z_w, T_air, uneven, dxi):

    # Numerical solution
    sol_num = f_num(theta_0, w_bar, strain, beta_L, G_k,
                    tau_f, dtau, n_s, n_out, theta_zz, bc, theta_z_w, uneven, dxi)[0]

    # Analytical solution (stationary).
    sol_ana = np.transpose(stat_sol(w_0_bar, G_k, beta_L, strain, m, adv, xi, n_s)[1])

    # Transform to dimensional variables.
    #T_air = 223.15
    sol_num = T_air * np.array(sol_num) - 273.15
    sol_ana = T_air * np.array(sol_ana) - 273.15

    # Differences and error between numerical and analytical.
    diff = sol_num[:,n_out-1] - sol_ana
    error = np.linalg.norm(diff)

    return [sol_ana, sol_num, diff, error]


if numerical == True:

    vertical_res        = False
    discretisation      = True
    plot_vertical_res   = False
    plot_discretisation = True
    uneven = True

    if vertical_res == True:
        # Temporal output frequency.
        n_out = 3

        # Array with values to explore resolution.
        n_s = np.array([5, 10, 15, 20, 25, 30], dtype=int)
        #n_s = np.array([5, 10, 15], dtype=int)
        l = len(n_s)

        #dtau = np.array([5.0e-5, 5.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 2.0e-6, 2.0e-6, 1.0e-6])
        dtau = np.array([5.0e-5, 5.0e-5, 5.0e-5, 2.0e-5, 1.0e-5, 1.0e-5])

        # Dimensionless parameters of the solution.
        G_k = - 0.4 # 2.0

        # We now define the maximum dimensionless time.
        tau_f = 4.0   # 2.5
        out = np.linspace(0.0, 1.0, n_out)
        tau = out * tau_f

        # Power-law formulation of velocity vertical profiles.
        m = 1.0
        adv = 'linear'

        # Discretization options.
        theta_zz = '3p'
        bc = '2p'             # '2p'
        theta_z_w = 'symmetric_2p'    # 'backward_2p'

        # Prepare all variables.
        sol_num_all = []
        sol_ana_all = []
        sol_num_dif = []
        sol_ana_dif = []
        sol_num_adv = []
        sol_ana_adv = []
        sol_num_str = []
        sol_ana_str = []
        sol_num_hor = []
        sol_ana_hor = []

        error_dif = []
        diff_dif = []
        error_adv = []
        diff_adv = []
        error_str = []
        diff_str = []
        error_hor = []
        diff_hor = []

        xi_s = []
        xi_uneven = []

        # Loop over all spatial resolutions.
        for i in range(l):

            print('n = ', n_s[i])

            # Initial conditions.
            theta_0 = np.full(n_s[i], 1.0)

            # Uneven grid.
            zeta = np.linspace(0.0, 1.0, n_s[i])
            n_grid = 2.0
            xi = zeta**n_grid
            dxi = np.empty(n_s[i]-1)

            # Save array with vertical coordinate.
            xi_s.append(xi)

            # Create mesh.
            XI, TAU = np.meshgrid(xi, tau)

            # Create array with uneven spacing.
            for j in range(n_s[i]-1):
                dxi[j] = xi[j+1] - xi[j]

            ###########################################################################################
            # EXP-1. PURELY DIFUSSIVE CASE.
            # Dimensionless parameters of the solution.
            w_0_bar = 1.0e-4     # vertical advection at the surface
            # Linear dependency from w_0 to 0 at the base.
            w_bar = w_0_bar * xi
            beta_L = 0.0
            strain = 0.0
            G_k    = - 0.1
            T_air = 223.15

            # Call numerical and analytical solution for a certain parameter choice.
            sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                  G_k, tau_f, dtau[i], n_s[i], n_out, m, adv,
                                                  xi, theta_zz, bc, theta_z_w, T_air, uneven, dxi)

            sol_ana_dif.append(sol_ana)
            sol_num_dif.append(sol_num)
            diff_dif.append(diff)
            error_dif.append(error)
            ###########################################################################################

            ###########################################################################################
            # DIFUSSION + ADVECTION.
            w_0_bar = 5.0     # vertical advection at the surface, 7.0
            # Linear dependency from w_0 to 0 at the base.
            w_bar = - w_0_bar * xi
            beta_L = 0.0
            strain = 0.0
            G_k = - 0.35
            T_air = 223.15

            # Call numerical and analytical solution for a certain parameter choice.
            sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                  G_k, tau_f, dtau[i], n_s[i], n_out, m, adv, xi,
                                                  theta_zz, bc, theta_z_w, T_air, uneven, dxi)

            sol_ana_adv.append(sol_ana)
            sol_num_adv.append(sol_num)
            diff_adv.append(diff)
            error_adv.append(error)
            ###########################################################################################

            ###########################################################################################
            # DIFUSSION + ADVECTION + STRAIN.
            w_0_bar = 7.0     # vertical advection at the surface
            # Linear dependency from w_0 to 0 at the base.
            w_bar = - w_0_bar * xi
            beta_L = 0.0
            # Negative value implies heat dissipation due to deformation.
            strain = -0.75 #-2.0
            G_k = - 0.05
            T_air = 223.15

            # Call numerical and analytical solution for a certain parameter choice.
            sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                  G_k, tau_f, dtau[i], n_s[i], n_out, m, adv, xi,
                                                  theta_zz, bc, theta_z_w, T_air, uneven, dxi)

            sol_ana_str.append(sol_ana)
            sol_num_str.append(sol_num)
            diff_str.append(diff)
            error_str.append(error)
            ###########################################################################################

            ###########################################################################################
            # DIFUSSION + ADVECTION + STRAIN + HOR. ADVECTION.
            w_0_bar = 7.0     # vertical advection at the surface. 7.0
            # Linear dependency from w_0 to 0 at the base.
            w_bar = - w_0_bar * xi
            beta_L = 0.0
            # Positive value implies the horizontal advection brings colder ice.
            strain = 1.2 # 7.5
            G_k    = -0.5
            T_air = 253.15

            # Call numerical and analytical solution for a certain parameter choice.
            sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                  G_k, tau_f, dtau[i], n_s[i], n_out, m, adv,
                                                  xi, theta_zz, bc, theta_z_w, T_air, uneven, dxi)

            sol_ana_hor.append(sol_ana)
            sol_num_hor.append(sol_num)
            diff_hor.append(diff)
            error_hor.append(error)
            ###########################################################################################

    if discretisation == True:

        # Temporal output frequency.
        n_out = 3

        # Number of points.
        n = 10

        # List with discretizations.
        # theta_z_w_s = ['backward_2p', 'backward_3p',  'backward_4p', 'forward_2p', 'forward_3p', 'symmetric_2p'].
        # theta_zz_s  = ['3p', '5p']
        theta_zz_s = ['3p']
        bc_s = ['2p', '3p']
        theta_z_w_s = ['backward_2p', 'symmetric_2p', 'backward_3p']

        theta_zz = theta_zz_s[0]

        #l_zz = len(theta_zz_s)
        l_bc = len(bc_s)
        l_w = len(theta_z_w_s)

        # Time definitions.
        tau_f = 4.0   # 2.5
        dtau = 5.0e-5  # 2.0e-6. 0.5 * dz / w_0_bar

        # Initial conditions.
        theta_0 = np.empty(n)
        theta_0[:] = 1.0

        # Create mesh.
        # Uneven grid.
        zeta = np.linspace(0.0, 1.0, n)

        # Polynomial spacing.
        n_grid = 2.0
        xi = zeta**n_grid

        # Exponential spacing.
        """s = 2.0
        xi = ( np.exp(s*zeta) - 1.0 ) / ( np.exp(s*1) - 1.0 )"""

        # Create mesh.
        XI, TAU = np.meshgrid(xi, tau)

        # Create array with uneven spacing.
        dxi = np.empty(n-1)
        for i in range(n-1):
            dxi[i] = xi[i+1] - xi[i]

        # Dimensionless parameters of the solution.
        G_k = - 2.0

        # We now define the maximum dimensionless time.
        out = np.linspace(0.0, 1.0, n_out)
        tau = out * tau_f

        # Power-law formulation of velocity vertical profiles.
        m = 1.0
        adv = 'linear'

        # Prepare variables.
        sol_ana_dif = np.empty([l_w, l_bc, n, n_out])
        sol_num_dif = np.empty([l_w, l_bc, n, n_out])
        sol_num_adv = np.empty([l_w, l_bc, n, n_out])
        sol_num_str = np.empty([l_w, l_bc, n, n_out])
        sol_num_hor = np.empty([l_w, l_bc, n, n_out])
        sol_ana_all = np.empty([l_w, l_bc, n])
        sol_ana_dif = np.empty([l_w, l_bc, n])
        sol_ana_adv = np.empty([l_w, l_bc, n])
        sol_ana_str = np.empty([l_w, l_bc, n])
        sol_ana_hor = np.empty([l_w, l_bc, n])

        error_dif = np.empty([l_w, l_bc])
        error_adv = np.empty([l_w, l_bc])
        error_str = np.empty([l_w, l_bc])
        error_hor = np.empty([l_w, l_bc])

        # Loop over all spatial resolutions.
        c = 0
        for i in range(l_w):
            for j in range(l_bc):

                print('Theta dis. = ', theta_z_w_s[i])
                print('BC dis.    = ', bc_s[j])

                ###########################################################################################
                # EXP-1. PURELY DIFUSSIVE CASE.
                # Dimensionless parameters of the solution.
                # Vertical advection at the surface.
                w_0_bar = 1.0e-4
                # Linear dependency from w_0 to 0 at the base.
                w_bar = w_0_bar * xi
                beta_L = 0.0
                strain = 0.0
                G_k   = -0.1
                T_air = 223.15

                # Call numerical and analytical solution for a certain parameter choice.
                sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                      G_k, tau_f, dtau, n, n_out, m, adv,
                                                      xi, theta_zz, bc_s[j], theta_z_w_s[i], T_air, uneven, dxi)

                error_dif[i, j] = error
                sol_ana_dif[i, j, :] = sol_ana
                sol_num_dif[i, j, :, :] = sol_num

                ###########################################################################################

                ###########################################################################################
                # EXP-2. DIFUSSION + ADVECTION.
                # Dimensionless parameters of the solution.
                w_0_bar = 5.0     # vertical advection at the surface
                # Linear dependency from w_0 to 0 at the base.
                w_bar = -w_0_bar * xi
                beta_L = 0.0
                strain = 0.0
                G_k = - 0.35
                T_air = 223.15

                # Call numerical and analytical solution for a certain parameter choice.
                sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                      G_k, tau_f, dtau, n, n_out, m, adv,
                                                      xi, theta_zz, bc_s[j], theta_z_w_s[i], T_air, uneven, dxi)

                error_adv[i, j] = error
                sol_ana_adv[i, j, :] = sol_ana
                sol_num_adv[i, j, :, :] = sol_num
                ###########################################################################################

                ###########################################################################################
                # EXP-3. DIFUSSION + ADVECTION + STRAIN.
                # Dimensionless parameters of the solution.
                w_0_bar = 7.0     # vertical advection at the surface
                # Linear dependency from w_0 to 0 at the base.
                w_bar = -w_0_bar * xi
                beta_L = 0.0
                # Negative value implies heat dissipation due to deformation.
                strain = -0.75 #-2.0
                G_k = - 0.05
                T_air = 223.15

                # Call numerical and analytical solution for a certain parameter choice.
                sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                      G_k, tau_f, dtau, n, n_out, m, adv,
                                                      xi, theta_zz, bc_s[j], theta_z_w_s[i], T_air, uneven, dxi)

                error_str[i, j] = error
                sol_ana_str[i, j, :] = sol_ana
                sol_num_str[i, j, :, :] = sol_num
                ###########################################################################################

                ###########################################################################################
                # EXP-4. DIFUSSION + ADVECTION + STRAIN + HOR. ADVECTION.
                # Dimensionless parameters of the solution.
                w_0_bar = 7.0     # vertical advection at the surface. 7.0
                # Linear dependency from w_0 to 0 at the base.
                w_bar = - w_0_bar * xi
                beta_L = 0.0
                # Positive value implies the horizontal advection brings colder ice.
                strain = 1.2 # 7.5
                G_k    = -0.5
                T_air = 253.15

                # Call numerical and analytical solution for a certain parameter choice.
                sol_ana, sol_num, diff, error = f_sol(theta_0, w_0_bar, w_bar, strain, beta_L,
                                                      G_k, tau_f, dtau, n, n_out, m, adv,
                                                      xi, theta_zz, bc_s[j], theta_z_w_s[i], T_air, uneven, dxi)

                error_hor[i, j] = error
                sol_ana_hor[i, j, :] = sol_ana
                sol_num_hor[i, j, :, :] = sol_num
                ###########################################################################################

                c += 1

    if plot_vertical_res == True:

        fig = plt.figure(dpi=600, figsize=(8, 8))
        plt.rcParams['text.usetex'] = True

        fig.subplots_adjust(hspace=0.3)

        gs = gridspec.GridSpec(2, 4, height_ratios=[
                               5, 2], width_ratios=[1, 1, 1, 1])

        # Create the first subplot with a size ratio of 2:1
        ax = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[0, 3])
        ax5 = plt.subplot(gs[1, :])

        # Create an inset axis at the upper right corner 25%
        axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
        axins2 = inset_axes(ax2, width="30%", height="30%", loc='upper right')
        axins3 = inset_axes(ax3, width="30%", height="30%", loc='upper right')
        axins4 = inset_axes(ax4, width="30%", height="30%", loc='upper right')

        # Numerical stationary.
        idx_plot = np.array([0, 1, 5])
        #idx_plot = np.array([0, 1, 2])
        #l = len(idx_plot)
        l = len(n_s)

        # Colours spatial resolution.
        col_1 = np.linspace(0.0, 1.0, l)
        col_2 = np.linspace(0.0, 1.0e-3, l)
        col_3 = np.linspace(1.0, 0.0, l)

        linestyle = ['--']

        c = 0
        # for i in range(0, l, 1):
        for i in idx_plot:

            # Difussive.
            ax.plot(sol_num_dif[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                    marker='None', linewidth=1.5, alpha=1.0,
                    label=r'$ n = \ $'+str(n_s[i]))

            axins.plot(sol_num_dif[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                       marker='None', linewidth=0.75, alpha=1.0,
                       label=r'$ n = \ $'+str(n_s[i]))

            # Difussion + advection.
            ax2.plot(sol_num_adv[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                     marker='None', linewidth=1.5, alpha=1.0,
                     label=r'$ n = \ $'+str(n_s[i]))

            axins2.plot(sol_num_adv[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                        marker='None', linewidth=0.75, alpha=1.0,
                        label=r'$ n = \ $'+str(n_s[i]))

            # Difussion + advection + strain.
            ax3.plot(sol_num_str[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                     marker='None', linewidth=1.5, alpha=1.0,
                     label=r'$ n = \ $'+str(n_s[i]))

            axins3.plot(sol_num_str[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                        marker='None', linewidth=0.75, alpha=1.0,
                        label=r'$ n = \ $'+str(n_s[i]))

            # Difussion + advection + strain + hor. advection.
            ax4.plot(sol_num_hor[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                     marker='None', linewidth=1.5, alpha=1.0,
                     label=r'$ n = \ $'+str(n_s[i]))

            axins4.plot(sol_num_hor[i][:, n_out-1], xi_s[i], linestyle=linestyle[0], color=[col_1[i], col_2[i], col_3[i]],
                        marker='None', linewidth=0.75, alpha=1.0,
                        label=r'$ n = \ $'+str(n_s[i]))

            c += 1

        # Plot analytical solution.
        ax.plot(sol_ana_dif[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                marker='None', linewidth=2.5, alpha=1.0,
                label=r'$  \mathrm{Ana.} $')

        axins.plot(sol_ana_dif[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                   marker='None', linewidth=1.25, alpha=1.0,
                   label=r'$  \mathrm{Analytical} $')

        ax2.plot(sol_ana_adv[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                 marker='None', linewidth=2.5, alpha=1.0,
                 label=r'$  \mathrm{Ana.} $')

        axins2.plot(sol_ana_adv[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                    marker='None', linewidth=1.25, alpha=1.0,
                    label=r'$  \mathrm{Analytical} $')

        ax3.plot(sol_ana_str[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                 marker='None', linewidth=2.5, alpha=1.0,
                 label=r'$  \mathrm{Analytical} $')

        axins3.plot(sol_ana_str[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                    marker='None', linewidth=1.25, alpha=1.0,
                    label=r'$  \mathrm{Analytical} $')

        ax4.plot(sol_ana_hor[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                 marker='None', linewidth=2.5, alpha=1.0,
                 label=r'$  \mathrm{Analytical} $')

        axins4.plot(sol_ana_hor[l-1], xi_s[l-1], linestyle='-', color='black', zorder=0,
                    marker='None', linewidth=1.25, alpha=1.0,
                    label=r'$  \mathrm{Analytical} $')

        # Error compared to analytical as a function of resolution.
        ax5.plot(n_s, error_dif, linestyle='None', color='black',
                 marker='x', linewidth=1.5, alpha=1.0,
                 label=r'$ \mathrm{Exp. \ 1} $')

        ax5.plot(n_s, error_adv, linestyle='None', color='purple',
                 marker='s', linewidth=1.5, alpha=1.0,
                 label=r'$ \mathrm{Exp. \ 2}  $')

        ax5.plot(n_s, error_str, linestyle='None', color='darkgreen',
                 marker='o', linewidth=1.5, alpha=1.0, zorder=4,
                 label=r'$ \mathrm{Exp. \ 3} $')

        ax5.plot(n_s, error_hor, linestyle='None', color='darkorange',
                 marker='^', linewidth=1.5, alpha=1.0, zorder=0,
                 label=r'$ \mathrm{Exp. \ 4} $')

        # Legend.
        ax.legend(loc='lower left', ncol=1, frameon=True, framealpha=1.0,
                  fontsize=11, fancybox=True)

        ax5.legend(loc='lower left', ncol=1, frameon=True, framealpha=1.0,
                   fontsize=11, fancybox=True)

        # Add labels and title to plot
        ax.set_ylabel(r'$ \xi \ $', fontsize=20)
        ax.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)
        ax2.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)
        ax3.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)
        ax4.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)

        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])

        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticklabels(
            ['$0$', '$0.1$', '$0.2$', '$0.3$', '$0.4$'], fontsize=12)

        ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax2.set_yticklabels(['', '', '', '', ''], fontsize=12)

        ax3.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax3.set_yticklabels(['', '', '', '', ''], fontsize=12)

        ax4.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax4.set_yticklabels(['', '', '', '', ''], fontsize=12)

        ax5.set_yticks([1.0e-3, 1.0e-2, 1.0e-1])
        ax5.set_yticklabels(
            ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$'], fontsize=12)

        ax.set_xticks([-40, -20, 0])
        ax.set_xticklabels(['$-40$', '$-20$', '$0$'], fontsize=12)

        ax2.set_xticks([-40, -20, 0])
        ax2.set_xticklabels(['$-40$', '$-20$', '$0$'], fontsize=12)

        ax3.set_xticks([-30, -15, 0])
        ax3.set_xticklabels(['$-30$', '$-15$', '$0$'], fontsize=12)

        ax4.set_xticks([-50, -35, -20])
        ax4.set_xticklabels(['$-50$', '$-35$', '$-20$'], fontsize=12)

        ax5.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax5.set_xticklabels(['$0$', '$5$', '$10$', '$15$',
                            '$20$', '$25$', '$30$'], fontsize=12)
        

        axins.set_yticks([0, 0.5, 1.0])
        axins.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)

        axins2.set_yticks([0, 0.5, 1.0])
        axins2.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)

        axins3.set_yticks([0, 0.5, 1.0])
        axins3.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)

        axins4.set_yticks([0, 0.5, 1.0])
        axins4.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)


        ax2.tick_params(axis='y', which='major', length=0, colors='black')
        ax3.tick_params(axis='y', which='major', length=0, colors='black')
        ax4.tick_params(axis='y', which='major', length=0, colors='black')

        ax5.set_xlabel(r'$ n $', fontsize=20)
        ax5.set_ylabel(r'$ \varepsilon $', fontsize=20)

        ax.grid(axis='y', which='major', alpha=0.85)
        ax2.grid(axis='y', which='major', alpha=0.85)
        ax3.grid(axis='y', which='major', alpha=0.85)
        ax4.grid(axis='y', which='major', alpha=0.85)

        ax5.grid(axis='y', which='major', alpha=0.85)
        ax5.grid(axis='y', which='minor', alpha=0.25)
        ax5.grid(axis='x', which='major', alpha=0.85)

        ax5.set_yscale('log')

        # Limits
        ax.set_ylim(0.0, 0.4)
        ax2.set_ylim(0.0, 0.4)
        ax3.set_ylim(0.0, 0.4)
        ax4.set_ylim(0.0, 0.4)
        ax5.set_ylim(1.0e-1, 1.0e1)

        """ax.set_xlim(2.0, 3.0)
        ax2.set_xlim(1.0, 2.0)
        ax3.set_xlim(1.5, 2.5)
        ax4.set_xlim(0.0, 0.6)
        ax5.set_xlim(2.5, 32.5)"""

        ax.set_xlim(-40, 0.0)
        ax2.set_xlim(-40, 0.0)
        ax3.set_xlim(-30, 0.0)
        ax4.set_xlim(-50, -20.0)
        ax5.set_xlim(4, 31)

        # Adjust the inset axis properties as needed

        axins.set_xlim(-50, 0.0)
        axins.set_ylim(0.0, 1.0)
        axins2.set_xlim(-50, 0.0)
        axins2.set_ylim(0.0, 1.0)
        axins3.set_xlim(-50, 0.0)
        axins3.set_ylim(0.0, 1.0)
        axins4.set_xlim(-50, 0.0)
        axins4.set_ylim(0.0, 1.0)

        # Title.
        ax.set_title(r'$  (\mathrm{a}) $', fontsize=18, loc='center', pad=10)
        ax2.set_title(r'$ (\mathrm{b})  $', fontsize=18, loc='center', pad=10)
        ax3.set_title(r'$ (\mathrm{c})  $', fontsize=18, loc='center', pad=10)
        ax4.set_title(r'$ (\mathrm{d})  $', fontsize=18, loc='center', pad=10)
        ax5.set_title(r'$ (\mathrm{e})  $', fontsize=18, loc='center', pad=10)

        # Save figure.
        if save_fig == True:
            name_fig = 'numerical_res'
            plt.savefig(path_fig+name_fig+'.png',
                        format="png", bbox_inches='tight')
            plt.savefig(path_fig+name_fig+'.pdf',
                        format="pdf", bbox_inches="tight")

        # Display plot
        plt.show()
        plt.close(fig)

    if plot_discretisation == True:

        # Dimensions.
        #l_zz = len(theta_zz_s)
        l_bc = len(bc_s)
        l_w = len(theta_z_w_s)

        # Vertical axis.
        #n  = 10
        #xi = np.linspace(0.0, 1.0, n)

        # Plots.
        fig = plt.figure(dpi=600, figsize=(8, 8))
        plt.rcParams['text.usetex'] = True

        fig.subplots_adjust(hspace=0.3)

        gs = gridspec.GridSpec(2, 4, height_ratios=[5, 2], width_ratios=[
                               1, 1, 1, 1])  # height_ratios=[5, 2]

        # Create the first subplot with a size ratio of 2:1
        ax = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[0, 3])
        ax5 = plt.subplot(gs[1, :])

        # Create an inset axis at the upper right corner
        axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
        axins2 = inset_axes(ax2, width="30%", height="30%", loc='upper right')
        axins3 = inset_axes(ax3, width="30%", height="30%", loc='upper right')
        axins4 = inset_axes(ax4, width="30%", height="30%", loc='upper right')

        # Colours spatial resolution.
        col_1 = np.linspace(0.0, 1.0, l_bc)
        col_2 = np.linspace(0.25, 0.0, l_bc)
        col_3 = np.linspace(1.0, 0.0, l_bc)

        # Styles.
        line_style = ['--', '-.', ':']
        alpha_s = [1.0, 1.0, 1.0]  # [0.4, 1.0]
        marker_s = ['o', 'x', '^']

        # Error compared to analytical as a function of resolution.
        #colour_s = ['blue', 'red']

        # Plot analytical solution.
        # Analytical solution (stationary).
        n_ana = 100
        xi_ana = np.linspace(0.0, 1.0, n_ana)
        G_k = -2.0
        w_0_bar = 7.0
        beta_L = 0.0
        
        sol_ana_dif = np.transpose(stat_sol(1.0e-4, -0.1, beta_L, 0.0, m, adv, xi_ana, n_ana)[1])
        sol_ana_adv = np.transpose(stat_sol(5.0, -0.35, beta_L, 0.0, m, adv, xi_ana, n_ana)[1])
        sol_ana_str = np.transpose(stat_sol(w_0_bar, -0.05, beta_L, -0.75, m, adv, xi_ana, n_ana)[1])
        sol_ana_hor = np.transpose(stat_sol(w_0_bar, -0.5, beta_L, 1.2, m, adv, xi_ana, n_ana)[1])

        T_air = 223.15
        sol_ana_dif = T_air * np.array(sol_ana_dif) - 273.15
        sol_ana_adv = T_air * np.array(sol_ana_adv) - 273.15
        sol_ana_str = T_air * np.array(sol_ana_str) - 273.15
        sol_ana_hor = 253.15 * np.array(sol_ana_hor) - 273.15

        ax.plot(sol_ana_dif, xi_ana, linestyle='-', color='black',
                marker='None', linewidth=2.0, alpha=1.0,
                label=r'$  \mathrm{Ana.} $')

        axins.plot(sol_ana_dif, xi_ana, linestyle='-', color='black',
                   marker='None', linewidth=1.0, alpha=1.0,
                   label=r'$  \mathrm{Ana.} $')

        ax2.plot(sol_ana_adv, xi_ana, linestyle='-', color='black',
                 marker='None', linewidth=2.0, alpha=1.0,
                 label=r'$  \mathrm{Ana.} $')

        axins2.plot(sol_ana_adv, xi_ana, linestyle='-', color='black',
                    marker='None', linewidth=1.0, alpha=1.0,
                    label=r'$  \mathrm{Analytical} $')

        ax3.plot(sol_ana_str, xi_ana, linestyle='-', color='black',
                 marker='None', linewidth=2.0, alpha=1.0,
                 label=r'$  \mathrm{Analytical} $')

        axins3.plot(sol_ana_str, xi_ana, linestyle='-', color='black',
                    marker='None', linewidth=1.0, alpha=1.0,
                    label=r'$  \mathrm{Analytical} $')

        ax4.plot(sol_ana_hor, xi_ana, linestyle='-', color='black',
                 marker='None', linewidth=2.0, alpha=1.0,
                 label=r'$  \mathrm{Analytical} $')

        axins4.plot(sol_ana_hor, xi_ana, linestyle='-', color='black',
                    marker='None', linewidth=1.0, alpha=1.0,
                    label=r'$  \mathrm{Analytical} $')

        # Error plots.
        n_exp = 4
        x_error = np.linspace(0, 1, n_exp)

        # Numerical stationary.
        for i in range(l_w):
            for j in range(l_bc):

                # Difussive.
                ax.plot(sol_num_dif[i, j, :, n_out-1], xi, linestyle=line_style[i],
                        color=[col_1[j], col_2[j], col_3[j]],
                        marker='None', linewidth=1.5, alpha=alpha_s[i])

                axins.plot(sol_num_dif[i, j, :, n_out-1], xi, linestyle=line_style[i],
                           color=[col_1[j], col_2[j], col_3[j]],
                           marker='None', linewidth=0.75, alpha=alpha_s[i])

                # Difussion + advection.
                ax2.plot(sol_num_adv[i, j, :, n_out-1], xi, linestyle=line_style[i],
                         color=[col_1[j], col_2[j], col_3[j]],
                         marker='None', linewidth=1.5, alpha=alpha_s[i])

                axins2.plot(sol_num_adv[i, j, :, n_out-1], xi, linestyle=line_style[i],
                            color=[col_1[j], col_2[j], col_3[j]],
                            marker='None', linewidth=0.75, alpha=alpha_s[i])

                # Difussion + advection + strain.
                ax3.plot(sol_num_str[i, j, :, n_out-1], xi, linestyle=line_style[i],
                         color=[col_1[j], col_2[j], col_3[j]],
                         marker='None', linewidth=1.5, alpha=alpha_s[i])

                axins3.plot(sol_num_str[i, j, :, n_out-1], xi, linestyle=line_style[i],
                            color=[col_1[j], col_2[j], col_3[j]],
                            marker='None', linewidth=0.75, alpha=alpha_s[i])

                # Difussion + advection + strain + hor. advection.
                ax4.plot(sol_num_hor[i, j, :, n_out-1], xi, linestyle=line_style[i],
                         color=[col_1[j], col_2[j], col_3[j]],
                         marker='None', linewidth=1.5, alpha=alpha_s[i])

                axins4.plot(sol_num_hor[i, j, :, n_out-1], xi, linestyle=line_style[i],
                            color=[col_1[j], col_2[j], col_3[j]],
                            marker='None', linewidth=0.75, alpha=alpha_s[i])

        ax5.set_yscale('log')

        error_s = [error_dif, error_adv, error_str, error_hor]

        # Loop over all experiments.
        for i in range(n_exp):

            # Loop over derivative discretisations.
            for j in range(l_w):

                # Loop over BC discretisations.
                for k in range(l_bc):
                    ax5.plot(x_error[i], error_s[i][j, k], linestyle='None', color=[col_1[k], col_2[k], col_3[k]],
                             marker=marker_s[j], linewidth=2.5, alpha=1.0)

        # Labels profiles.
        ax.plot(np.nan, np.nan, linestyle=line_style[0],
                color='grey', marker='None', linewidth=2.5,
                alpha=1.0, label=r'$ \mathrm{F \mbox{-} 2p} $')

        ax.plot(np.nan, np.nan, linestyle=line_style[2],
                color='grey', marker='None', linewidth=2.5,
                alpha=1.0, label=r'$ \mathrm{F \mbox{-} 3p} $')

        ax.plot(np.nan, np.nan, linestyle=line_style[1],
                color='grey', marker='None', linewidth=2.5,
                alpha=1.0, label=r'$ \mathrm{S \mbox{-} 2p} $')

        # Labels error.
        ax5.plot(np.nan, np.nan, linestyle='None',
                 color='grey', marker=marker_s[0], linewidth=3.0,
                 alpha=1.0, label=r'$ \mathrm{F \mbox{-} 2p} $')

        ax5.plot(np.nan, np.nan, linestyle='None',
                 color='grey', marker=marker_s[2], linewidth=3.0,
                 alpha=1.0, label=r'$ \mathrm{F \mbox{-} 3p} $')

        ax5.plot(np.nan, np.nan, linestyle='None',
                 color='grey', marker=marker_s[1], linewidth=3.0,
                 alpha=1.0, label=r'$ \mathrm{S \mbox{-} 2p} $')

        ax5.plot(np.nan, np.nan, linestyle='--',
                 color=[col_1[0], col_2[0], col_3[0]], marker='None', linewidth=7.5,
                 alpha=1.0, label=r'$ \mathcal{O}_{\mathrm{BC} } (\varepsilon^1) $')

        ax5.plot(np.nan, np.nan, linestyle='--',
                 color=[col_1[1], col_2[1], col_3[1]], marker='None', linewidth=7.5,
                 alpha=1.0, label=r'$ \mathcal{O}_{\mathrm{BC} } (\varepsilon^2) $')

        # ax5.plot(np.nan, np.nan, linestyle='--', \
        #                    color=[col_1[2], col_2[2], col_3[2]], marker='None',linewidth=7.5, \
        #                        alpha=1.0, label=r'$ \mathcal{O}_{\mathrm{BC} } (\varepsilon^3) $')

        # Legend.
        ax.legend(loc='lower left', ncol=1, frameon=True, framealpha=1.0,
                  fontsize=12, fancybox=True)

        ax5.legend(loc='center left', ncol=1, frameon=True, framealpha=1.0,
                   fontsize=12, fancybox=True)

        # Add labels and title to plot
        ax.set_ylabel(r'$ \xi $', fontsize=20)
        ax.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)
        ax2.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)
        ax3.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)
        ax4.set_xlabel(r'$ \vartheta(\xi)$', fontsize=16)

        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticklabels(
            ['$0.0$', '$0.1$', '$0.2$', '$0.3$', '$0.4$'], fontsize=12)

        ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax2.set_yticklabels(['', '', '', '', ''], fontsize=12)

        ax3.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax3.set_yticklabels(['', '', '', '', ''], fontsize=12)

        ax4.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax4.set_yticklabels(['', '', '', '', ''], fontsize=12)

        ax.set_xticks([-40, -20, 0])
        ax.set_xticklabels(['$-40$', '$-20$', '$0$'], fontsize=12)

        ax2.set_xticks([-40, -20, 0])
        ax2.set_xticklabels(['$-40$', '$-20$', '$0$'], fontsize=12)

        ax3.set_xticks([-30, -15, 0])
        ax3.set_xticklabels(['$-30$', '$-15$', '$0$'], fontsize=12)

        ax4.set_xticks([-50, -35, -20])
        ax4.set_xticklabels(['$-50$', '$-35$', '$-20$'], fontsize=12)

        ax5.set_xticks([x_error[0], x_error[1], x_error[2], x_error[3]])
        ax5.set_xticklabels(['$\mathrm{Exp. \ 1}$', '$\mathrm{Exp. \ 2}$', 
                             '$\mathrm{Exp. \ 3}$', '$\mathrm{Exp. \ 4}$'], fontsize=15)
        
        ax5.set_yticks([1.0e-1, 1.0, 1.0e1])
        ax5.set_yticklabels(['$10^{-1}$', '$10^{0}$', '$10^{1}$'], fontsize=15)
        

        axins.set_yticks([0, 0.5, 1.0])
        axins.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)

        axins2.set_yticks([0, 0.5, 1.0])
        axins2.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)

        axins3.set_yticks([0, 0.5, 1.0])
        axins3.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)

        axins4.set_yticks([0, 0.5, 1.0])
        axins4.set_yticklabels(['$0$', '$0.5$', '$1.0$'], fontsize=10)


        ax2.tick_params(axis='y', which='major', length=0, colors='black')
        ax3.tick_params(axis='y', which='major', length=0, colors='black')
        ax4.tick_params(axis='y', which='major', length=0, colors='black')

        #ax5.set_xlabel(r'$ n $', fontsize=20)
        ax5.set_ylabel(r'$ \varepsilon $', fontsize=20)

        ax.grid(axis='y', which='major', alpha=0.85)
        ax2.grid(axis='y', which='major', alpha=0.85)
        ax3.grid(axis='y', which='major', alpha=0.85)
        ax4.grid(axis='y', which='major', alpha=0.85)

        ax5.grid(axis='y', which='major', alpha=0.85)
        ax5.grid(axis='y', which='minor', alpha=0.25)
        ax5.grid(axis='x', which='major', alpha=0.85)

        ax5.set_yscale('log')

        # Limits
        ax.set_ylim(0.0, 0.4)
        ax2.set_ylim(0.0, 0.4)
        ax3.set_ylim(0.0, 0.4)
        ax4.set_ylim(0.0, 0.4)
        ax5.set_ylim(1.0e-1, 1.0e1)

        ax.set_xlim(-40, 0.0)
        ax2.set_xlim(-40, 0.0)
        ax3.set_xlim(-30, 0.0)
        ax4.set_xlim(-50, -20.0)
        ax5.set_xlim(-0.1, 1.1)

        # Adjust the inset axis properties as needed

        axins.set_xlim(-50, 0.0)
        axins.set_ylim(0.0, 1.0)
        axins2.set_xlim(-50, 0.0)
        axins2.set_ylim(0.0, 1.0)
        axins3.set_xlim(-50, 0.0)
        axins3.set_ylim(0.0, 1.0)
        axins4.set_xlim(-50, 0.0)
        axins4.set_ylim(0.0, 1.0)

        # Title.
        ax.set_title(r'$  (\mathrm{a}) $', fontsize=18, loc='center', pad=10)
        ax2.set_title(r'$ (\mathrm{b})  $', fontsize=18, loc='center', pad=10)
        ax3.set_title(r'$ (\mathrm{c})  $', fontsize=18, loc='center', pad=10)
        ax4.set_title(r'$ (\mathrm{d})  $', fontsize=18, loc='center', pad=10)
        ax5.set_title(r'$ (\mathrm{e})  $', fontsize=18, loc='center', pad=10)

        # Save figure.
        if save_fig == True:
            name_fig = 'numerical'
            plt.savefig(path_fig+name_fig+'.png',
                        format="png", bbox_inches='tight')
            plt.savefig(path_fig+name_fig+'.pdf',
                        format="pdf", bbox_inches="tight")

        # Display plot
        plt.show()
        plt.close(fig)
