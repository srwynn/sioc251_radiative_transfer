import json

def md(t):  return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def code(t):return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)}

cells=[]

cells.append(md("""# SIOC 251 — Homework 4
## Two-Stream SW Radiative Transfer and the Aerosol Direct Radiative Effect

**Part I (Section 5): Two-Stream Model Construction**

This notebook is a *starter scaffold* for Section 5.1. It sets up the wavelength grid,
reads the data, and defines the two-stream layer reflectance/transmittance functions
plus the layer-combining (adding) step. Sections 5.2 (verification), 5.3 (question),
and Parts II/III are stubbed for you to complete.

Equations referenced below are from the Module 8 class notes.
"""))

cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'figure.dpi': 120,
})

print('Imports complete.')"""))

cells.append(md("""## 5.1a — Spectral grid

$\\Delta\\lambda = 0.1\\,\\mu m$, $\\lambda \\in [0.4, 4]\\,\\mu m$."""))

cells.append(code("""lam = np.arange(0.4, 4.0 + 0.1, 0.1)   # wavelength grid (microns)
print(f'{lam.size} wavelengths from {lam[0]:.1f} to {lam[-1]:.1f} um')"""))

cells.append(md("""## 5.1b, 5.1c — Read aerosol optical properties and ocean reflectance

`aerosol_properties.csv` columns: `lambda (mu)`, `dust Be/g/omega`, `pollution Be/g/omega`.
`Be` is extinction normalized to 1 at 0.55 um. `omega` = single-scatter albedo, `g` = asymmetry.

`ocean_reflectance.csv` columns: `lambda (mu)`, `reflectance` -> surface reflectance r_s.

Both files are already on the 0.4->4.0 um / 0.1 um grid, so they line up with `lam`."""))

cells.append(code("""aero  = pd.read_csv('data/aerosol_properties.csv')
ocean = pd.read_csv('data/ocean_reflectance.csv')

# sanity check that the data grid matches our model grid
assert np.allclose(aero['lambda (mu)'].values, lam), 'aerosol grid mismatch'
assert np.allclose(ocean['lambda (mu)'].values, lam), 'ocean grid mismatch'

r_s = ocean['reflectance'].values   # spectral ocean surface reflectance

# pack each aerosol species into a dict for convenience
species = {
    'dust':      {'Be': aero['dust Be'].values,
                  'g':  aero['dust g'].values,
                  'omega': aero['dust omega'].values},
    'pollution': {'Be': aero['pollution Be'].values,
                  'g':  aero['pollution g'].values,
                  'omega': aero['pollution omega'].values},
}
aero.head()"""))

cells.append(md("""## Two-stream layer reflectance & transmittance

For a homogeneous absorbing+scattering layer of optical depth $\\tau^*$, single-scatter
albedo $\\tilde\\omega$, and asymmetry $g$:

**Similarity reflectance (Eq. 1):**
$$r_\\infty = \\frac{\\sqrt{1-\\tilde\\omega g}-\\sqrt{1-\\tilde\\omega}}{\\sqrt{1-\\tilde\\omega g}+\\sqrt{1-\\tilde\\omega}}$$

**Eigenvalue** (from the page-4 two-stream ODEs,
$\\tfrac12 d(I^\\uparrow-I^\\downarrow)/d\\tau=(1-\\tilde\\omega)(I^\\uparrow+I^\\downarrow)$ and
$\\tfrac12 d(I^\\uparrow+I^\\downarrow)/d\\tau=(1-\\tilde\\omega g)(I^\\uparrow-I^\\downarrow)$):
$$\\Gamma = 2\\sqrt{(1-\\tilde\\omega)(1-\\tilde\\omega g)}$$

> **VERIFY THIS (AI-use policy):** confirm the factor of 2 from your notes' page-4 ODEs.
> It sets the diffusivity: pure absorption ($\\tilde\\omega=0$) should give $t=e^{-2\\tau^*}$.

**Layer r and t (Eqs. 9 & 10):**
$$r=\\frac{r_\\infty(e^{\\Gamma\\tau^*}-e^{-\\Gamma\\tau^*})}{e^{\\Gamma\\tau^*}-r_\\infty^2 e^{-\\Gamma\\tau^*}},\\qquad
  t=\\frac{1-r_\\infty^2}{e^{\\Gamma\\tau^*}-r_\\infty^2 e^{-\\Gamma\\tau^*}}$$

For conservative scattering ($\\tilde\\omega=1$) these are 0/0, so use the non-absorbing
limit (Eqs. 6 & 7): $r=\\frac{(1-g)\\tau^*}{1+(1-g)\\tau^*}$, $t=\\frac{1}{1+(1-g)\\tau^*}$."""))

cells.append(code("""def layer_rt(tau, omega, g):
    \"\"\"Two-stream reflectance r and transmittance t of a single layer.

    Parameters (all array-like, per-wavelength):
        tau   : layer optical depth tau*
        omega : single-scatter albedo (0..1)
        g     : asymmetry parameter
    Returns: (r, t)
    \"\"\"
    tau   = np.asarray(tau, float)
    omega = np.asarray(omega, float)
    g     = np.asarray(g, float)

    r = np.empty_like(tau)
    t = np.empty_like(tau)

    # --- conservative-scattering case (omega == 1): Eqs 6-7 ---
    cons = np.isclose(omega, 1.0)
    if np.any(cons):
        u = (1 - g[cons]) * tau[cons]
        r[cons] = u / (1 + u)
        t[cons] = 1 / (1 + u)

    # --- general absorbing+scattering case: Eqs 1, 9, 10 ---
    gen = ~cons
    if np.any(gen):
        a = np.sqrt(1 - omega[gen]*g[gen])   # sqrt(1 - omega*g)
        b = np.sqrt(1 - omega[gen])          # sqrt(1 - omega)
        r_inf = (a - b) / (a + b)
        Gamma = 2 * a * b                    # 2*sqrt((1-omega)(1-omega*g))  <-- verify
        e = np.exp(Gamma * tau[gen])
        denom = e - r_inf**2 / e
        r[gen] = r_inf * (e - 1/e) / denom
        t[gen] = (1 - r_inf**2) / denom

    return r, t"""))

cells.append(md("""## 5.1d, 5.1e — Build the two atmospheric layers

**Aerosol layer (top), 5.1e:** $\\tau_{a,0.55}=1.0$, and since `Be` is normalized to 0.55 um,
$\\tau_a(\\lambda)=\\tau_{a,0.55}\\,\\mathrm{Be}(\\lambda)$.

**Rayleigh layer (below aerosol), 5.1d:** $\\tau_{R,0.55}=0.1$, $\\tau_R(\\lambda)=\\tau_{R,0.55}(\\lambda/0.55)^{-4}$,
pure scattering ($\\tilde\\omega=1$) and isotropic ($g=0$)."""))

cells.append(code("""TAU_A_055 = 1.0    # aerosol optical depth at 0.55 um
TAU_R_055 = 0.1    # Rayleigh optical depth at 0.55 um

# Rayleigh layer (same for both species)
tau_R = TAU_R_055 * (lam/0.55)**(-4)
r_R, t_R = layer_rt(tau_R, omega=np.ones_like(lam), g=np.zeros_like(lam))

def aerosol_layer(sp, tau0=TAU_A_055):
    \"\"\"Return (r_a, t_a) for a species dict at aerosol optical depth tau0 (at 0.55um).\"\"\"
    tau_a = tau0 * sp['Be']
    return layer_rt(tau_a, sp['omega'], sp['g'])"""))

cells.append(md("""## Combine layers over the reflecting ocean (adding method, Eqs. 11 & 12)

Each homogeneous layer reflects/transmits identically up or down, so we stack top-down.
Aerosol (1) over Rayleigh (2):
$$R_{aR}=r_a+\\frac{t_a^2\\,r_R}{1-r_a r_R},\\qquad T_{aR}=\\frac{t_a t_R}{1-r_a r_R}$$
Then the combined atmosphere over surface reflectance $r_s$:
$$\\underbrace{\\tilde r}_{\\text{TOA albedo}}=R_{aR}+\\frac{T_{aR}^2 r_s}{1-R_{aR}r_s},\\qquad
  \\underbrace{\\tilde t}_{\\text{surface transmittance}}=\\frac{T_{aR}}{1-R_{aR}r_s}$$"""))

cells.append(code("""def combine_two(r1, t1, r2, t2):
    \"\"\"Adding method: stack layer 1 (top) over layer 2 (bottom).\"\"\"
    d = 1 - r1*r2
    R = r1 + t1**2 * r2 / d
    T = t1 * t2 / d
    return R, T

def scene(sp, tau0=TAU_A_055, tau_R_055=TAU_R_055):
    \"\"\"Full column: aerosol + Rayleigh over ocean.
    Returns dict with TOA albedo and surface transmittance (spectral).\"\"\"
    r_a, t_a = aerosol_layer(sp, tau0)
    R_aR, T_aR = combine_two(r_a, t_a, r_R, t_R)   # atmosphere reflect/transmit
    # add reflecting surface (surface only reflects: Eqs 11-12)
    albedo_TOA = R_aR + T_aR**2 * r_s / (1 - R_aR*r_s)
    trans_sfc  = T_aR / (1 - R_aR*r_s)
    return {'albedo_TOA': albedo_TOA, 'trans_sfc': trans_sfc,
            'r_a': r_a, 't_a': t_a, 'R_aR': R_aR, 'T_aR': T_aR}

results = {name: scene(sp) for name, sp in species.items()}
print('Computed spectral albedo & transmittance for:', list(results))"""))

cells.append(md("""## 5.3 — Spectral transmittance (surface) and albedo (TOA)

Plot both quantities for dust and pollution. Then interpret: are the differences
consistent with each species' single-scatter properties ($\\tilde\\omega$, $g$)?"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for name in ['dust', 'pollution']:
    axes[0].plot(lam, results[name]['albedo_TOA'], lw=1.8, label=name)
    axes[1].plot(lam, results[name]['trans_sfc'],  lw=1.8, label=name)

axes[0].set(xlabel=r'$\\lambda$ ($\\mu$m)', ylabel='TOA albedo', title='TOA albedo')
axes[1].set(xlabel=r'$\\lambda$ ($\\mu$m)', ylabel='Surface transmittance', title='Surface transmittance')
for ax in axes:
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure_5p3_spectral_albedo_transmittance.png', bbox_inches='tight')
plt.show()"""))

cells.append(md("""## 5.2 — Verification (limiting cases) — TODO

Add at least two tests of limiting behavior, e.g.:
- **No aerosol, no Rayleigh** (set both optical depths ~0): TOA albedo should collapse to the
  ocean surface reflectance `r_s`, and surface transmittance -> 1.
- **Pure scattering aerosol** ($\\tilde\\omega=1$) vs **strongly absorbing** ($\\tilde\\omega$ small):
  check albedo rises with $\\tilde\\omega$, falls with $g$ (matches Module-8 Fig. 1).
- **Conservative + non-reflecting surface**: scene should conserve energy (albedo + transmittance = 1)."""))

cells.append(code("""# TODO: verification tests
# Example skeleton:
# tiny = scene(species['dust'], tau0=1e-6)
# assert np.allclose(tiny['albedo_TOA'], r_s, atol=1e-3)
"""))

cells.append(md("""---
# Part II (Section 6) — Shortwave fluxes & the DRE — TODO

- 6.1a: read `data/Fo.csv` (nm, W/m^2/nm); bin/interpolate F0 onto the 0.1 um grid.
- 6.1b: spectral up/down fluxes at TOA and surface; integrate to broadband SW fluxes.
- 6.1c: TOA, surface, atmospheric instantaneous DRE for dust & pollution (Eqs. 2-7).
- 6.2: a flux/DRE sanity test.  6.3: answer questions a-e (incl. forcing efficiency).

# Part III (Section 7) — Student investigation — TODO

# AI-use statement — TODO
State which AI tools you used, how, what you checked independently, and one thing you verified
(e.g., the Gamma factor of 2, or the no-aerosol limit collapsing to r_s)."""))

nb={"cells":cells,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3"}},
    "nbformat":4,"nbformat_minor":5}

with open('two_stream_aerosol_DRE.ipynb','w') as f:
    json.dump(nb,f,indent=1)
print('Notebook written with', len(cells), 'cells')
