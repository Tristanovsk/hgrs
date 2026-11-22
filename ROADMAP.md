# Roadmap of hGRS

Status: 22 November 2026, after the release of 2.1.0. The tasks are grouped by priority; the links between the
packages (radcalnet_oc, GRS, grstbx, invRrs, coxmunk, SOS-ABS) are noted. Related: the roadmap of radcalnet_oc
(tasks 24, 26 and 28 concern hGRS).

## Where things stand

- **2.1.0** (21 November 2026): coupling of the gaseous absorption with the scattering (path-length model of
  radcalnet_oc, line-by-line O2-B), Rayleigh pressure scaling, log-log wavelength interpolation of the LUT, solar
  spectrum of the drivers, faster convolution, image catalog. Experimental and off by default: CAMS priors of the
  water vapour and aerosol retrievals, constraints of the coupled water-vapour model, EnMAP VNIR/SWIR detector
  correction.
- **AERONET-OC validation of EnMAP** (`validation/aeronet_oc`, 103 matchups, last runs 18 November, before 2.1.0):
  the AOT(550) is biased low (-0.08, MAPD 95 %): over dark water the retrieval stays at its lower bound, because the
  near-infrared TOA reflectance of EnMAP is below the simulated Rayleigh + water signal. The water vapour column is
  within -1.1 kg m-2 of AERONET (MAD 4.5).
- **O2-A band** (`docs/source/tutorials/research/o2a_enmap_closure.rst`): the 763.7 nm band of EnMAP reads about
  9 % low (bright land); the default REPTRAN in O2-A compensates it; 8 % over water left unexplained.
- **Closure per band with radcalnet_oc 1.2** (22 November, `closure_v12/`, 60 matchups after QC; direct glint,
  pure-water Rrs extension, pressure scaling): EnMAP / simulation 0.96-1.04 in the visible (445-707 nm), 1.09-1.39
  in the SWIR (+0.001 in reflectance), but **0.70-0.87 in the near infrared of the VNIR detector** (742: 0.82,
  779: 0.87, 864: 0.83, 888: 0.70; -0.004 to -0.006 in reflectance). The deficit is not the water term (7 % of the
  simulated signal at 864 nm, all matchups with a measured Rrs at 865-870 nm, no correlation), nor the aerosol
  model (no increase with the AOT; 0.85 at 864 nm for the 18 matchups with AOT < 0.06, where the path is mostly
  Rayleigh): **the VNIR detector of EnMAP reads 15-20 % low in the near infrared over dark water**, while the
  SWIR agrees once the glint is in. This is the VNIR/SWIR mismatch that `enmap_detector_offset: vnir_ramp`
  corrects (radcalnet_oc task 26). In the validation of 14 November, `vnir_ramp` halves the AOT bias (-0.075 to
  -0.034, MAPD 94 to 41 %) and improves the Rrs (MAPD 443 nm 79 to 22 %, 490 nm 62 to 29 %, 560 nm 35 to 17 %;
  865 nm 145 to 228 %); with the soft non-negativity and the CAMS prior of the AOT the AOT bias is -0.004.

## 1. Now (days)

1. **Closure with radcalnet_oc 1.2** (radcalnet_oc task 26). *Done (22 November)*, see above.
2. **Validation of 2.1.0** against AERONET-OC (EnMAP and PRISMA), two versions with `run_versions.py` when the
   workstation is free (memory budget, CLAUDE.local.md): the 2.1.0 defaults, and 2.1.0 with
   `enmap_detector_offset: vnir_ramp`, soft non-negativity and the CAMS prior of the AOT.
3. **Shape of the VNIR correction.** The closure gives the deficit per band (0.82 at 742 nm, 0.70 at 888 nm):
   compare it with the `vnir_ramp` ramp (0 at 700 nm, 1 at 990 nm) and with an offset or a gain; per-band
   ratio of the VNIR bands up to 990 nm and of the overlap with the SWIR.

## 2. Next (weeks)

4. **Defaults of the experimental options** from the 2.1.0 validation: the EnMAP detector correction first (the
   closure shows it is a sensor effect), then the CAMS priors (water vapour, aerosol) and the constraints of the
   coupled water-vapour model. Each becomes default only with a better validation (both sensors for the
   sensor-independent options).
5. **O2-A default.** Line by line + coupling is the consistent physics; REPTRAN is right on EnMAP only through the
   band deficit. Decide per sensor (band correction of EnMAP, line by line for PRISMA?) once the band deficit is
   fixed by an absolute reference: RadCalNet land closure (RVUS, GONA: EnMAP L1C order at DLR, RadCalNet portal).
6. **Shared physics tests** (radcalnet_oc task 28): reference inputs and outputs of the path-length coupling, the
   Rayleigh pressure scaling (reference pressure 1015.2 hPa) and the log-log interpolation, tested in the CI of
   radcalnet_oc, GRS and hGRS.
7. **Blue bias and pressure** (radcalnet_oc task 24): close the check of the Rayleigh pressure scaling on the 103
   EnMAP matchups (`compare_pressure`).

## 3. Later (months)

8. **Full scenes in bulk**: closed-form linear parameters and vectorized search of the 1-D LUT parameter
   (performance notes of CLAUDE.local.md), validated against the current version.
9. **Sun glint statistics**: the heavy-tailed slope statistics of coxmunk (`cm_iso_mix`) in the glint term of
   hGRS, with one wind source (radcalnet_oc task 29).
10. **Rrs reconstruction and fluorescence**: invRrs as the hyperspectral reference of the validation (in place of
    the linear interpolation of the AERONET-OC bands), and the fluorescence band of the water signal.
11. **Other sensors**: DESIS, EMIT, PACE/OCI (drivers, band responses, smile).
