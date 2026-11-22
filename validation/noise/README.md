# Instrument noise in the hGRS inversion (PRISMA, EnMAP)

Study plan, 22 November 2026. Goal: include the radiometric noise of the sensors in the inversion of the aerosol
load and the sunglint, to limit the artefacts of the retrieved AOT: along-track stripes (PRISMA scene of
2026-02-06, Figure 3 of the article draft), and the AOT pinned at its lower bound when the non-negativity
constraint is infeasible (EnMAP, AERONET-OC validation).

## Why

The aerosol fit uses the SWIR bands (SWIR detector), and the non-negativity constraint the visible-NIR bands
(VNIR detector, 430-865 nm, plus 1020 nm). Over dark water, the NIR water signal is of the order of 1e-3 in TOA
reflectance, the same order as the noise and as the radiometric differences between the detectors. The current
inversion treats the bands as noise-free: the hard constraint is set by the most limiting band, and the soft one uses
a single tolerance `sigma_+ = 2e-3` and weight `sigma_R = 1e-3` for all bands and sensors.

## First results (`noise_study.py`)

PRISMA, 2026-02-06 (Brazil, strong sunglint, 1e6 water pixels), native geometry, TOA reflectance:

| band (nm) | water signal | random (pixel) | column pattern | cross-track (20 columns) | mega-pixel total |
|---|---|---|---|---|---|
| 490 | 0.187 | 2.0e-3 | 1.5e-4 | 1.0e-3 | 1.0e-3 |
| 665 | 0.117 | 2.3e-3 | 1.5e-4 | 4.2e-4 | 4.3e-4 |
| 865 | 0.109 | 2.5e-3 | 1.6e-4 | 3.8e-4 | 4.0e-4 |
| 1650 | 0.095 | 1.9e-3 | 2.3e-4 | 3.5e-4 | 3.6e-4 |
| 2200 | 0.074 | 1.8e-3 | - | 2.3e-4 | 2.5e-4 |

- The random term includes the pixel-to-pixel variability of the glint (waves), an upper bound of the noise; in the
  mean of a mega-pixel (400 pixels) it is negligible (1e-4).
- The pattern across the swath does not average out along track. The VNIR/SWIR ratio over water (750/1650,
  865/2200 nm) varies by +-1 % across the swath, with features 50-100 columns wide, i.e. -0.6e-3 to +1.3e-3 in
  R(865) (`colprofile_brazil.png`); the ratio 1020/1650 (end of the VNIR detector) varies by +-0.2 % only. This
  relative non-uniformity of the two detectors, at the scale of 2-3 mega-pixels, matches the width of the AOT
  stripes: where the VNIR NIR is ~1e-3 low relative to the SWIR, the constraint is infeasible and the AOT drops to
  its bound.
- Part of the cross-track variation is real (glint and aerosol gradients across the swath): the quadratic trend is
  removed, but the residual is an upper bound of the instrumental part.

Outputs: `/data/satellite/noise_study/` (`noise_<product>.csv`, `.png`, `colprofile_<product>.npy`).

## Plan

1. **Noise characterisation** (per sensor, per band):
   - random noise as a function of the signal, `sigma^2(L) = a + b L`, from the validation crops of clear,
     glint-free scenes (the glint inflates the pixel differences) and from the overlap of the two detectors;
   - cross-track pattern of the VNIR relative to the SWIR, per scene (water pixels, glint used as a spectrally flat
     reference) and its stability between scenes (one table per sensor if stable);
   - EnMAP: the L1C is orthorectified (columns are not detector columns, resampling correlates the pixels): use the
     L1B products, or map the L1C pixels to detector columns with the RPC of the metadata;
   - compare with the mission SNR requirements and published in-flight performances (to cite after checking).
2. **Noise model in the inversion** (experimental settings, defaults unchanged):
   - weights of the SWIR fit per band, `sigma_R(lambda)^2 = sigma_random^2 / N + sigma_pattern^2`;
   - tolerance of the non-negativity per band, hard constraint relaxed to `R - R_sim >= -k sigma(lambda)` and soft
     penalty with `sigma_+(lambda) = k sigma(lambda)`; `k` to choose (1-2);
   - optional relative destriping: per-column VNIR/SWIR correction estimated in the scene over water before the
     inversion, against an anisotropic smoothing of the AOT along track.
3. **Evaluation**:
   - stripe metric: cross-track power of the AOT map at 2-5 mega-pixels on glint scenes (Brazil and others),
     fraction of mega-pixels at the lower bound of the AOT;
   - AERONET-OC and HYPERNETS statistics (AOT, Rrs per band) for each setting, both sensors, against the hard and
     soft constraints of 2.1.0;
   - effect on the residual peaks of the absorption bands.
4. **Article**: noise model in Section 3 (inversion), stripe metric and validation in the results.
