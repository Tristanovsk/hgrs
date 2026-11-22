O2-A band: closure with EnMAP
=============================

Radiometric closure of the O2-A band (755-775 nm) of EnMAP at AERONET-OC stations and over bright bare land
(November 2026), to decide between the O2 transmittance models of hGRS (``o2_model``, ``gas_coupling``).

**Summary.** The EnMAP band at 763.7 nm reads about 9 % below the line-by-line O2 transmittance over bright bare
land, where the coupling of absorption and scattering hardly matters; its neighbours at 756 and 771 nm agree within
2.5 %. This band deficit explains about half of the closure gap over water with the line-by-line O2 and the coupling.
The other half (8 %) is not explained: once the band is corrected, the water data behave as if the coupling factor
of the path reflectance were 1.03-1.05, against 1.12-1.20 in SOS-ABS. The good closure of REPTRAN with the coupling
(``o2_model: lbl_o2b``, the default) is a compensation: REPTRAN absorbs too much, by about the band deficit.

Closure over water
------------------

``validation/aeronet_oc/closure.py`` simulates the TOA reflectance with ``radcalnet_oc`` from the AERONET-OC
measurements at the overpass (aerosol, water vapour, pressure, :math:`R_{rs}`) in the geometry of the EnMAP L1C image
(5 x 5 pixel median at the station), convolved with Gaussian bands of the metadata. The O2-A depth is the ratio of
the 763.7 nm band to the linear continuum of the 749.1 and 778.6 nm bands (``closure_stats.py``). 60 matchups at 9
sites, all processed with the EnMAP processor V010502:

.. list-table::
   :header-rows: 1

   * - model
     - band ratio
     - error (sim - EnMAP)
   * - EnMAP L1C
     - 0.574
     -  
   * - line by line, path-length coupling
     - 0.683
     - +0.109 ± 0.023
   * - line by line, decoupled
     - 0.614
     - +0.040
   * - REPTRAN, path-length coupling
     - 0.597
     - +0.023

The error with the line by line and the coupling is positive for all 60 matchups and the same at every site
(+0.087 to +0.132). It does not come from the simulation chain:

* **radcalnet_oc against SOS-ABS**: the path-length model reproduces the SOS-ABS coupling factor
  :math:`f_{path} = T_{atm}/T_d` within 1-2 % in the EnMAP band; line-resolved SOS-ABS runs (HITRAN Voigt
  cross sections in place of the CKD) give a coupling 0.03-0.04 stronger, the wrong sign to close the gap.
* **aerosol vertical profile**: :math:`f_{path}` is 1.12-1.20 for an aerosol scale height of 2 km and drops by
  only 0.009-0.019 at 0.5 km; polarisation and the aerosol model change it by less than 0.01.
* **surface-like signal** (water-leaving, glint, adjacency, which keep the direct-beam transmittance): no
  dependence on the glint angle (:math:`|\rho| \le 0.12`); bringing the apparent :math:`f_{path}` to 1.0 would
  need 80 % of the band signal from the surface, against about 5-15 % in the simulations.
* **spectral smile**: the O2-A band of EnMAP shifts by -0.11 to +0.27 nm across the swath (-0.06 nm at a typical
  station), and the error shows no trend with the column (left, centre, right thirds: +0.115, +0.119, +0.106).

The error behaves as a gain of the 763.7 nm band: in reflectance it is proportional to the signal
(:math:`\Delta R \approx -0.0005 + 0.23\, R`), 19-22 % of the signal at all brightness levels, and the 756 and
771 nm bands agree with every model (residual after removal of the smooth continuum, 756 / 763.7 / 771 nm:
+0.3 / +19.9 / +3.2 % for line by line with the coupling, -2.1 / +7.2 / +1.5 % decoupled, -2.0 / +5.2 / +0.5 %
for REPTRAN with the coupling).

**Band response.** With the simulations at full resolution (0.06 nm), a common shift and width change of the
Gaussian responses of the 756, 763.7 and 771 nm bands cannot close the line by line with the coupling: the best fit
(-0.25 nm, FWHM -3 nm) leaves 5-7 % in all three bands, and changing the 763.7 nm band alone (-1 nm, FWHM -2 nm)
leaves 9.5 %. The decoupled and REPTRAN models are closed within 2.4 % by -0.25 nm and FWHM -1 nm.

Bright bare land
----------------

Over a bright surface the TOA signal is mostly the surface term, whose transmittance is the direct-beam one: the
coupling raises the band transmittance by only 0.4-0.9 % at an albedo of 0.2-0.3 (SOS-ABS, AOT 0.05-0.2), against
13-20 % for dark water. The band ratio of bright bare pixels is therefore a test of the EnMAP band with little
dependence on the coupling model.

21 EnMAP L1C scenes of Bahia Blanca, Lampedusa, Senegal and Actium; pixels with :math:`R_{TOA}(778) > 0.15`,
NDVI < 0.12 and a flat continuum (:math:`|R_{778}/R_{749} - 1| < 0.12`); band ratio of each band against the
linear continuum of the 749.1 and 778.6 nm bands, divided by the line-by-line two-way direct transmittance of O2
(``hgrs.gas_coupling.O2LineByLine``, 1013 hPa, air mass of the scene) convolved with the same bands (median of the
pixels, then of the scenes, range):

.. list-table::
   :header-rows: 1

   * - processor version (scenes)
     - 756 nm
     - 763.7 nm
     - 771 nm
   * - V010502 (14), as the water matchups
     - 1.004
     - 0.917 (0.873-1.009)
     - 0.994
   * - V010506 (7), 2026
     - 1.023
     - 0.947 (0.897-0.963)
     - 0.988

With the coupling correction of the bright surface (+0.6 %), the deficit of the 763.7 nm band is about 9 % for
V010502 and 6 % for V010506. The V010506 scenes are all at an air mass of about 2.1, so the difference between the
versions is not established. The deficit tends to increase with the air mass (Spearman -0.32; 0.87-0.90 for
Bahia Blanca at air mass 3), but the slope is not significant (95 % interval -0.047 to +0.026 in ln per unit air
mass). The bright-pixel selection can include clouds, which raise the ratio: the single scene at 1.009 is probably
one.

**Signal level.** To test a deficit larger on dark targets (non-linearity, out-of-band response), the bare pixels
(0.02 < NDVI < 0.12, :math:`R_{865}/R_{778} > 0.95`, flat continuum) of the 21 scenes are binned by
:math:`R_{TOA}(778)` from 0.03 to 0.6. The coupling correction of each bin is interpolated in the SOS-ABS values
(albedo 0, 0.1, 0.2, 0.3: about 1.16, 1.02, 1.009, 1.005), with the albedo estimated as
:math:`(R_{778} - 0.015)/0.85`. Observed / predicted ratio at 763.7 nm, V010502 (median of the scenes of each bin):

.. list-table::
   :header-rows: 1

   * - :math:`R_{TOA}(778)`
     - scenes
     - uncorrected
     - coupling correction
     - corrected
   * - 0.06
     - 3
     - 0.903
     - 1.046
     - 0.862
   * - 0.08
     - 4
     - 0.891
     - 1.035
     - 0.862
   * - 0.12
     - 13
     - 0.902
     - 1.018
     - 0.886
   * - 0.13
     - 13
     - 0.914
     - 1.016
     - 0.901
   * - 0.18
     - 13
     - 0.928
     - 1.010
     - 0.918
   * - 0.21
     - 12
     - 0.925
     - 1.008
     - 0.917
   * - 0.29
     - 9
     - 0.906
     - 1.005
     - 0.901
   * - 0.41
     - 3
     - 0.896
     - 1.004
     - 0.892

Before correction the ratio is flat (0.89-0.93) over a factor of 7 in signal, and within each scene it does not
depend on the signal (median slope of :math:`\ln g` against :math:`\ln R_{778}`: +0.010, interquartile -0.03 to
+0.05, 20 scenes). The lower corrected values of the darkest bins come from the coupling correction, which is
uncertain there (albedo estimate, mixed pixels, few pixels). V010506 gives the same picture, with ratios 0.90-0.95.
Down to :math:`R_{778} \approx 0.06` there is no sign of a deficit growing at low signal. The water matchups are
darker (:math:`R_{778} \approx 0.03`), outside this range.

Water with the band corrected
-----------------------------

Dividing the EnMAP band ratio of the water matchups by the land deficit (median, or as a function of the air mass:
same result, the land scenes cover the air masses of the water matchups, 2.1-3.4), the ratio EnMAP / simulation
becomes (median [p25, p75]):

.. list-table::
   :header-rows: 1

   * - model
     - EnMAP corrected / sim
   * - line by line, path-length coupling
     - 0.92 [0.90, 0.94]
   * - line by line, decoupled
     - 1.03 [1.00, 1.07]
   * - REPTRAN, path-length coupling
     - 1.05 [1.02, 1.08]

The water data sit between the decoupled and the coupled line by line, closer to the decoupled one.

Interpretation and open questions
---------------------------------

* The EnMAP 763.7 nm band reads low by about 9 % (V010502). Soppa et al. (2024, Opt. Express 32(16), p. 28220)
  note O2-A residuals in the water-leaving reflectance of the EnMAP L2A processor, and ACOLITE leaves the band out.
* The remaining 8 % over water would need a coupling factor of 1.03-1.05 instead of 1.12-1.20, which nothing found
  in the SOS-ABS set-up or the ``radcalnet_oc`` chain gives. A band deficit larger on dark targets (non-linearity or
  out-of-band response of the 763.7 nm band) would also explain it, but the land pixels show none down to
  :math:`R_{TOA}(778) \approx 0.06`; the water signal (0.03) is only a factor of 2 below.
* PRISMA points the other way: in the comparison of versions (:doc:`../prisma_version_comparison`) the line by line
  leaves a smaller O2-A residual than REPTRAN. This also suggests a sensor-specific effect in EnMAP.
* A closure on a RadCalNet land site (Railroad Valley, Gobabeb), with the measured pressure and surface reflectance,
  would fix the absolute value of the band deficit.

Consequences for hGRS
---------------------

* The default ``o2_model: lbl_o2b`` (REPTRAN in the O2-A band) is kept: on EnMAP it gives the smallest residual, but
  through the compensation of two errors (REPTRAN too absorbing, EnMAP band too low). With a correction of the EnMAP
  763.7 nm band, the line by line would be the consistent choice, and the coupling of the path reflectance would
  still be too strong by about 8 % over water.
* A gain on the 763.7 nm band of EnMAP is not applied: its value may depend on the processor version and on the air
  mass, and it is not fixed by an absolute reference.

Scripts and data
----------------

In ``validation/aeronet_oc`` (outputs in ``closure/`` of the validation directory, ``HGRS_VAL_OUT``):

* ``closure.py``, ``closure_stats.py``: closure at the AERONET-OC stations and band ratios
  (``closure/band_depth_ratio.csv``);
* ``o2a_band_response.py``: full-resolution simulations (``closure/fine/``) and fit of the band responses
  (``closure/srf_fit.csv``);
* ``o2a_land.py``: bright bare land, per scene and per bin of :math:`R_{TOA}(778)` (``closure/land_o2a.csv``,
  ``closure/land_o2a_bins.csv``).

The tables of this page (November 2026) are in ``closure/o2a_20261121/``. The SOS-ABS simulations of the coupling
factor (aerosol scale height, bright surfaces) were made with ``radcalnet_oc``.
