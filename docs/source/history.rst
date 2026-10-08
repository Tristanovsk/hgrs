Version history
===============

1.0.0
    Open and visualize PRISMA images.
1.0.1
    Orient North of rasters (waiting for proper georeference).
1.0.2
    Improvement of solar irradiance data and convolution for reflectance conversion.
1.0.3
    Development of the process kernel.
1.0.4
    Reorganizing modules.
1.0.5
    Investigate solar irradiance reference model.
1.0.6
    Add reprojection feature.
1.0.7 (2025-11-07)
    Correct bug for Earth-Sun distance, option to remove bad EnMAP bands, refactoring.
1.1.0
    New processor for aerosol optical thickness based on the non-negativity of the retrieved
    reflectance.
1.1.1
    Clean up; check which function (Gaussian, super-Gaussian) is the most suited for spectral
    integration.
1.1.2
    Set windows at [1, 1] for ``filter2d`` (smoothing).
