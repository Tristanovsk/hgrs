Research
========

Studies behind the processing choices of hGRS, with the simulations and the test scenes they are based on. The
notebooks are stored executed; they need local data (SOS-ABS simulations, EnMAP test scenes, see
``coupling_tools.py``) and ``radcalnet_oc`` to be run again.

**Coupling of absorption and scattering.** The gaseous transmittance of the path reflectance is not that of the
direct beam. The three notebooks quantify the difference with the SOS-ABS coupled simulations of ``radcalnet_oc``,
test the water vapour retrieval and the O2 correction, and support the ``gas_coupling: path_length`` and
``o2_model: lbl_o2b`` settings (:py:mod:`hgrs.gas_coupling`).

.. toctree::
   :maxdepth: 1

   coupling_01_transmittance
   coupling_02_water_vapor
   coupling_03_o2_bands

**O2-A band of EnMAP.** Closure of the O2-A band with EnMAP at AERONET-OC stations and over bright bare land: the
763.7 nm band of EnMAP reads low, which the default REPTRAN model in the O2-A band compensates.

.. toctree::
   :maxdepth: 1

   o2a_enmap_closure
