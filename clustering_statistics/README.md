# DESI Clustering Statistics Pipeline and Products
## Overview
This repository contains…

## Data Access
The base directory is 
```/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/``` TODO: Establish an official directory.

Within the base directory, sub-directories correspond to the data and mocks clustering products. Below we list them:
* ```glam-uchuu-v1-altmtl```
* ```holi-v1-altmtl```
* ```abacus-2ndgen-complete```

Furthermore, within each of these, there is a corresponding directory relevant to each key project:
* ```full_shape```
* ```png_local```
* ```lensing```
* ```merged```: Merged data catalogs used for reshuffling the randoms to estimate the RIC. This folder only exists for mocks.

Finally, within each, we save the clustering products to folders named based on the statistic:
* ```pk```: power spectrum measurements.
* ```bk```: bispectrum measurements.
* ```xi```: two-point correlation function measurements.

## Documentation
### Reading clustering statistics

All clustering products follow a `base_filename` structure such that `base_filename = {tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_weight-{weight_type}{extra}_{imock}`, with:
* tracer ```tracer```: 'LRG', 'ELG_LOPnotqso', 'QSO'.

* region ```region```: 'NGC', 'SGC', or 'GCcomb'. Combined power spectrum measurements 'GCcomb' are the average of 'NGC' and 'SGC' power spectra, weighted by their normalization factor.

* redshift range ```zrange```:
  * For `full_shape`: (0.4, 0.6), (0.6, 0.8), (0.8, 1.1), (1.1, 1.6), (0.8, 2.1)
  * For `png_local`:  (0.4, 1.1), (0.8, 1.6), (0.8, 3.5)

* `weight_type`: identifies how the tracers were weighted. This can be any combination of weights, but the default choices are dependent on the KP and are `default-FKP` ('full shape') and `default-oqe` ('local png').

* `extra` is a suffix that can be any combination of extra processing done before, during, or after the measurement, separated by an underscore (`_`). Some default choices below:
    *  `_thetacut`: $\theta$-cut removes all pairs with angular separation < 0.05°, to mitigate fiber assignment effects.
    *  `_auw`: angular upweighting scheme [Bianchi et al. 2025](https://arxiv.org/pdf/2411.12025)...
    *  `_noric`: The redshifts of the randoms catalogs were reshuffled to remove the nulling of radial modes due to the 'shuffling' method. The 'shuffling' method subsamples the redshifts of the randoms from the data. NOTE: These are only used for the estimation of the radial integral constraint (RIC).

* `imock`: an integer that identifies for which mock it was computed. NOTE: This is not used for the observed data measurements.

Therefore, for each statistic: 
* ```pk```: `mesh2_spectrum_poles_{base_filename}.h5`
* ```bk```: `mesh3_spectrum_{basis}_poles_{base_filename}.h5`
    * `basis`: `sugiyama-diagonal`...
* ```xi```: `particle2_correlation_{base_filename}.h5`


An example of how the full path of a file would look: 
```
$BASEDIR/glam-uchuu-v1-altmtl/full_shape/pk/mesh2_spectrum_poles_LRG_z0.4-0.6_GCcomb_weight-default-FKP_thetacut_100.h5
```
Please refer to the `nb/example_read_stats.ipynb` for an example on how to load clustering statistics.

