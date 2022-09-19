# Welcome to the light curve approximation project

This is the official repository with the scripts and results of the experiments described in the following paper:

- M. Demianenko, K. Malanchev, E. Samorodova, M. Sysak, A. Shiriaev, D. Derkach, M. Hushchyn. Toward an understanding of the properties of neural network approaches for supernovae light curve approximation, 	arXiv:2209.07542 (2022). [[arxiv](https://doi.org/10.48550/arXiv.2209.07542)]

The light curve approximation methods are implemented in the [Fulu](https://github.com/HSE-LAMBDA/fulu) python library: [https://github.com/HSE-LAMBDA/fulu](https://github.com/HSE-LAMBDA/fulu)

## PLAsTiCC data

Find the full unblind dataset [on Zenodo](https://zenodo.org/record/2539456).
You can use [this script](https://github.com/HSE-LAMBDA/supernovae_classification/blob/master/notebooks/download_data.py) to download the data.

Column description (WIP)
- **MJD** is the [Modified Julian day](https://en.wikipedia.org/wiki/Julian_day). Note that the final user of the utility could use any time coordinate system in any units

## The Zwicky Transient Facility (ZTF) Bright Transient Survey

Find columns description [on BTS explorer](https://sites.astro.caltech.edu/ztf/bts/explorer_info.html)

Description mission and statistic data [on BTS webcite](https://sites.astro.caltech.edu/ztf/bts/bts.php)

The client for downloading ZTF data [ANTARES client](https://noao.gitlab.io/antares/client/api.html)
