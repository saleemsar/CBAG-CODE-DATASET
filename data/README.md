# Dataset: UAV-NIDD

The experiments use the public **UAV-NIDD** dataset (CC BY 4.0), which is not
redistributed here because of its size (about 575 MB).

Download it from Zenodo: <https://zenodo.org/records/15125851>

The notebooks use one labelled CSV per topology:

| Topology | File | Notebook |
|---|---|---|
| UAV1 (direct link) | `UAV-Case1-Label.csv` | `notebooks/UAV1_direct_link.ipynb` |
| UAV2 (access point) | `Access Point Case2 Label.csv` | `notebooks/UAV2_access_point.ipynb` |
| UAV3 (ground control station) | `GSC Case3 Label .csv` | `notebooks/UAV3_ground_control_station.ipynb` |

Place the files in this folder (or anywhere) and set the path in the CONFIG cell of
each notebook. The label column is `Label` and the benign class is `Normal`; the
Reconnaissance class is dropped.

Please cite the dataset authors when using UAV-NIDD:

> Hadi HJ, Cao Y, Khan MK, Ahmad N, Hu Y, Fu C (2025) UAV-NIDD: a dynamic dataset for
> cybersecurity and intrusion detection in UAV networks. IEEE Transactions on Network
> Science and Engineering 12(4):2739–2757. https://doi.org/10.1109/TNSE.2025.3553442
