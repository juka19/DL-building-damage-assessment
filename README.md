<h1 align="center"> "Evaluating Deep Learning based Building Damage Assessment Methods in earthquake-affected, densely built-up urban areas: The case of Kahramanmaraş" </h1> <br>

<p align="center">
 *Repository for my master thesis for the degree of
 Master of Data Science for Public Policy*
<p align="center">

## Table of Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Data](#data)
- [xView-baseline](#xView-baseline)
- [ChangeOS](#ChangeOS)
- [MS4D-Net](#MS4D-Net)

## Abstract (#abstract)
<p align="center">
In post-disaster settings, damage assessments need to be conducted fast and reliably. To this end, deep learning approaches for building damage assessment have been researched and various models have been developed. However, the real-world performance on off-nadir post-event imagery of earthquakes in densely built-up urban areas still remains underexplored. In this analysis, a dataset for Kahramanmaraş, a Turkish city affected by the 2023 earthquake in the East Anatolian Fault Zone is created by combining open-source building footprints, emergency mapping information, and high-resolution open satellite imagery. Three different approaches are tested against the dataset: the [xView2-baseline](https://github.com/DIUx-xView/xView2_baseline) damage classification model component, combined with open-source building footprints as localization, the Multitask-Based Semi-Supervised Semantic Segmentation Framework [MS4D-Net](https://github.com/YJ-He/MS4D-Net-Building-Damage-Assessment), and the deep object-based semantic change detection framework [ChangeOS](https://github.com/Z-Zheng/ChangeOS). The findings suggest that earthquake building damage in densely built-up urban setting poses significant challenges for model performance. The ChangeOS framework outperforms the other approaches, although robustness checks indicate that the model does not reliably predict the same damage scene on different imagery.
<p align="center">

## Overview (#overview)

The project relies on one main data preparation pipeline that was used to create the main dataset for Kahramanmaraş. The data pipeline brings together OpenStreetMap building footprints, EU Copernicus EMS building damage information and satellite data from the Maxar Open Data Program. Next to this, a subset of the xBD dataset is used to train the supervised component of the MS4D-Net.


## Data (#data)

The pipeline starts with the retrieval of the satellite data from the Maxar Open Data Program. First, the download links are retrieved, then starts the retrieval of the iamgery. Then, Building footprints are triangulated with the building damage information form the Copernicus EMS for Kahramanmaraş. 
