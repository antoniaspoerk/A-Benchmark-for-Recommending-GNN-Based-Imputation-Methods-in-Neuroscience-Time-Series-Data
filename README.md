# A Benchmark for Recommending GNN-Based Imputation Methods in Neuroscience Sensory Data

Master's thesis project, MSc in Digital Neuroscience, University of Fribourg  
Author: Antonia Spörk · Supervisor: Dr. Mourad Khayati (eXascale Infolab)  

## Overview

Missing data is a well-known problem in neuroscience time series such as EEG. Electrode displacement, movement or electrical interference create gaps in which the underlying signal cannot be recovered. Such gaps can distort waveforms and the functional connectivity derived from them. Ultimately they also affect clinical decisions such as epilepsy diagnosis or seizure monitoring.

Since the channels of a multi-channel recording are not independent but reflect spatial, anatomical or functional connectivity, graph neural networks (GNNs) are generally considered a good fit for this issue. Existing GNN-based imputation methods, however, were often designed for traffic and environmental sensor data, and it is unclear how well they transfer to neuroscience recordings, that differ more strongly in sampling frequency,sequence length and number of channels.

This project builds on **ImputeGAP**, an open-source library for time series imputation developed at the eXascale Infolab (University of Fribourg). 

The aim of this thesis is a benchmark of graph-construction strategies combined with GNN imputation architectures across neuroscience datasets with different characteristics. The outcome is a performance matrix per dataset category and a recommendation layer that suggests the most suitable graph-construction–GNN combination for a new dataset.

## Research questions

1. Which combination of graph-construction strategy and GNN architecture is best suited to impute missing data in neuroscience time series, and how does this depend on the characteristics of the dataset?

2. How do different combinations of graph-construction strategies and GNN architectures compare in imputation accuracy across neuroscience datasets?

3. How do dataset characteristics (sampling frequency, sequence length, number of channels) and missingness characteristics(rate, gap length, pattern, affected channels) affect the imputation accuracy?

4. Is a higher downstream accuracy naturally reflected in a better downstream classification performance and preservation of functional connectivity structure?
