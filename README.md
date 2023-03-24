# MVTSI

This repository provides the python code for the model used in the paper

> Bülte et. al. (2023).
> Multivariate time series imputation for energy data using neural networks. Energy and AI, Volume 13 (2023) https://doi.org/10.1016/j.egyai.2023.100239
 
## Abstract
Multivariate time series with missing values are common in a wide range of applications, including energy data. Existing imputation methods often fail to focus on the temporal dynamics and the cross-dimensional correlation simultaneously. In this paper we propose a two-step method based on an attention model to impute missing values in multivariate energy time series. First, the underlying distribution of the missing values in the data is learned. This information is then further used to train an attention based imputation model. By learning the distribution prior to the imputation process, the model can respond flexibly to the specific characteristics of the underlying data. The developed model is applied to European energy data, obtained from the European Network of Transmission System Operators for Electricity. Using different evaluation metrics and benchmarks, the conducted experiments show that the proposed model is preferable to the benchmarks and is able to accurately impute missing values.
Keywords: Missing value estimation; Multivariate time series; Neural networks; Attention model; Energy data

 
## Data
The data is not licensed to be published, but can be obtained via the ENTSO-E transparency platform (https://transparency.entsoe.eu/).
An API Access can be made available via request.
