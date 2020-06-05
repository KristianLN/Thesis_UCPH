# Thesis_UCPH
This repository contain all work related to our thesis at University of Copenhagen.

The current outline is as follows:

Reference paper: (Kercheval and Zhang, 2015)

Goal:

We want to build on the research conducted by (Kercheval and Zhang, 2015), by considering alternative variations of logistic regression and alternative models for the task of predicting the probability of observing a positive price sign in period i+j,P(yi+j=1| Xi ), withj =0,1,5,10 as an example.
Furthermore, we want to quantify the effect of using bootstrap versus cross-validation in the training phase of the model building process. 
Lastly, we want to touch on online prediction in the context of price direction prediction using LOB data.

Summary:

* We want to build on the research conducted by (Kercheval and Zhang, 2015).
* We want to consider alternative variations of Logistic Regression and alternative models (Random Forest and/or Neural Networks).
* We want to discuss/quantify the effect of using bootstrap versus cross-validation in training/estimation phase on LOB data and more generally on time series data.
* We want to consider the use of online prediction in the context of price direction prediction using LOB data.

Structure:

* Introduction
* Literature Review
* Data
  * Market microstructure
  * LOB’s
  * The dataset
    * Descriptive Statistics (e.g. Number of observations, empirical autocorrelation, average number of orders, average number of different signs)
* Models
  * Logistic Regression
    * Linear models (Examples: LASSO, Ridge, Sparse Multinomial (Krishnapuram et al., 2005), Additive LR (LogitBoost) (Friedman et al., 2000))
    * Nonlinear models (Neural Network)
  * Random Forest as an alternative.
  * Bootstrapping vs. Cross-validation for time series data
  * Online Prediction | (Wang, 2019) and references within.
* Results
* Discussion
* Future Work
* Conclusion
* Bibliography

Friedman et al., 2000. Additive Logistic Regression: A Statistical View of Boosting. The Annals of Statistics, Vol. 28, No. 2, 337-407. URL: https://projecteuclid.org/download/pdf_1/euclid.aos/1016218223

Krishnapuram et al., 2005. Sparse Multinomial Logistic Regression: Fast Algorithms and Generalisation Bounds. In IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 27, no. 6, pp. 957-968, June 2005. URL: https://ieeexplore.ieee.org/abstract/document/1424458#deqn2

Kercheval and Zhang, 2015. Modelling High-Frequency Limit Order Book Dynamics With Support Vector Machines. Quantitative Finance, Vol. 15, Issue 8: Special Issue on High Frequency Data Modelling in Finance, Pages 1315-1329. URL: https://www.tandfonline.com/doi/full/10.1080/14697688.2015.1032546

Wang, Dr. Jun, 2019. Computational Advertising: With Applications to Online Auctions. University College London Lecture Notes, Multi-Agent Artificial Intelligence. URL: https://drive.google.com/open?id=1mUULla52-MMejOnRVzQUZVchIOSgEhBM
