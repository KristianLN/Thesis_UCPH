# Forecasting Price Directions Using High Frequency Stock Data

This repository contain all work related to our thesis at University of Copenhagen.

The current outline is as follows:

Reference paper: TBD

Goal:

The overall goal is to evaluate different models ability to predict the future price direction using high frequency price data, on potentially all American stocks.
An important feature of our research is that it should be practically relevant, by which we measure the degree of relevance by implementing our research as a trading strategy.
We address the classical pitfalls arising when only considering the basic probabilities of the future price direction going up, down or stay unchanged, and we implement a more advanced framework.
Our information set, i.e. our features/explanatory variables, consists, for now, of classical technical features based on realized (past) prices. We aim to explore a range of alternative features to improve our models ability to forecast the future price direction, and thereby improve the risk/reward ratio of our trading strategy.
Some of the alternative features we intend to investigate, to uncover the potential, is:

* Greeks - Between the relevant sector ETF and the stock.
* Technical features of the whole market, the related sector or similar stocks (The pool of similar stocks could be divided into size buckets, to control for varying information in size as well.)
* Price derivatives and other price-based features from (Kercheval and Zhang, 2015).

# Data

Our data is from the TAQ database, which contains high frequency data on all American stocks from 19930101-20200430.
We still have to decide on the exact range and the selection of stocks used for our analysis. The size of the data is most definitely a challenge, which we aim to address and tackle, to ensure that your research is practically relevant.

## Data Cleaning

Reference: [(Lunde, 2016)](https://econ.au.dk/fileadmin/site_files/filer_oekonomi/subsites/creates/Diverse_2016/PhD_High-Frequency/HF_TrQuData_v01.pdf)

There are, at least, three other areas we want to explore:

## Cointegration

Can cointegration be used effectively to improve our models?
Ideally, we want to use cointegration dynamically to improve the trading strategy, through the information on the long and short term relationship but it is unclear exactly how at the moment.

## Fitting Procedures (of the Models)

Cross-validation or Bootstrapping? What are the differences and is one better than the other?

## Aggregation Horizon

Is there an information-wise improvement in considering different aggregation horizons?

Reference: [(Brownlees and Gallo, 2006)][https://ieeexplore.ieee.org/abstract/document/1380003]

The state of the research is provided below, by showcasing a history of working plans with deadlines.

# History of Working Plans (Latest to-do first):

* Data Cleaning – based on the slides of Asger Lunde.
  * Implement dynamic aggregation
* Features Engineering + Labels
  * Implement the basic classical features and labels
* Implement models
  * Tensorflow / Scikit
* Check Sector ETFS (Done)
  * They are available in our data.
* Github (Done)
  * It is running!

Deadline: 15 June of 2020
