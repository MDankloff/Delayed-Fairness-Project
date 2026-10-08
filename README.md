# Long-term Effects of Fairness Metrics on Population Dynamics: A Multi-agent Simulation Study
This repository contains the code for the paper "Long-term Effects of Fairness Metrics on Population Dynamics: A Multi-agent Simulation Study."

# Abstract
Algorithmic fairness is often evaluated as a static property, overlooking that populations evolve over time and that individuals may disengage from systems they perceive as unfair. 
We introduce a dynamic framework of perceived fairness that models how repeated unjust denials and peer-observed outcomes influence individuals’ continued participation. 
We operationalise this framework in a multi-agent simulation where agents interact through social networks, respond to classifier decisions, and may opt out based on perceived unfairness. 
We evaluate static and long-term fairness metrics under population dynamics across varying fraud-rate disparities and social network structures, using a synthetic lending dataset and three datasets from the Bank Account Fraud (BAF)
detection suite. Our results show that: (1) perceived unfairness can induce survivorship bias, reducing measured long-term unfairness as disadvantaged groups disproportionately opt out; 
(2) selective opt-out of disadvantaged groups increases with fraud-rate disparities; (3) the informativeness of long-term fairness metrics depends on the source and magnitude of bias; 
(4) retention disparity between groups is largely insensitive to network structure. 
These findings demonstrate that fairness interventions overlooking individual perceived fairness and population dynamics may inadvertently exclude vulnerable populations from essential services.

# Repository content 

The repository is structured as follows:

* Synthetic/src/

  simulator.py   # simulation loop: decisions, opt-out, network, feature dynamics

  baselines.py   # lender policies: logistic regression, DP, EO, (cvxpy)

  fair_model.py   # long-term counterfactual fairness (LCF) model

  evaluation.py   # metrics: retention rate, disparity, long-term unfairness, accuracy

  direct_fairness.py   # direct counterfactual-fairness estimation used for evaluation

  utils.py   # helpers

  graph_sensitivity.py   #  graph sensitivity sweeps


* Synthetic_final_results.ipynb      # RQ1: synthetic setting

* BAF_refactor_final.ipynb         # RQ2: BAF datasets, varying fraud rate disparities

* BAF_graph_sensitivity.ipynb     # RQ3: network-structure sensitivity

All results are averaged over 20 random seeds

# Data
* Synthetic. Generated in-code; no download needed.

* Bank account fraud (BAF). download from (Sergio Jesus, Jose Pombal, Duarte Alves, Andre Ferreira, Pedro Saleiro, Rita Ribeiro, Joao Gama, and Pedro Bizarro. 2022. Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track).
