"""
ml/
---
CarmPom machine learning module.

Pipeline:
  1. kaggle_loader.py  — load Kaggle NCAA competition CSVs into DataFrames
  2. features.py       — build per-matchup feature matrix for training / prediction
  3. train.py          — compare models, track with MLflow, select best
  4. predict.py        — generate bracket win probabilities from trained model

Data required (place in data/kaggle/):
  MTeams.csv
  MNCAATourneyCompactResults.csv
  MNCAATourneySeeds.csv
  MMasseyOrdinals.csv          (has KenPom 'POM' ordinal ranks)
  MRegularSeasonCompactResults.csv

Source: https://www.kaggle.com/competitions/march-machine-learning-mania-2026
"""
