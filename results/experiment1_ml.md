# Experiment 1 - ML threat detection

Data: SYNTHETIC flows (pipeline validation only - not a dataset result)  
Protocol: stratified 75/25 hold-out  
Train/test: 9000/3000 flows

| model | accuracy | precision (macro) | recall (macro) | F1 (macro) | ROC-AUC (malicious) | FPR | FNR | train s | infer ms / 1k |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | 0.8887 | 0.9354 | 0.7227 | 0.7827 | 0.9709 | 0.0056 | 0.2962 | 1.87 | 35.30 |
| hist_gradient_boosting | 0.9397 | 0.9359 | 0.8753 | 0.9017 | 0.9813 | 0.0195 | 0.1219 | 5.18 | 49.65 |
| mlp | 0.8887 | 0.8355 | 0.8265 | 0.8296 | 0.9537 | 0.0718 | 0.1514 | 2.07 | 2.49 |
