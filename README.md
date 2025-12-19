# Neighbor Relation-Guided Transformer (NRGTrans)
Multivariate Time Series Anomaly Detection Using Neighbor Relation-Guided Transformer

Deep learning (DL)-based unsupervised methods have been widely applied in multivariate time series anomaly detection. However, existing DL-based unsupervised methods do not utilize the neighbor associations within time series data. To address these issues, we propose a Neighbor Relation-Guided Transformer (NRGTrans) for MTSAD.

- We propose a neighbor relation-guided Transformer, named NRGTrans, for multivariate time series anomaly detection. 
- We introduce the neighbor attention-guided module to implement neighbor relation guidance, which consists of two channels: one channel learns global associations, while the other learns neighbor associations.
- We design the neighbor attention-guided loss (NAGLoss) specifically for NRGTrans, which effectively enhances the learning of neighbor associations and significantly improves anomaly detection performance.



## Get Started

1. Install Python 3.6, PyTorch >= 1.4.0.
2. Data availability. All datasets used in this paper are publicly available and can be downloaded from their respective sources.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/SWaT.sh
bash ./scripts/NIPS_TS_Swan.sh
bash ./scripts/NIPS_TS_Water.sh
```

## Acknowledgement

We acknowledge the following GitHub repositories for their valuable open-source code:

https://github.com/thuml/Anomaly-Transformer

https://github.com/DAMO-DI-ML/KDD2023-DCdetector