# MCGC-SPFFN
Code for Self-Perceptive Feature Fusion Network with Multi-Channel Graph Convolution for Brain Disorder Diagnosis.

![image](https://github.com/user-attachments/assets/afe3fe18-f30b-4708-bbe1-79745a9a2357)

(1) The edges were adaptively constructed by the adaptive edge learning network (AELN), which captures the subtle discrepancies of graphs.

(2) A self-perceptive feature fusion (SPFF) module, which includes an accuracy-weighted voting strategy for fusing features within the same channel, and a multi-head cross-attention mechanism for fusing multi-scale features. 

(3) The channel diversity and scale correlation constraints were implemented to thoroughly investigate the latent relationships among features.
# Requirement
pytorch 2.1.0

python 3.11.5

cuda 12.1

numpy 1.26.2

pandas 2.1.3

scikit-learn 1.3.2
