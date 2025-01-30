# Coarse-grain-autoregulatory-gene-expression-network

Coarse grain model for autoregulator gene expression network. The method identifies the change of mean and variance of marginal protein distribution at different mRNA value, and do interpolation to identify condintinoal marginal protein distribution at inactive and inactive gene state based on overall marginal distribution. Feed it back to gene state distribution, then solve for mRNA distribution consequently
The following figure shows the comparison of the coarse-grain model output vs the exact output from cme:
![alt text](https://github.com/junhag2/CoarseAutoregulatory/blob/main/approximate_exact.png?raw=true)
