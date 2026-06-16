# Differential Privacy & Zero-Knowledge Proofs

This document will contain the formal mathematical definitions and proofs for the DP-SGD noise addition and the ZKP circuits used for update verification.

## Differential Privacy via Opacus
FusionNet uses the PyTorch `opacus` library to guarantee mathematical correctness of the Differential Privacy implementation. `opacus` provides an automated mechanism to wrap the optimizer and dataloader, ensuring that per-sample gradient norms are strictly clipped to $C$ and that calibrated Gaussian noise $N(0, \sigma^2)$ is added to the averaged gradients before taking a training step. The `PrivacyEngine` also provides a rigorous tracking mechanism to calculate the total $\epsilon$ privacy budget expended over $E$ epochs for a given target $\delta$.
