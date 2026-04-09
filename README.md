# The Mini Neural Network Verifier

A mini neural network verifier for educational purposes.
The `mininnverifier` includes a simple machine learning framework à la [Jax](https://jax.readthedocs.io/).
For neural network verification, it implements a basic variant of [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN).

To get started, clone this repository, run

```
pip install -e .[demos]
```

and have fun with the code!

## Acknowledgements

For the mini machine learning framework, this repo is inspired by the awesome [micrograd](https://github.com/karpathy/micrograd/) repository that implements a jet smaller PyTorch-style deep learning framework. [Autodidax](https://docs.jax.dev/en/latest/autodidax.html) from the Jax docs taught me about the core ideas behind Jax. This repo is essentially Autodidax but in the micrograd format instead of a tutorial. I simplified some parts some more and tried to use less jargon.

While the neural network verifier implementation is mostly home-grown, the theory behind the verifier is almost exclusively due to the [alpha-beta-CROWN team](https://github.com/Verified-Intelligence/alpha-beta-CROWN), who have had an exceptional impact on the field of neural network verification.
