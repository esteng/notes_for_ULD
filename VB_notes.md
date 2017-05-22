# VB Code

## VB-DPHMM
### code
- repo : `https://github.com/iondel/amdtk`
- VB training done in script `amdtk/recipes/timit/utils/phone_loop_train.sh`
- strong improvement of using VB compare to Gibbs Sampling
- difference from initialization, not training
- in GS, initial partion of segments using Chinese restaurant process without resetting customers during the first training epoch
- in VB, random initialization of Gaussian in HMM (mean sampled from entire dataset) 
- VB also faster
