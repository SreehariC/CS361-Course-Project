# Structure
- The structure of the folders are as follows
- The name contains the number of GMM components
- Inside the folder, it contains the notebook in which results are present and the name of the models which can be inferred as
  - `gmm{type}_{num_comp}_{num_pca}_{lang}.pkl`
    Where `type` : `diag` for diagonal and `full` for full covariance matrix
          `num_comp` : Number of Gaussian components
          `num_pca`  : Number of Principal compnents considered
          `lang`     : Gujarati - 1, Tamil - 2, Telugu - 3
