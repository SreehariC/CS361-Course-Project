# Language Prediciton using GMM

- In terminal , where this code is downloaded, type
```
pip install -r requirements.txt
```
- Run `gmm-final.ipynb` that has the code for training the gmms on the dataset and prints metrics like `Time Taken`, `AIC` , `BIC` , `Accuracy`, `F1 Score` and also has comparision with `sklearn GMM` along with different PCA components
- Please ignore the accuracy in `gmm-final.ipynb` as it was trained on 5 training examples only.
- The final results along with the model pickle files are present in `GMM-models` folder. The folder name denotes the number of GMM components used. Refer to the output of notebook for more details.
- The `gmm-ensembling.ipynb` contains the ensembling approach we tried in which we tried to combine the predictions of several weak models to make a stronger accurate model, but didnt give sufficient results.