# When Can You Trust Your Explanations? A Robustness Analysis On Feature Importances

This repository contains the code used for the paper "**When can you trust your explanations? A robustness analysis on feature importances**" (*Vascotto, Rodriguez, Bonaita, Bortolussi*), submitted at the 3rd World Conference on eXplainable Artificial Intelligence.

The script ```neighbourhood_attr.py``` generates the neighbourhoods and the IG, DL and LRP attributions. These are then aggregated via ```aggregation.py``` and the robustness score is computed.
```tuning.ipynb``` should be used to tune the $k$nn regressor and the choice of the threshold $r_{th}$. Chosen values presented in Appendix A are saved in ```parameters.joblib```.

The script ```new_net_training.py``` should be used when a new dataset (either already preprocessed or not) is added for testing and to train the three required nets. All datasets included in the paper are already saved in the ```dataset``` folder and the corresponding model weights are stored in ```models```.
The folders ```results_medoid``` and ```results_random``` contains the robustness scores, ensemble and individual attributions for each dataset-model pair, for both validation and test sets. 
The folder  ```results_roc_auc``` contains the data needed for ROC/AUC analysis, divided by aggregation method and neighbourhood generation. 

Example:
> python neighbourhood_attr.py --dataset adult --model_name model1 --type test --random False --alpha 0.05 --alpha_cat 0.05 --k 5 --num 100

> python aggregation.py --dataset adult --model_name model1 --type test --agg ensemble --neigh medoid