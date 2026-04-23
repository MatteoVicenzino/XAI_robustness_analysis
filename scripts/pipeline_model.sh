set -e

VENV_PATH="./.venv"
source "$VENV_PATH/bin/activate"


########## complete pipeline run for dataset adult and model1r:
# Not parallelized yet!! Around 2h run!!!
# run from XAI_robustness_analysis
# chmod +x ./scripts/pipeline_model.sh
# ./scripts/pipeline_model.sh


# MEDOID VALIDATION

python3 neighbourhood_attr.py --dataset adult --model_name model3r --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model3r --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model3r --type validation --agg mean --neigh medoid

# RANDOM VALIDATION

python3 neighbourhood_attr.py --dataset adult --model_name model3r --type validation --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model3r --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model3r --type validation --agg mean --neigh random

# MEDOID TEST

python3 neighbourhood_attr.py --dataset adult --model_name model3r --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model3r --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model3r --type test --agg mean --neigh medoid

# RANDOM TEST

python3 neighbourhood_attr.py --dataset adult --model_name model3r --type test --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model3r --type test --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model3r --type test --agg mean --neigh random

deactivate