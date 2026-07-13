set -e

VENV_PATH="./.venv"
source "$VENV_PATH/bin/activate"


########## complete pipeline run for everything with correct parameters:
# Don't run!!!


# ADULT
python3 neighbourhood_attr.py --dataset adult --model_name model1 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset adult --model_name model1 --type validation --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset adult --model_name model1 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset adult --model_name model1 --type test --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model1 --type test --agg mean --neigh random


# BANK
python3 neighbourhood_attr.py --dataset bank --model_name model1 --type validation --alpha 0.05 --alpha_cat 0.1 --k 5
python3 aggregation.py --dataset bank --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset bank --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset bank --model_name model1 --type validation --random --alpha 0.05 --alpha_cat 0.1
python3 aggregation.py --dataset bank --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset bank --model_name model1 --type test --alpha 0.05 --alpha_cat 0.1 --k 5
python3 aggregation.py --dataset bank --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset bank --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset bank --model_name model1 --type test --random --alpha 0.05 --alpha_cat 0.1
python3 aggregation.py --dataset bank --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model1 --type test --agg mean --neigh random


# BEANS
python3 neighbourhood_attr.py --dataset beans --model_name model1 --type validation --alpha 0.1 --k 10
python3 aggregation.py --dataset beans --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset beans --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset beans --model_name model1 --type validation --random --alpha 0.02
python3 aggregation.py --dataset beans --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset beans --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset beans --model_name model1 --type test --alpha 0.1 --k 10
python3 aggregation.py --dataset beans --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset beans --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset beans --model_name model1 --type test --random --alpha 0.02
python3 aggregation.py --dataset beans --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset beans --model_name model1 --type test --agg mean --neigh random


# CANCER
python3 neighbourhood_attr.py --dataset cancer --model_name model1 --type validation --alpha 0.1 --k 4
python3 aggregation.py --dataset cancer --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset cancer --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset cancer --model_name model1 --type validation --random --alpha 0.1
python3 aggregation.py --dataset cancer --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset cancer --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset cancer --model_name model1 --type test --alpha 0.1 --k 4
python3 aggregation.py --dataset cancer --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset cancer --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset cancer --model_name model1 --type test --random --alpha 0.1
python3 aggregation.py --dataset cancer --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset cancer --model_name model1 --type test --agg mean --neigh random


# HELOC
python3 neighbourhood_attr.py --dataset heloc --model_name model1 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset heloc --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset heloc --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset heloc --model_name model1 --type validation --random --alpha 0.03 --alpha_cat 0.1
python3 aggregation.py --dataset heloc --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset heloc --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset heloc --model_name model1 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset heloc --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset heloc --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset heloc --model_name model1 --type test --random --alpha 0.03 --alpha_cat 0.1
python3 aggregation.py --dataset heloc --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset heloc --model_name model1 --type test --agg mean --neigh random


# MUSHROOM
python3 neighbourhood_attr.py --dataset mushroom --model_name model1 --type validation --alpha_cat 0.15 --k 10
python3 aggregation.py --dataset mushroom --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset mushroom --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset mushroom --model_name model1 --type validation --random --alpha_cat 0.15
python3 aggregation.py --dataset mushroom --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset mushroom --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset mushroom --model_name model1 --type test --alpha_cat 0.15 --k 10
python3 aggregation.py --dataset mushroom --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset mushroom --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset mushroom --model_name model1 --type test --random --alpha_cat 0.15
python3 aggregation.py --dataset mushroom --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset mushroom --model_name model1 --type test --agg mean --neigh random


# OCEAN
python3 neighbourhood_attr.py --dataset ocean --model_name model1 --type validation --alpha 0.05 --k 5
python3 aggregation.py --dataset ocean --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset ocean --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset ocean --model_name model1 --type validation --random --alpha 0.001
python3 aggregation.py --dataset ocean --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset ocean --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset ocean --model_name model1 --type test --alpha 0.05 --k 5
python3 aggregation.py --dataset ocean --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset ocean --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset ocean --model_name model1 --type test --random --alpha 0.001
python3 aggregation.py --dataset ocean --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset ocean --model_name model1 --type test --agg mean --neigh random


# WINE
python3 neighbourhood_attr.py --dataset wine --model_name model1 --type validation --alpha 0.15 --k 5
python3 aggregation.py --dataset wine --model_name model1 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset wine --model_name model1 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset wine --model_name model1 --type validation --random --alpha 0.03
python3 aggregation.py --dataset wine --model_name model1 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset wine --model_name model1 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset wine --model_name model1 --type test --alpha 0.15 --k 5
python3 aggregation.py --dataset wine --model_name model1 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset wine --model_name model1 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset wine --model_name model1 --type test --random --alpha 0.03
python3 aggregation.py --dataset wine --model_name model1 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset wine --model_name model1 --type test --agg mean --neigh random

deactivate