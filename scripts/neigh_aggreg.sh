set -e

VENV_PATH="./../venv"
source "$VENV_PATH/bin/activate"


# model 9 resnet gia runnato per adult e bank
# model 8 dropout 0.1 runnato per adult

################################### MODEL 8 ###################################


# MEDOID VALIDATION

python3 neighbourhood_attr.py --dataset bank --model_name model8 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset bank --model_name model8 --type validation --agg ensemble --neigh medoid 
python3 aggregation.py --dataset bank --model_name model8 --type validation --agg mean --neigh medoid

# RANDOM VALIDATION

python3 neighbourhood_attr.py --dataset bank --model_name model8 --type validation --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset bank --model_name model8 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model8 --type validation --agg mean --neigh random

# MEDOID TEST

python3 neighbourhood_attr.py --dataset bank --model_name model8 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset bank --model_name model8 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset bank --model_name model8 --type test --agg mean --neigh medoid

# RANDOM TEST

python3 neighbourhood_attr.py --dataset bank --model_name model8 --type test --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset bank --model_name model8 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model8 --type test --agg mean --neigh random


################################### MODEL 7 ###################################


# MEDOID VALIDATION

python3 neighbourhood_attr.py --dataset adult --model_name model7 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model7 --type validation --agg ensemble --neigh medoid 
python3 aggregation.py --dataset adult --model_name model7 --type validation --agg mean --neigh medoid

# RANDOM VALIDATION

python3 neighbourhood_attr.py --dataset adult --model_name model7 --type validation --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model7 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model7 --type validation --agg mean --neigh random

# MEDOID TEST

python3 neighbourhood_attr.py --dataset adult --model_name model7 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model7 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model7 --type test --agg mean --neigh medoid

# RANDOM TEST

python3 neighbourhood_attr.py --dataset adult --model_name model7 --type test --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model7 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model7 --type test --agg mean --neigh random



################################### MODEL 8 ###################################



# MEDOID VALIDATION

#python3 neighbourhood_attr.py --dataset bank --model_name model8 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
#python3 aggregation.py --dataset bank --model_name model8 --type validation --agg ensemble --neigh medoid 
#python3 aggregation.py --dataset bank --model_name model8 --type validation --agg mean --neigh medoid

# RANDOM VALIDATION

#python3 neighbourhood_attr.py --dataset bank --model_name model8 --type validation --random --alpha 0.05 --alpha_cat 0.05
#python3 aggregation.py --dataset bank --model_name model8 --type validation --agg ensemble --neigh random
#python3 aggregation.py --dataset bank --model_name model8 --type validation --agg mean --neigh random

# MEDOID TEST

#python3 neighbourhood_attr.py --dataset bank --model_name model8 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
#python3 aggregation.py --dataset bank --model_name model8 --type test --agg ensemble --neigh medoid
#python3 aggregation.py --dataset bank --model_name model8 --type test --agg mean --neigh medoid

# RANDOM TEST

#python3 neighbourhood_attr.py --dataset bank --model_name model8 --type test --random --alpha 0.05 --alpha_cat 0.05
#python3 aggregation.py --dataset bank --model_name model8 --type test --agg ensemble --neigh random
#python3 aggregation.py --dataset bank --model_name model8 --type test --agg mean --neigh random



################################### MODEL 7 ###################################


# MEDOID VALIDATION

python3 neighbourhood_attr.py --dataset bank --model_name model7 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset bank --model_name model7 --type validation --agg ensemble --neigh medoid 
python3 aggregation.py --dataset bank --model_name model7 --type validation --agg mean --neigh medoid

# RANDOM VALIDATION

python3 neighbourhood_attr.py --dataset bank --model_name model7 --type validation --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset bank --model_name model7 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model7 --type validation --agg mean --neigh random

# MEDOID TEST

python3 neighbourhood_attr.py --dataset bank --model_name model7 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset bank --model_name model7 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset bank --model_name model7 --type test --agg mean --neigh medoid

# RANDOM TEST

python3 neighbourhood_attr.py --dataset bank --model_name model7 --type test --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset bank --model_name model7 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model7 --type test --agg mean --neigh random



deactivate