set -e

VENV_PATH="./.venv"
source "$VENV_PATH/bin/activate"

########## run everything
# run from XAI_robustness_analysis
# chmod +x ./scripts/al2.sh
# ./scripts/al2.sh

# ADULT
python3 new_net_training.py --dataset adult --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset adult --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset adult --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# BANK
python3 new_net_training.py --dataset bank --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset bank --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset bank --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# BEANS
python3 new_net_training.py --dataset beans --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset beans --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset beans --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# CANCER
python3 new_net_training.py --dataset cancer --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset cancer --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset cancer --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# HELOC
python3 new_net_training.py --dataset heloc --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset heloc --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset heloc --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# MUSHROOM
python3 new_net_training.py --dataset mushroom --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset mushroom --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset mushroom --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# OCEAN
python3 new_net_training.py --dataset ocean --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset ocean --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset ocean --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

# WINE
python3 new_net_training.py --dataset wine --model_type doSmallNN --model_name model1do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset wine --model_type doDeeperNN --model_name model2do2 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset wine --model_type doShallowNN --model_name model3do2 --new False --validation True --history True --results True

(

# ADULT
python3 neighbourhood_attr.py --dataset adult --model_name model1do2 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset adult --model_name model1do2 --type validation --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset adult --model_name model1do2 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset adult --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset adult --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset adult --model_name model1do2 --type test --random --alpha 0.05 --alpha_cat 0.05
python3 aggregation.py --dataset adult --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset adult --model_name model1do2 --type test --agg mean --neigh random

# BANK
python3 neighbourhood_attr.py --dataset bank --model_name model1do2 --type validation --alpha 0.05 --alpha_cat 0.1 --k 5
python3 aggregation.py --dataset bank --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset bank --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset bank --model_name model1do2 --type validation --random --alpha 0.05 --alpha_cat 0.1
python3 aggregation.py --dataset bank --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset bank --model_name model1do2 --type test --alpha 0.05 --alpha_cat 0.1 --k 5
python3 aggregation.py --dataset bank --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset bank --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset bank --model_name model1do2 --type test --random --alpha 0.05 --alpha_cat 0.1
python3 aggregation.py --dataset bank --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset bank --model_name model1do2 --type test --agg mean --neigh random

# BEANS
python3 neighbourhood_attr.py --dataset beans --model_name model1do2 --type validation --alpha 0.1 --k 10
python3 aggregation.py --dataset beans --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset beans --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset beans --model_name model1do2 --type validation --random --alpha 0.02
python3 aggregation.py --dataset beans --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset beans --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset beans --model_name model1do2 --type test --alpha 0.1 --k 10
python3 aggregation.py --dataset beans --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset beans --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset beans --model_name model1do2 --type test --random --alpha 0.02
python3 aggregation.py --dataset beans --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset beans --model_name model1do2 --type test --agg mean --neigh random

# CANCER
python3 neighbourhood_attr.py --dataset cancer --model_name model1do2 --type validation --alpha 0.1 --k 4
python3 aggregation.py --dataset cancer --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset cancer --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset cancer --model_name model1do2 --type validation --random --alpha 0.1
python3 aggregation.py --dataset cancer --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset cancer --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset cancer --model_name model1do2 --type test --alpha 0.1 --k 4
python3 aggregation.py --dataset cancer --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset cancer --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset cancer --model_name model1do2 --type test --random --alpha 0.1
python3 aggregation.py --dataset cancer --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset cancer --model_name model1do2 --type test --agg mean --neigh random

# HELOC
python3 neighbourhood_attr.py --dataset heloc --model_name model1do2 --type validation --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset heloc --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset heloc --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset heloc --model_name model1do2 --type validation --random --alpha 0.03 --alpha_cat 0.1
python3 aggregation.py --dataset heloc --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset heloc --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset heloc --model_name model1do2 --type test --alpha 0.05 --alpha_cat 0.05 --k 5
python3 aggregation.py --dataset heloc --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset heloc --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset heloc --model_name model1do2 --type test --random --alpha 0.03 --alpha_cat 0.1
python3 aggregation.py --dataset heloc --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset heloc --model_name model1do2 --type test --agg mean --neigh random

# MUSHROOM
python3 neighbourhood_attr.py --dataset mushroom --model_name model1do2 --type validation --alpha_cat 0.15 --k 10
python3 aggregation.py --dataset mushroom --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset mushroom --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset mushroom --model_name model1do2 --type validation --random --alpha_cat 0.15
python3 aggregation.py --dataset mushroom --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset mushroom --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset mushroom --model_name model1do2 --type test --alpha_cat 0.15 --k 10
python3 aggregation.py --dataset mushroom --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset mushroom --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset mushroom --model_name model1do2 --type test --random --alpha_cat 0.15
python3 aggregation.py --dataset mushroom --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset mushroom --model_name model1do2 --type test --agg mean --neigh random

# WINE
python3 neighbourhood_attr.py --dataset wine --model_name model1do2 --type validation --alpha 0.15 --k 5
python3 aggregation.py --dataset wine --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset wine --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset wine --model_name model1do2 --type validation --random --alpha 0.03
python3 aggregation.py --dataset wine --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset wine --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset wine --model_name model1do2 --type test --alpha 0.15 --k 5
python3 aggregation.py --dataset wine --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset wine --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset wine --model_name model1do2 --type test --random --alpha 0.03
python3 aggregation.py --dataset wine --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset wine --model_name model1do2 --type test --agg mean --neigh random


) & PID1=$!

(

# OCEAN
python3 neighbourhood_attr.py --dataset ocean --model_name model1do2 --type validation --alpha 0.05 --k 5
python3 aggregation.py --dataset ocean --model_name model1do2 --type validation --agg ensemble --neigh medoid
python3 aggregation.py --dataset ocean --model_name model1do2 --type validation --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset ocean --model_name model1do2 --type validation --random --alpha 0.001
python3 aggregation.py --dataset ocean --model_name model1do2 --type validation --agg ensemble --neigh random
python3 aggregation.py --dataset ocean --model_name model1do2 --type validation --agg mean --neigh random
python3 neighbourhood_attr.py --dataset ocean --model_name model1do2 --type test --alpha 0.05 --k 5
python3 aggregation.py --dataset ocean --model_name model1do2 --type test --agg ensemble --neigh medoid
python3 aggregation.py --dataset ocean --model_name model1do2 --type test --agg mean --neigh medoid
python3 neighbourhood_attr.py --dataset ocean --model_name model1do2 --type test --random --alpha 0.001
python3 aggregation.py --dataset ocean --model_name model1do2 --type test --agg ensemble --neigh random
python3 aggregation.py --dataset ocean --model_name model1do2 --type test --agg mean --neigh random

) & PID2=$!

wait $PID1
wait $PID2

deactivate