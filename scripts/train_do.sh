set -e

VENV_PATH="./.venv"
source "$VENV_PATH/bin/activate"

# Not parallelized yet!!
# run from XAI_robustness_analysis
# chmod +x ./scripts/train_do.sh
# ./scripts/train_do.sh

# ADULT
python3 new_net_training.py --dataset adult --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# BANK
python3 new_net_training.py --dataset bank --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# BEANS
python3 new_net_training.py --dataset beans --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset beans --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset beans --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# CANCER
python3 new_net_training.py --dataset cancer --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset cancer --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset cancer --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# HELOC
python3 new_net_training.py --dataset heloc --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset heloc --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset heloc --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# MUSHROOM
python3 new_net_training.py --dataset mushroom --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset mushroom --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset mushroom --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# OCEAN
python3 new_net_training.py --dataset ocean --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset ocean --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset ocean --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


# WINE
python3 new_net_training.py --dataset wine --model_type doSmallNN --model_name model1do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset wine --model_type doDeeperNN --model_name model2do1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset wine --model_type doShallowNN --model_name model3do1 --new False --validation True --history True --results True


deactivate