set -e

VENV_PATH="./.venv"
source "$VENV_PATH/bin/activate"

# Not parallelized yet!! Around 6h run!!!


# ADULT
python3 new_net_training.py --dataset adult --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset adult --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# BANK
python3 new_net_training.py --dataset bank --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset bank --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# BEANS
python3 new_net_training.py --dataset beans --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset beans --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset beans --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset beans --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset beans --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset beans --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# CANCER
python3 new_net_training.py --dataset cancer --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset cancer --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset cancer --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset cancer --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset cancer --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset cancer --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# HELOC
python3 new_net_training.py --dataset heloc --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset heloc --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset heloc --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset heloc --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset heloc --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset heloc --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# MUSHROOM
python3 new_net_training.py --dataset mushroom --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset mushroom --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset mushroom --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset mushroom --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset mushroom --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset mushroom --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# OCEAN
python3 new_net_training.py --dataset ocean --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset ocean --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset ocean --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset ocean --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset ocean --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset ocean --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


# WINE
python3 new_net_training.py --dataset wine --model_type smallNN --model_name model1 --new False --validation True --history True --results True
python3 new_net_training.py --dataset wine --model_type deeperNN --model_name model2 --new False --validation True --history True --results True
python3 new_net_training.py --dataset wine --model_type shallowNN --model_name model3 --new False --validation True --history True --results True

python3 new_net_training.py --dataset wine --model_type regSmallNN --model_name model1r --new False --validation True --history True --results True
python3 new_net_training.py --dataset wine --model_type regDeeperNN --model_name model2r --new False --validation True --history True --results True
python3 new_net_training.py --dataset wine --model_type regShallowNN --model_name model3r --new False --validation True --history True --results True


deactivate