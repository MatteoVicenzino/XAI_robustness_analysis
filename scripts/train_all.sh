set -e

VENV_PATH="./../venv"
source "$VENV_PATH/bin/activate"

# ADULT
#python3 new_net_training.py --dataset adult --model_type regularizedNN --model_name model8 --new False --validation True --history True --results True
python3 new_net_training.py --dataset adult --model_type CNN2 --model_name model7 --new False --validation True --history True --results True

# BANK
#python3 new_net_training.py --dataset bank --model_type regularizedNN --model_name model8 --new False --validation True --history True --results True
python3 new_net_training.py --dataset bank --model_type CNN2 --model_name model7 --new False --validation True --history True --results True

#python3 new_net_training.py --dataset cancer --model_type regularizedNN --model_name model8 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset cancer --model_type CNN2 --model_name model7 --new False --validation True --history True --results True

#python3 new_net_training.py --dataset heloc --model_type regularizedNN --model_name model8 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset heloc --model_type CNN2 --model_name model7 --new False --validation True --history True --results True

#python3 new_net_training.py --dataset mushroom --model_type regularizedNN --model_name model8 --new False --validation True --history True --results True
#python3 new_net_training.py --dataset mushroom --model_type CNN2 --model_name model7 --new False --validation True --history True --results True

# BEANS
#python3 new_net_training.py --dataset beans --model_type CNN3 --model_name model10 --new False --validation True --history True --results True

# CANCER
#python3 new_net_training.py --dataset cancer --model_type CNN3 --model_name model10 --new False --validation True --history True --results True

# HELOC
#python3 new_net_training.py --dataset heloc --model_type CNN3 --model_name model10 --new False --validation True --history True --results True

# MUSHROOM
#python3 new_net_training.py --dataset mushroom --model_type CNN3 --model_name model10 --new False --validation True --history True --results True

# OCEAN
#python3 new_net_training.py --dataset ocean --model_type CNN3 --model_name model10 --new False --validation True --history True --results True

# WINE
#python3 new_net_training.py --dataset wine --model_type CNN3 --model_name model10 --new False --validation True --history True --results True


deactivate