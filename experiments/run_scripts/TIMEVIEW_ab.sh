# This script runs TIMEVIEW model on additional datasets for Antonin Berthon's challenge
# This data is needed to view the dashboard of each dataset

# Get the argument --debug to run the experiments in debug mode. This will run the experiments with a smaller number of tune iterations.
# Example: bash TIMEVIEW_interface_only.sh --debug
if [[ " $@ " =~ " --debug " ]]; then
    n_tune=1
else
    n_tune=100
fi

#python benchmark.py --datasets sine_trans_200_20 beta_900_20 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device cpu --n_basis 5 --rnn_type lstm
#python benchmark.py --datasets synthetic_tumor_wilkerson_1 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device cpu --n_basis 9 --rnn_type lstm
#python benchmark.py --datasets stress-strain-lot-max-0.2 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device cpu --n_basis 9 --rnn_type lstm
#python benchmark.py --datasets flchain_1000 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device cpu --n_basis 9 --rnn_type lstm
#python benchmark.py --datasets airfoil_log --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device cpu --n_basis 9 --rnn_type lstm

python benchmark.py --datasets beta-sin_3000_20 --baselines TTS --n_trials 10 --n_tune $n_tune --seed 0 --device cpu --n_basis 5 --rnn_type lstm

