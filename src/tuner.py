import argparse
import logging
import sys
import traceback

import optuna

from train_gpt2 import main

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))


parser = argparse.ArgumentParser(description='Tune a GPT model.')
parser.add_argument('-s', '--study_name', type=str, default='default-study', help='Unique identifier of the Optuna study')
parser.add_argument('--load-if-exists', action='store_true', default=False, help='Load existing study if it exists')
parser.add_argument('-n', '--n-trials', type=int, default=100, help='Number of trials to run')
args = parser.parse_args()
print(f'Tuner args: {args}')

storage_name = 'sqlite:///{}.db'.format(args.study_name)
study = optuna.create_study(
    direction='minimize',  # minimize both val loss and train time
    study_name=args.study_name,
    storage=storage_name,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(n_startup_trials=2),
    pruner=optuna.pruners.HyperbandPruner(min_resource=512, max_resource=6312),
)

study.add_trial(
    optuna.trial.create_trial(
        params={
            'learning_rate': 0.00036,
            'emb_learning_rate': 0.0036,
            # 'num_iterations': 6312,
        },
        distributions={
            'learning_rate': optuna.distributions.FloatDistribution(0.00025, 0.0004),
            'emb_learning_rate': optuna.distributions.FloatDistribution(0.0025, 0.004),
            # 'num_iterations': optuna.distributions.IntDistribution(5200, 6312),
        },
        value=3.2801,
        intermediate_values={0: 11.01, 128: 5.23088, 256: 4.5845, 384: 4.2225, 512: 4.01784, 640: 3.9076, 768: 3.84132, 896: 3.78286, 1024: 3.73806, 1152: 3.70246, 1280: 3.67281},
    )
)

for i in range(args.n_trials):
    print(f'Starting trial {i}')
    trial = study.ask()

    override_args = {
        'learning_rate': trial.suggest_float('learning_rate', 0.00025, 0.0004),
        'emb_learning_rate': trial.suggest_float('emb_learning_rate', 0.0025, 0.004),
        # 'num_iterations': trial.suggest_int('num_iterations', 5200, 6312),
        'disable_wandb': True,
    }
    try:
        val_loss, training_time_ms = main(hparam_overrides=override_args, model_overrides=None, trial=trial)

        # if val_loss > SPEEDRUN_TARGET:
        #     training_time_ms = float('inf')  # did not achieve the goal (bad: sparse reward)

        study.tell(trial, values=val_loss)
    except optuna.exceptions.TrialPruned:
        print('Trial was pruned!')
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    except Exception as ex:
        print(f'Exception: {ex}')
        print('Stacktrace:')
        print(traceback.format_exc())
        study.tell(trial, state=optuna.trial.TrialState.FAIL)


print('Best trial:')
trial = study.best_trial
print(f'  Value: val_loss={trial.value:.4f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print()
