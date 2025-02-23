from train_gpt2 import main

override_args = {
    'do_profile': True,
    'num_iterations': 10,
    'disable_wandb': True,
}
val_loss, training_time_ms = main(
    hparam_overrides=override_args,
    model_overrides=None,
)
print('done')
