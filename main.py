# %%
import json
import threading
import multiprocessing
import os
import time
import numpy as np
import torch
import tensorflow as tf
from config import config
from vae_pu_occ.vae_pu_occ_trainer import VaePuOccTrainer
from data_loading.vae_pu_dataloaders import create_vae_pu_adapter, get_dataset
from external.LBE import train_LBE, eval_LBE
from external.sar_experiment import SAREMThreadedExperiment

# label_frequencies = [0.7, 0.5, 0.3, 0.1, 0.02]
label_frequencies = [0.5]
# label_frequencies = [0.7, 0.5, 0.3, 0.02]
# label_frequencies = [0.3, 0.7]
# label_frequencies = [0.02, 0.5]

start_idx = 666
num_experiments = 1
epoch_multiplier = 1

config['data'] = 'MNIST 3v5'
# config['data'] = 'CIFAR CarTruck'
# config['data'] = 'STL MachineAnimal'
# config['data'] = 'Gas Concentrations'
# config['data'] = 'MNIST OvE'
# config['data'] = 'CIFAR MachineAnimal'

# config['data'] = 'STL MachineAnimal SCAR'

if 'SCAR' in config['data']:
    config['use_SCAR'] = True
else:
    config['use_SCAR'] = False

config['occ_methods'] = ['OC-SVM', 'IsolationForest', 'ECODv2', 'A^3']

config['use_original_paper_code'] = False
# config['use_original_paper_code'] = True
config['use_old_models'] = True
# config['use_old_models'] = False

config['training_mode'] = 'VAE-PU'
# config['training_mode'] = 'SAR-EM'
# config['training_mode'] = 'LBE'


config['train_occ'] = True
config['occ_num_epoch'] = round(100 * epoch_multiplier)

config['early_stopping'] = True
config['early_stopping_epochs'] = 10

if config['use_original_paper_code']:
    config['mode'] = 'near_o'
else:
    config['mode'] = 'near_y'

config['device'] = 'auto'

# used by SAR-EM
n_threads = multiprocessing.cpu_count()
sem = threading.Semaphore(n_threads)
threads = []

for idx in range(start_idx, start_idx + num_experiments):
    for base_label_frequency in label_frequencies:
        config['base_label_frequency'] = base_label_frequency

        np.random.seed(idx)
        torch.manual_seed(idx)
        tf.random.set_seed(idx)

        if config['device'] == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_samples, val_samples, test_samples, label_frequency, pi_p, n_input = \
            get_dataset(config['data'], device, base_label_frequency, use_scar_labeling=config['use_SCAR'])
        vae_pu_data = \
            create_vae_pu_adapter(train_samples, val_samples, test_samples, device)

        config['label_frequency'] = label_frequency
        config['pi_p'] = pi_p
        config['n_input'] = n_input

        config['pi_pl'] = label_frequency * pi_p
        config['pi_pu'] = pi_p - config['pi_pl']
        config['pi_u'] = 1 - config['pi_pl']

        batch_size = 1000
        pl_batch_size = int(np.ceil(config['pi_pl'] * batch_size))
        u_batch_size = batch_size - pl_batch_size
        config['batch_size_l'], config['batch_size_u'] = (pl_batch_size, u_batch_size)
        config['batch_size_l_pn'], config['batch_size_u_pn'] = (pl_batch_size, u_batch_size)

        config['n_h_y'] = 10
        config['n_h_o'] = 2
        config['lr_pu'] = 3e-4
        config['lr_pn'] = 1e-5

        config['num_epoch_pre'] = round(100 * epoch_multiplier)
        config['num_epoch_step1'] = round(400 * epoch_multiplier)
        config['num_epoch_step_pn1'] = round(500 * epoch_multiplier)
        config['num_epoch_step_pn2'] = round(600 * epoch_multiplier)
        config['num_epoch_step2'] = round(500 * epoch_multiplier)
        config['num_epoch_step3'] = round(700 * epoch_multiplier)
        config['num_epoch'] = round(800 * epoch_multiplier)

        config['n_hidden_cl'] = []
        config['n_hidden_pn'] = [300, 300, 300, 300]


        if config['data'] == 'MNIST OvE':
            config['alpha_gen'] = 0.1
            config['alpha_disc'] = 0.1
            config['alpha_gen2'] = 3
            config['alpha_disc2'] = 3
        elif ('CIFAR' in config['data'] or 'STL' in config['data']) and config['use_SCAR']:
            config['alpha_gen'] = 3
            config['alpha_disc'] = 3
            config['alpha_gen2'] = 1
            config['alpha_disc2'] = 1
            ### What is it?
            config['alpha_test'] = 1.
        elif ('CIFAR' in config['data'] or 'STL' in config['data']) and not config['use_SCAR']:
            config['alpha_gen'] = 0.3
            config['alpha_disc'] = 0.3
            config['alpha_gen2'] = 1
            config['alpha_disc2'] = 1
            ### What is it?
            config['alpha_test'] = 1.
        else:
            config['alpha_gen'] = 1
            config['alpha_disc'] = 1
            config['alpha_gen2'] = 10
            config['alpha_disc2'] = 10

        config['device'] = device
        config['directory'] = os.path.join('result', config['data'], str(base_label_frequency), 'Exp' + str(idx))
        
        if config['training_mode'] == 'VAE-PU':
            trainer = VaePuOccTrainer(num_exp=idx, model_config=config, pretrain=True)
            trainer.train(vae_pu_data)
        else:
            np.random.seed(idx)
            torch.manual_seed(idx)
            tf.random.set_seed(idx)
            method_dir = os.path.join(config['directory'], 'external', config['training_mode'])

            if config['training_mode'] == 'SAR-EM':
                exp_thread = SAREMThreadedExperiment(train_samples, test_samples, idx, base_label_frequency, config, method_dir, sem)
                exp_thread.start()
                threads.append(exp_thread)
            if config['training_mode'] == 'LBE':
                log_prefix = f'Exp {idx}, c: {base_label_frequency} || '

                lbe_training_start = time.perf_counter()
                lbe = train_LBE(train_samples, val_samples, verbose=True, log_prefix=log_prefix)
                lbe_training_time = time.perf_counter() - lbe_training_start

                accuracy, precision, recall, f1 = eval_LBE(lbe, test_samples, verbose=True, log_prefix=log_prefix)
                metric_values = {
                    'Method': 'LBE',
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 score': f1,
                    'Time': lbe_training_time,
                }
                
                os.makedirs(method_dir, exist_ok=True)
                with open(os.path.join(method_dir, 'metric_values.json'), 'w') as f:
                    json.dump(metric_values, f)

for t in threads:
    t.join()

# %%
config['directory'] = './result/'+ config['data'] + '/Exp' + str(idx) + '/'

model = torch.load(config['directory'] + 'model.pt')
# model = torch.load(config['directory'] + 'model_pre_occ.pt')
model.model_en.eval()
model.model_de.eval()
model.model_disc.eval()
model.model_cl.eval()
model.model_pn.eval()

# %%
