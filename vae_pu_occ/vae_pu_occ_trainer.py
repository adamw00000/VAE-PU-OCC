import numpy as np
import time
import torch
import os
import json
import copy
from .vae_pu_trainer import VaePuTrainer
from sklearn import metrics
import tensorflow as tf

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

from external.cccpv.methods import ConformalPvalues
from external.ecod_v2 import ECODv2
from external.A3_adapter import A3Adapter
from external.pyod_wrapper import PyODWrapper


class VaePuOccTrainer(VaePuTrainer):
    def __init__(self, num_exp, model_config, pretrain=True):
        super(VaePuOccTrainer, self).__init__(num_exp, model_config, pretrain)

    def train(self, vae_pu_data):
        super(VaePuOccTrainer, self).train(vae_pu_data)

        if not self.config['train_occ'] or self.use_original_paper_code:
            return

        self.modelOrig = copy.deepcopy(self.model)
        for occ_method in self.config['occ_methods']:
            print('Starting', occ_method)
            self.occ_training_start = time.perf_counter()

            model = copy.deepcopy(self.modelOrig)
            np.random.seed(self.num_exp)
            torch.manual_seed(self.num_exp)
            tf.random.set_seed(self.num_exp)

            pi_pl, pi_u, pi_pu = self.config['pi_pl'], self.config['pi_u'], self.config['pi_pu']
            pu_to_u_ratio = pi_pu / pi_u

            x_pu_gen = self._generate_x_pu()

            if 'Clustering' not in occ_method:
                self._train_occ(occ_method, x_pu_gen)

            best_epoch = self.config['num_epoch']
            best_f1 = 0
            best_acc = 0
            best_model = copy.deepcopy(model)
            early_stopping_counter = 0

            for epoch in range(self.config['num_epoch'], self.config['num_epoch'] + self.config['occ_num_epoch']):
                start_time = time.time()
                targetLosses = []

                if 'Clustering' in occ_method:
                    true_x_pu, pu_indices = self._select_true_x_pu_clustering(self.x_pl_full, self.x_u_full, pu_to_u_ratio, occ_method)
                else:
                    true_x_pu, pu_indices = self._select_true_x_pu_occ(self.x_u_full, pu_to_u_ratio)
                
                targetClassifierLoss = self.model.train_step_pn_true_x_pu(self.x_pl_full, self.x_u_full, true_x_pu, pi_pl, pi_u, pi_pu)
                targetLosses.append(targetClassifierLoss)

                occ_metrics, val_metrics = self._calculate_occ_metrics(epoch, occ_method, pu_indices, targetLosses)
                val_acc, _, _, val_f1 = val_metrics

                # best f1 early stopping
                if val_f1 >= best_f1:
                    best_f1 = val_f1
                    best_acc = val_acc
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if self.config['early_stopping'] and early_stopping_counter == self.config['early_stopping_epochs']:
                        model = copy.deepcopy(best_model)
                        val_f1 = best_f1
                        val_acc = best_acc
                        occ_metrics, val_metrics = self._calculate_occ_metrics(epoch, occ_method, pu_indices, targetLosses)

                        print(f'Early stopping | Best model epoch: {best_epoch + 1}, f1: {best_f1:.4f}, acc: {best_acc:.4f}')
                        print('')
                        break

            self.occ_training_time = time.perf_counter() - self.occ_training_start
            self._save_vae_pu_occ_metrics(occ_method, occ_metrics, best_epoch)

        # # OCC training end

        if not self.use_original_paper_code:
            self._save_results()
            self._save_final_metrics()

        return model

    def _generate_x_pu(self):
        x_pus = []
                
        for x_pl, full_u in zip(self.DL_pl, self.DL_u_full):
            (x_u, _) = full_u
            x_pl = x_pl[0]

            # generate number of samples >= the whole dataset
            generated_batches = np.ceil(1 / self.config['pi_pl']).astype(int)

            for _ in range(generated_batches):
                if self.use_original_paper_code:
                    _, _, _, x_pu, _ = self.model.generate(x_pl, x_u, self.config['mode'])
                else:
                    # use x_u_full (whole U set) instead of x_u (batch only)
                    _, _, _, x_pu, _ = self.model.generate(x_pl, self.x_u_full, self.config['mode'])
                x_pus.append(x_pu.detach())

        x_pu = torch.cat(x_pus)
        x_pu = x_pu.cpu().numpy()

        return x_pu
    
    def _train_occ(self, occ_method, x_pu_gen):
        # Initialize the one-class classifier
        contamination = 0.001
        contamination = min(max(contamination,0.004),0.1)
        if "OC-SVM" in occ_method:
            occ = OneClassSVM(nu=contamination, kernel="rbf", gamma=0.1)
        elif "IsolationForest" in occ_method:
            occ = IsolationForest(random_state=self.num_exp, contamination=contamination)
        elif "LocalOutlierFactor" in occ_method:
            occ = LocalOutlierFactor(novelty=True, contamination=contamination)
        elif "A^3" in occ_method:
            occ = A3Adapter(target_epochs=10, a3_epochs=10)
        else:
            if "ECOD" in occ_method:
                occ = PyODWrapper(ECOD(contamination=contamination, n_jobs = -2))
            if "ECODv2" in occ_method:
                occ = PyODWrapper(ECODv2(contamination=contamination, n_jobs = -2))
            elif "COPOD" in occ_method:
                occ = PyODWrapper(COPOD(contamination=contamination, n_jobs = -2))

        self.cc = ConformalPvalues(x_pu_gen, occ, calib_size=0.5, random_state=self.num_exp)
        return self.cc

    def _select_true_x_pu_occ(self, x_u, pu_to_u_ratio):
        pvals_one_class = self.cc.predict(x_u.cpu().numpy(), delta=0.05, simes_kden=2)
        pvals = pvals_one_class['Marginal']
        pvals = torch.from_numpy(pvals).to(self.config['device'])

        # Order approach
        sorted_indices = torch.argsort(pvals, descending=True)
        n_pu_samples = round(pu_to_u_ratio * len(x_u))
        pu_indices = sorted_indices[:n_pu_samples]
        true_x_pu = x_u[pu_indices]
        return true_x_pu, pu_indices

    def _select_true_x_pu_clustering(self, x_pl, x_u, pu_to_u_ratio, occ_method):
        if 'QuantMean' in occ_method:
            clustering_metric = 'quantmean'
        elif 'Mean' in occ_method:
            clustering_metric = 'mean'
        elif 'Quantile' in occ_method:
            clustering_metric = 'quantile'
        else:
            raise NotImplementedError()

        best_distances, best_xus, best_pu_indices = \
            torch.empty(0).to(self.config['device']), \
            torch.empty((0, x_u.shape[1])).to(self.config['device']), \
            torch.empty(0, dtype=torch.int).to(self.config['device'])
        n_pu_samples = int(round(pu_to_u_ratio * len(x_u)))
                        
        batch_size = self.config['batch_size_u']

        for batch_start in range(0, len(x_u), batch_size):
            batch_end = min(batch_start + batch_size, len(x_u))

            x_u_batch = x_u[batch_start:batch_end]
            best_xus, best_pu_indices, best_distances = self.model.get_pu_from_clustering_batched(x_pl, x_u_batch, n_pu_samples, clustering_metric, batch_start, best_distances=best_distances, best_xus=best_xus, best_pu_indices=best_pu_indices)
        true_x_pu = best_xus
        pu_indices = best_pu_indices
        return true_x_pu, pu_indices

    def _calculate_occ_metrics(self, epoch, occ_method, pu_indices, epochTargetLosses):
        y_u_pred = torch.zeros_like(self.y_u_full)
        y_u_pred[pu_indices] = 1

        y_u_cpu = self.y_u_full.detach().cpu().numpy()
        y_u_cpu = np.where(y_u_cpu == 1, 1, 0)
        y_u_pred_cpu = y_u_pred.detach().cpu().numpy()

        occ_acc = metrics.accuracy_score(y_u_cpu, y_u_pred_cpu)
        occ_auc = metrics.roc_auc_score(y_u_cpu, y_u_pred_cpu)
        occ_precision = metrics.precision_score(y_u_cpu, y_u_pred_cpu)
        occ_recall = metrics.recall_score(y_u_cpu, y_u_pred_cpu)
        occ_f1 = metrics.f1_score(y_u_cpu, y_u_pred_cpu)
        occ_fdr = np.sum((y_u_pred_cpu == 1) & (y_u_cpu == 0)) / np.sum(y_u_pred_cpu == 1)

        print('OCC ({}) - epoch: {}, loss: {}'.format(occ_method, epoch + 1, np.mean(epochTargetLosses)))
        val_acc, val_pr, val_re, val_f1 = self.model.accuracy(self.DL_val)
        self.valAccuracies.append(val_acc)
        print('...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1 score: {3:.4f}'.format(val_acc, val_pr, val_re, val_f1))

        val_loss = self.model.loss_val(self.x_val[:20], self.x_val[20:])
        self.valLosses.append(val_loss)

        self.targetClassifierLosses.append(np.mean(epochTargetLosses))
        print(f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| PN loss: {np.mean(epochTargetLosses)}')

        print(f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| OCC training time: {(time.perf_counter() - self.occ_training_start):.2f} sec')
        return (occ_acc, occ_auc, occ_precision, occ_recall, occ_f1, occ_fdr), (val_acc, val_pr, val_re, val_f1)

    def _save_vae_pu_occ_metrics(self, occ_method, occ_metrics, best_epoch):
        os.makedirs(os.path.join(self.config['directory'], 'occ', occ_method), exist_ok=True)
        acc, precision, recall, f1_score = self.model.accuracy(self.DL_test)
        occ_acc, occ_auc, occ_precision, occ_recall, occ_f1, occ_fdr = occ_metrics
        metric_values = {
                    'Method': occ_method,
                    'OCC accuracy': occ_acc,
                    'OCC precision': occ_precision,
                    'OCC recall': occ_recall,
                    'OCC F1 score': occ_f1,
                    'OCC AUC': occ_auc,
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 score': f1_score,
                    'Best epoch': best_epoch + 1,
                }
        if self.baseline_training_time is not None:
            metric_values['Time'] = self.baseline_training_time + self.occ_training_time
                
        with open(os.path.join(self.config['directory'], 'occ', occ_method, 'metric_values.json'), 'w') as f:
            json.dump(metric_values, f)
        return self.model
    def _save_results(self):
        if len(self.timesAutoencoder) > 0 and len(self.timesTargetClassifier) > 0:
            print(np.mean(np.array(self.timesAutoencoder[1:])), np.mean(np.array(self.timesTargetClassifier[1:])))

        self._plotLoss(self.elbos, os.path.join(self.config['directory'], 'loss_vae.png'))
        self._plotLoss(self.advGenerationLosses, os.path.join(self.config['directory'], 'loss_disc.png'))
        self._plotLoss(self.discLosses, os.path.join(self.config['directory'], 'loss_gen.png'))
        self._plotLoss(self.labelLosses, os.path.join(self.config['directory'], 'loss_cl.png'))

        np.savez(os.path.join(self.config['directory'], 'PU_loss_val_VAEPU'), loss=self.valLosses)

    def _save_final_metrics(self):
        log2 = open(os.path.join(self.config['directory'], 'log_PN.txt'), 'a')
        acc, precision, recall, f1_score = self.model.accuracy(self.DL_test)

        if self.config['train_occ']:
            log2.write('final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}'.format(self.acc_pre_occ, self.precision_pre_occ, self.recall_pre_occ, self.f1_pre_occ) + '\n')
            print('final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}'.format(self.acc_pre_occ, self.precision_pre_occ, self.recall_pre_occ, self.f1_pre_occ))
        log2.write('final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}'.format(acc, precision, recall, f1_score))
        print('final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}'.format(acc, precision, recall, f1_score))
        log2.close()

        if self.config['train_occ']:
            torch.save(self.model, os.path.join(self.config['directory'], 'model.pt'))
