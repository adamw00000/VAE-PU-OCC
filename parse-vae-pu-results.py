# %%
import os
import re
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

root = 'result'
results = []

for dataset in os.listdir(root):
    if dataset == '.gitkeep':
        continue
    
    for c in os.listdir(os.path.join(root, dataset)):
        if c.startswith('Exp'):
            continue

        for exp in os.listdir(os.path.join(root, dataset, c)):
            exp_num = int(exp[3:])

            if not os.path.exists(os.path.join(root, dataset, c, exp, 'occ')):
                continue
            
            try:
                with open(os.path.join(root, dataset, c, exp, 'metric_values.json'), 'r') as f:
                    metrics = json.load(f)

                    metrics['Dataset'] = dataset
                    metrics['Experiment'] = exp_num
                    metrics['c'] = float(c)
                    
                    results.append(metrics)

                if os.path.exists(os.path.join(root, dataset, c, exp, 'metric_values_orig.json')):
                    with open(os.path.join(root, dataset, c, exp, 'metric_values_orig.json'), 'r') as f:
                        metrics = json.load(f)

                        metrics['Dataset'] = dataset
                        metrics['Experiment'] = exp_num
                        metrics['c'] = float(c)
                        
                        results.append(metrics)

                for occ_method in os.listdir(os.path.join(root, dataset, c, exp, 'occ')):
                    with open(os.path.join(root, dataset, c, exp, 'occ', occ_method, 'metric_values.json'), 'r') as f:
                        metrics = json.load(f)

                        metrics['Dataset'] = dataset
                        metrics['Experiment'] = exp_num
                        metrics['c'] = float(c)
                    
                        results.append(metrics)
                
                if os.path.exists(os.path.join(root, dataset, c, exp, 'external')):
                    for method in os.listdir(os.path.join(root, dataset, c, exp, 'external')):
                        with open(os.path.join(root, dataset, c, exp, 'external', method, 'metric_values.json'), 'r') as f:
                            metrics = json.load(f)

                            metrics['Dataset'] = dataset
                            metrics['Experiment'] = exp_num
                            metrics['c'] = float(c)
                        
                            results.append(metrics)
            except:
                continue
        # break

results_df = pd.DataFrame.from_records(results)
results_df = results_df.assign(BaseMethod = results_df.Method.str.replace('\+(Storey|VAERisk|PValBootstrap|G[0-9])', '', regex=True)) \
    .assign(Storey = np.where(results_df.Method.str.contains('\+Storey'), 'Storey', '-')) \
    .assign(VAERiskTraining = np.where(results_df.Method.str.contains('\+VAERisk'), '-', 'no VAE risk training')) \
    .assign(Bootstrap = np.where(results_df.Method.str.contains('\+PValBootstrap'), 'p-val', '-')) \
    .assign(GenerateAveraging = np.where(~pd.isnull(results_df.Method.str.extract('\+G([0-9])')[0]), results_df.Method.str.extract('\+G([0-9])')[0], '-'))

results_df = results_df \
    .drop(columns='Method') \
    .rename(columns={ 'BaseMethod': 'Method' })
results_df.Method = np.where(
    results_df.Method == 'A^3',
    r'$A^3$',
    results_df.Method,
)
results_df.Method = np.where(
    results_df.Method == 'No OCC',
    r'Baseline',
    results_df.Method,
)
results_df


# %%
def process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols, multicolumn=False, plot_results=True):
    filtered_df = results_df

    if min_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment >= min_exp]
    if max_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment < max_exp]
    if methods_filter is not None:
        filtered_df = filtered_df.loc[np.isin(filtered_df.Method, methods_filter)]
    if dataset_filter is not None:
        filtered_df = filtered_df.loc[filtered_df.Dataset == dataset_filter]
    
    processed_results = filtered_df \
        .drop(columns='Experiment')
    if methods_filter is not None:
        if 'Baseline' in methods_filter and 'Method' in grouping_cols:
            processed_results['IsNotBaseline'] = ~(processed_results.Method.str.contains('Baseline') | processed_results.Method.str.contains('EM'))
            grouping_cols_copy = grouping_cols
            grouping_cols_copy.insert(grouping_cols_copy.index('Method'), 'IsNotBaseline')

            processed_results = processed_results \
                .sort_values(grouping_cols_copy)

    for method, name in [
        ('Baseline', 'Baseline (modified)'),
        ('Baseline (orig)', 'Baseline (original)'),
        ('EM', 'SAR-EM'),
        ('ECODv2', 'ECOD'),
        ('$A^3$', 'VAE-PU+$A^3$'),
        ('IsolationForest', 'VAE-PU+IsolationForest'),
        ('OC-SVM', 'VAE-PU+OC-SVM'),
        ('ECOD', 'VAE-PU+ECOD'),
    ]:
        processed_results.Method = np.where(processed_results.Method == method, name, processed_results.Method)
    
    processed_results_mean = processed_results \
        .groupby(grouping_cols) \
        .mean() \
        .round(4) * 100
    processed_results_sem = processed_results \
        .groupby(grouping_cols) \
        .sem() \
        .round(4) * 100
    processed_results_counts = processed_results \
        .groupby(grouping_cols) \
        .size()
    # display(processed_results_counts)
    
    if 'IsNotBaseline' in processed_results_mean.index.names:
        processed_results_mean.index = processed_results_mean.index.droplevel('IsNotBaseline')
        processed_results_sem.index = processed_results_sem.index.droplevel('IsNotBaseline')

    if result_cols is not None:
        processed_results_mean = processed_results_mean.loc[:, result_cols]
        processed_results_sem = processed_results_sem.loc[:, result_cols]

    os.makedirs(os.path.join('processed_results', df_name), exist_ok=True)
    os.makedirs(os.path.join('processed_results', '_all_tables'), exist_ok=True)
    os.makedirs(os.path.join('processed_results', '_all_plots'), exist_ok=True)
    processed_results_mean.to_csv(os.path.join('processed_results', df_name, 'metrics.csv'))

    # PLOT RESULTS

    if plot_results:
        sns.set_theme()
        plt.figure(figsize=(8, 6))

        plot_df = processed_results.reset_index(drop=False)
        plot_df.Accuracy *= 100
        sns.lineplot(data=plot_df, x='c', y='Accuracy', hue='Method', err_style='bars', ci=68, err_kws={'capsize': 3})
        plt.xlabel('Label frequency $c$')
        plt.ylabel('Accuracy [%]')
        plt.xlim(0, 0.72)
        plt.ylim(None, 100)
        
        plt.savefig(os.path.join('processed_results', df_name, 'plot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join('processed_results', '_all_plots', f'{df_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # PREPARE RESULT TABLES

    def highlight_max(df, value_df):
        is_max = value_df.groupby(level=0).transform('max').eq(value_df)
        max_df = pd.DataFrame(np.where(is_max == True, '\\textbf{' + df + '}', df),
            index=df.index, columns=df.columns)
        return max_df
    
    processed_results = processed_results_mean.applymap(lambda a: f'{a:.2f}') + ' $\pm$ ' + processed_results_sem.applymap(lambda a: f'{a:.2f}')
    processed_results = highlight_max(processed_results, processed_results_mean)
    
    include_caption = True
    include_label = True

    latex_table = processed_results \
            .to_latex(
                index=True, 
                escape=False, 
                multirow=True,
                caption=df_name + '.' if include_caption else None, 
                label='tab:' + df_name.replace(" ", "_") if include_label else None,
                position=None if not include_label and not include_caption\
                    else 'tbp' if not multicolumn else 'btp'
            )
    cline_start = len(processed_results.index.names)
    cline_end = cline_start + len(processed_results.columns)

    # add full rule before baseline
    # latex_table = re.sub(r'(\\\\.*?\n)(.*?)Baseline', r'\1\\midrule \n\2Baseline', latex_table)

    # add mid rule after EM
    latex_table = re.sub(r'(EM.*? \\\\)', r'\1 \\cline{' \
        + str(cline_start) + '-' + str(cline_end) + \
    '}', latex_table)

    # merge headers
    def merge_headers(latex_table):
        table_lines = latex_table.split('\n')
        tabular_start = 0
        tabular_end = len(table_lines) - 3

        if include_caption or include_label:
            tabular_start += 3
            tabular_end -= 1
        if include_caption and include_label:
            tabular_start += 1

        def process_line(l):
            return ['\\textbf{' + name.replace('\\', '').strip() + '}' 
                for name in l.split('&')
                if name.replace('\\', '').strip() != '']
        header_line, index_line = table_lines[tabular_start + 2], table_lines[tabular_start + 3]
        headers = process_line(header_line)
        index_names = process_line(index_line)

        new_headers = index_names + headers
        new_headers[-1] += ' \\\\'
        new_headers = ' & '.join(new_headers)

        table_lines.remove(header_line)
        table_lines.remove(index_line)
        table_lines.insert(tabular_start + 2, new_headers)

        table_lines = ['\t' + l if i > tabular_start and i < tabular_end else l
            for i, l in enumerate(table_lines)]
        if include_caption or include_label:
            table_start = 0
            table_end = len(table_lines) - 2
            table_lines = ['\t' + l if i > table_start and i < table_end else l
                for i, l in enumerate(table_lines)]

        # insert scaling
        table_lines.insert(tabular_end + 1, '}')
        table_lines.insert(tabular_start, '\scalebox{0.9}{')
        # insert scaling

        return '\n'.join(table_lines)
    
    latex_table = merge_headers(latex_table)

    if multicolumn:
        latex_table = latex_table.replace('{table}', '{table*}')

    with open(os.path.join('processed_results', df_name, 'metrics.tex'), 'w') as f:
        f.write(latex_table)
    with open(os.path.join('processed_results', '_all_tables', f'{df_name.replace(" ", "_")}.tex'), 'w') as f:
        f.write(latex_table)
    
    print(df_name)
    display(processed_results)

    # return processed_results

### ---------------------------------------------------------

# df_name = 'Original'
# min_exp, max_exp = None, None
# methods_filter = None
# dataset_filter = None
# grouping_cols = ['Dataset', 'c', 'Method', 'Storey', 'VAERiskTraining', 'Bootstrap', 'GenerateAveraging']
# result_cols = None

# process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols, plot_results=False)

df_name = 'MNIST 3v5 Results -- No-SCAR'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'MNIST 3v5'
grouping_cols = ['c', 'Method']
result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
multicolumn = True
# result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols, multicolumn=multicolumn)

df_name = 'MNIST OvE Results -- No-SCAR'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'MNIST OvE'
grouping_cols = ['c', 'Method']
# result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols)

df_name = 'CIFAR CarTruck Results -- No-SCAR'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'CIFAR CarTruck'
grouping_cols = ['c', 'Method']
# result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols)

df_name = 'CIFAR MachineAnimal Results -- No-SCAR'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'CIFAR MachineAnimal'
grouping_cols = ['c', 'Method']
# result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols)

# https://arxiv.org/pdf/2106.03253.pdf, p. 12
df_name = 'Gas Concentrations Results -- No-SCAR'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'Gas Concentrations'
grouping_cols = ['c', 'Method']
# result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols)

df_name = 'STL MachineAnimal Results -- No-SCAR'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'STL MachineAnimal'
grouping_cols = ['c', 'Method']
# result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols)

df_name = 'STL MachineAnimal Results -- SCAR Setting'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = 'STL MachineAnimal SCAR'
grouping_cols = ['c', 'Method']
# result_cols = ['Accuracy', 'Precision', 'Recall', 'F1 score']
result_cols = ['Accuracy', 'F1 score']

process_results(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, result_cols)

# %%
def process_time(df_name, min_exp, max_exp, \
        methods_filter, dataset_filter, grouping_cols, \
        multicolumn=False, pivot_info=None, \
        include_caption=True, include_label=True, \
        col_order=None
    ):
    filtered_df = results_df

    if min_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment >= min_exp]
    if max_exp is not None:
        filtered_df = filtered_df.loc[filtered_df.Experiment < max_exp]
    if methods_filter is not None:
        filtered_df = filtered_df.loc[np.isin(filtered_df.Method, methods_filter)]
    if dataset_filter is not None:
        filtered_df = filtered_df.loc[filtered_df.Dataset.str.contains('|'.join(dataset_filter))]
    
    processed_results = filtered_df \
        .drop(columns='Experiment')
    if methods_filter is not None:
        if 'Baseline' in methods_filter and 'Method' in grouping_cols:
            processed_results['IsNotBaseline'] = ~(processed_results.Method.str.contains('Baseline') | processed_results.Method.str.contains('EM'))
            grouping_cols_copy = grouping_cols
            grouping_cols_copy.insert(grouping_cols_copy.index('Method'), 'IsNotBaseline')

            processed_results = processed_results \
                .sort_values(grouping_cols_copy)

    for method, name in [
        ('Baseline', 'Baseline (modified)'),
        ('Baseline (orig)', 'Baseline (original)'),
        ('EM', 'SAR-EM'),
        ('LBE', 'LBE'),
        ('ECODv2', 'ECOD'),
        ('$A^3$', 'VAE-PU+$A^3$'),
        ('IsolationForest', 'VAE-PU+IsolationForest'),
        ('OC-SVM', 'VAE-PU+OC-SVM'),
        ('ECOD', 'VAE-PU+ECOD'),
    ]:
        processed_results.Method = np.where(processed_results.Method == method, name, processed_results.Method)
    
    for dataset, name in [
        ('CIFAR_CarTruck_red_val', 'CIFAR CarTruck'),
        ('MNIST_35_bold_val', 'MNIST 3v5'),
        ('STL_MachineAnimal_red_val', 'STL MachineAnimal'),
        ('gas-concentrations', 'Gas Concentrations'),
        ('CIFAR_MachineAnimal_red_val', 'CIFAR MachineAnimal'),
        ('MNIST_evenodd_bold_val', 'MNIST OvE'),
    ]:
        processed_results.Dataset = np.where(processed_results.Dataset == dataset, name, processed_results.Dataset)
    
    results_pivot = pd.pivot_table(processed_results, index=pivot_info['index'], columns=pivot_info['columns'], values=pivot_info['values'])
    results_pivot.columns.name = results_pivot.index.name
    results_pivot.index.name = None
    results_pivot = results_pivot.round(2)

    if col_order is not None:
        results_pivot = results_pivot[col_order]

    display(results_pivot)
    
    latex_table = results_pivot.style.highlight_min(props='textbf:--rwrap;', axis=0) \
        .format("{:.2f}s".format) \
        .to_latex(
            caption=df_name + '.' if include_caption else None, 
            label='tab:' + df_name.replace(" ", "_") if include_label else None,
            position=None if not include_label and not include_caption\
                else 'tbp' if not multicolumn else 'btp',
            hrules=True,
            position_float='centering',
        )

    if multicolumn:
        latex_table = latex_table.replace('{table}', '{table*}')
    
    def add_scaling(latex_table):
        table_lines = latex_table.split('\n')
        tabular_start = 0
        tabular_end = len(table_lines) - 2

        if include_caption or include_label:
            tabular_start += 3
            tabular_end -= 1
        if include_caption and include_label:
            tabular_start += 1

        table_lines = ['\t' + l if i > tabular_start and i < tabular_end else l
            for i, l in enumerate(table_lines)]
        if include_caption or include_label:
            table_start = 0
            table_end = len(table_lines) - 2
            table_lines = ['\t' + l if i > table_start and i < table_end else l
                for i, l in enumerate(table_lines)]

        # insert scaling
        table_lines.insert(tabular_end + 1, '}')
        table_lines.insert(tabular_start, '\scalebox{0.95}{')
        # insert scaling

        # insert arraystretch
        table_lines.insert(1, '\t\\renewcommand{\\arraystretch}{1.1}')
        # insert arraystretch

        return '\n'.join(table_lines)
    
    latex_table = add_scaling(latex_table)
    
    os.makedirs(os.path.join('processed_results', df_name), exist_ok=True)
    os.makedirs(os.path.join('processed_results', '_all_tables'), exist_ok=True)
    os.makedirs(os.path.join('processed_results', '_all_plots'), exist_ok=True)

    with open(os.path.join('processed_results', df_name, 'metrics.tex'), 'w') as f:
        f.write(latex_table)
    with open(os.path.join('processed_results', '_all_tables', f'{df_name.replace(" ", "_")}.tex'), 'w') as f:
        f.write(latex_table)

df_name = 'Training Time per Dataset ($c = 0.5$)'
min_exp, max_exp = 0, None
methods_filter = ['Baseline', 'Baseline (orig)', 'OC-SVM', 'IsolationForest', 'ECODv2', r'$A^3$', 'EM']
dataset_filter = None
grouping_cols = ['Dataset', 'Method']
pivot_info = { 'index': 'Method', 'columns': 'Dataset', 'values': 'Time' }
result_cols = ['Time']

column_order = ['MNIST 3v5', 'MNIST OvE', 'CIFAR CarTruck', 'CIFAR MachineAnimal', 'STL MachineAnimal', 'Gas Concentrations']

process_time(df_name, min_exp, max_exp, methods_filter, dataset_filter, grouping_cols, multicolumn=True, pivot_info=pivot_info, col_order=column_order)

# %%
