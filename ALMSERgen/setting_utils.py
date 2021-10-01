import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import Counter
import random
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import os
from profiling_info import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from learning_utils import *
from score_aggregation import *

def calculate_cc_sizes(path, fv_splitter, per_task=True):

    all_records = pd.read_csv(path+"all_pairs.csv")

    all_records['datasource_pair'] = all_records['source'].str.rsplit('_', 1).str[0]+fv_splitter+all_records['target'].str.rsplit('_', 1).str[0]

    matching_pairs = all_records[(all_records.label)]
    non_matching_pairs = all_records[(~all_records.label)]
    
    isolates = set(list(non_matching_pairs['source'].values) + list(non_matching_pairs['target'].values)).difference(set(list(matching_pairs['source'].values) + list(matching_pairs['target'].values)))

    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(matching_pairs, source= 'source', target='target', edge_attr='datasource_pair', create_using=Graphtype)
    con_components = list(nx.connected_components(G))

    print("Complete multi-source task cluster size distribution")
    con_components_lengths_all = [len(x) for x in con_components]
    distribution = Counter(con_components_lengths_all)
    distribution[1] = len(isolates)
    plt.bar(distribution.keys(), distribution.values())
    plt.xticks(np.array(list(set(distribution.keys()))))
    plt.savefig('%s.pdf' % (path+'cc_distribution'), bbox_inches='tight', format='pdf')
    plt.savefig('%s.png' % (path+'cc_distribution'), bbox_inches='tight', format='png')
    print(distribution)
    plt.show()

    if per_task:
        for task in set(all_records['datasource_pair']):
            print("TASK %s" %task)
            components_with_task = list()
            for cc in con_components:
                edge_labels = nx.get_edge_attributes(G.subgraph(cc),'datasource_pair').values()
                if task in edge_labels: 
                    components_with_task.append(cc)

            con_components_lengths = [len(x) for x in components_with_task]

            plt.bar(Counter(con_components_lengths).keys(), Counter(con_components_lengths).values())
            plt.show()
        
   
def write_train_test_fv(main_path):
    
    fv_per_task = list([os.listdir(main_path+"feature_vector_files/")][0])

    pairs_fv = pd.DataFrame()
    
    for fv in fv_per_task:
        fv_of_task = pd.read_csv(main_path+"feature_vector_files/"+fv)
        task_left = fv.split('_')[0]
        task_right = fv.split('_')[1].replace('.csv','')
        fv_of_task['source'] = str(task_left)+'_'+fv_of_task['source_id'].astype(str)
        fv_of_task['target'] = str(task_right)+'_'+fv_of_task['target_id'].astype(str)                      
        pairs_fv = pd.concat([pairs_fv, fv_of_task], ignore_index=True)
    
    #now set agg score and unsupervised label
    columns_w_data = list(set(pairs_fv.columns)-set(['source_id','target_id','label', 'pair_id', 'source', 'target' ]))
    pairs_fv['agg_score'] =  aggregateScores(pairs_fv[columns_w_data])                        
    threhold_value = calculateThreshold(pairs_fv['agg_score'].values, 'elbow')
    pairs_fv['unsupervised_label'] = pairs_fv['agg_score']> threhold_value
                                 
    matching_pairs = pairs_fv[pairs_fv.label]
    non_matching_pairs = pairs_fv[~pairs_fv.label]
    
    #print("Matching Pairs: ", matching_pairs.shape[0])
    #print("Non-Matching Pairs: ", non_matching_pairs.shape[0])


    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(matching_pairs, source= 'source', target='target', create_using=Graphtype)

    con_components = list(nx.connected_components(G))
    subgraphs =  [G.subgraph(c).copy() for c in nx.connected_components(G)]
    con_components_lengths = [len(x) for x in con_components]
    #print(Counter(con_components_lengths))

    random.Random(42).shuffle(con_components)
    train_components = con_components[:int(0.7*len(con_components))]


    test_components = con_components[int(0.7*len(con_components)):]

    #print("Components train: ", len(train_components))
    #print("Components test: ", len(test_components))

    subgraph_train = [G.subgraph(c).copy() for c in train_components]
    train_graph = nx.compose_all(subgraph_train)
    subgraph_test = [G.subgraph(c).copy() for c in test_components]
    test_graph = nx.compose_all(subgraph_test)

    pairs_fv['train_or_test'] = 'not_assigned'
    clean_train_neg=0
    clean_test_neg=0
    for ind, row in pairs_fv.iterrows():
        is_match = row['label']
        assigned=False
        if is_match and train_graph.has_edge(row.source,row.target):
            pairs_fv.at[ind, 'train_or_test']='train'
            assigned=True
        if is_match and test_graph.has_edge(row.source,row.target):
            if assigned: 
                import pdb;pdb.set_trace();
                print("Already assigned")
            pairs_fv.at[ind, 'train_or_test']='test'
            assigned=True
        if not(is_match) and train_graph.has_node(row.source) and train_graph.has_node(row.target):
            clean_train_neg +=1 
            pairs_fv.at[ind, 'train_or_test']='train'
            if assigned:
                import pdb;pdb.set_trace();
                print("Already assigned")
            assigned=True
        if not(is_match) and test_graph.has_node(row.source) and test_graph.has_node(row.target):
            clean_test_neg +=1 
            pairs_fv.at[ind, 'train_or_test']='test'
            if assigned: 
                import pdb;pdb.set_trace();
                print("Already assigned")
            assigned=True
        elif not(is_match):
            flip = random.randint(1, 10)
            if flip<=3: pairs_fv.at[ind, 'train_or_test']='test'
            else:  pairs_fv.at[ind, 'train_or_test']='train'

    #replace nan with -1
    pairs_fv = pairs_fv.fillna(-1)
    
    train_fv= pairs_fv[pairs_fv.train_or_test=='train'].drop(columns=['train_or_test'])
    test_fv= pairs_fv[pairs_fv.train_or_test=='test'].drop(columns=['train_or_test'])

    train_fv.to_csv(main_path+"train_pairs_fv.csv", index= False)
    test_fv.to_csv(main_path+"test_pairs_fv.csv", index= False)
    print("Wrote train and test files in path", main_path)
    pairs_fv.drop(['train_or_test'], axis=1, inplace=True)

    all_pairs_fv = pd.concat([train_fv,test_fv], ignore_index=True)
    all_pairs_fv.to_csv(main_path+"all_pairs.csv", index= False)
    
    return all_pairs_fv


def calculate_current_cc_level(feature_vector, import_features):
    positives = copy.copy(feature_vector[feature_vector['label']==True])
    negatives = copy.copy(feature_vector[feature_vector['label']==False])

    positives = positives.replace(-1, 0)
    negatives = negatives.replace(-1, 0)

    positive_values = positives[import_features].mean(axis=1).values
    negative_values = negatives[import_features].mean(axis=1).values

    thresholds = []
    fp_fn = []
    for t in np.arange(0.0, 1.01, 0.01):
        fn = len(np.where(positive_values<t)[0])
        fp = len(np.where(negative_values>=t)[0])
        thresholds.append(t)
        fp_fn.append(fn+fp)


    optimal_threshold = thresholds[fp_fn.index(min(fp_fn))]
    hard_cases = min(fp_fn)

    groups_positives = positives[import_features].groupby(import_features).size().reset_index()

    cc_current = hard_cases/len(positive_values)
    return cc_current, hard_cases, fn, fp, positives, negatives, positive_values, negative_values, optimal_threshold

def write_and_profile_setting(directory):
    if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(directory+'almser')

    pairs_fv=write_train_test_fv(directory)
    
    #print connected components of setting
    calculate_cc_sizes(directory, '_', per_task=False)

    get_passive_results(directory)

    get_unsupervised_results(directory)
    #profile info
    get_matching_task_profile_info(directory)
    importantProfilingDimensions(directory)
    #get heatmap of setting
    pairs_fv['datasource_pair'] = pairs_fv['source'].str.rsplit('_', 1).str[0]+'_'+pairs_fv['target'].str.rsplit('_', 1).str[0]
    heatm = getHeatmapOfSetting(pairs_fv, set(pairs_fv['datasource_pair'].values))
    heatm.to_csv(directory+"heatmap.csv", index=True)
        
    
def get_unsupervised_results(directory):
    pairs_fv_test= pd.read_csv(directory+"test_pairs_fv.csv")
    pairs_fv_test['datasource_pair'] = pairs_fv_test['source'].str.rsplit('_', 1).str[0]+'_'+pairs_fv_test['target'].str.rsplit('_', 1).str[0]
    
    index_values = ['micro', 'macro']+ list(set(pairs_fv_test['datasource_pair']))
    unsupervised_results= pd.DataFrame(columns=['Precision','Recall','F1'], index = index_values)
    
    prec, recall, fscore, support  = precision_recall_fscore_support(pairs_fv_test['label'].values, pairs_fv_test['unsupervised_label'].values, average='binary')
    
    print("Unsupervised results (elbow method) micro: %f P, %f R, %f F1" % (prec,recall,fscore))
    
    unsupervised_results.at['micro', 'Precision'] = prec
    unsupervised_results.at['micro', 'Recall'] = recall
    unsupervised_results.at['micro', 'F1'] = fscore

    p_per_task = dict()
    r_per_task = dict()
    f1_score_per_task = dict()

    for ds_pair in set(pairs_fv_test['datasource_pair'].values):
        test_X_task = pairs_fv_test[pairs_fv_test.datasource_pair==ds_pair]
        prec_task, recall_task, fscore_task, support_task  = precision_recall_fscore_support(test_X_task['label'].values, test_X_task['unsupervised_label'].values, average='binary')
        
        f1_score_per_task[ds_pair] = fscore_task
        p_per_task[ds_pair] = prec_task
        r_per_task[ds_pair] = recall_task
        
        unsupervised_results.at[ds_pair, 'Precision'] = prec_task
        unsupervised_results.at[ds_pair, 'Recall'] = recall_task
        unsupervised_results.at[ds_pair, 'F1'] = fscore_task
    
    macro_f1=np.mean(list(f1_score_per_task.values()))
    macro_p = np.mean(list(p_per_task.values()))
    macro_r = np.mean(list(r_per_task.values()))

    unsupervised_results.at['macro', 'Precision'] = macro_p
    unsupervised_results.at['macro', 'Recall'] = macro_r
    unsupervised_results.at['macro', 'F1'] = macro_f1
    print("Unsupervised learning results macro: %f P, %f R, %f F1" % (macro_p,macro_r, macro_f1))
    unsupervised_results.to_csv(directory+"unsupervised_results.csv", index=True)
    
    
def get_passive_results(directory):

    print(directory)
    pairs_fv_train= pd.read_csv(directory+"train_pairs_fv.csv")
    pairs_fv_test= pd.read_csv(directory+"test_pairs_fv.csv")

    pairs_fv_train['datasource_pair'] = pairs_fv_train['source'].str.rsplit('_', 1).str[0]+'_'+pairs_fv_train['target'].str.rsplit('_', 1).str[0]
    pairs_fv_test['datasource_pair'] = pairs_fv_test['source'].str.rsplit('_', 1).str[0]+'_'+pairs_fv_test['target'].str.rsplit('_', 1).str[0]

    
    print("----------------")
    metadata_columns = ['source_id','target_id','pair_id', 'agg_score','source','target', 'label', 'unsupervised_label']
    train_X = pairs_fv_train.drop(metadata_columns, axis=1)
    train_y = pairs_fv_train['label']

    test_X = pairs_fv_test.drop(metadata_columns, axis=1)
    test_y = pairs_fv_test['label']
    
    model = getClassifier('rf', random_state=1)
    model.fit(train_X,train_y)
    predictions = model.predict(test_X)
    prec, recall, fscore, support  = precision_recall_fscore_support(test_y, predictions, average='binary')

    print("Passive learning results micro with origin info: %f P, %f R, %f F1" % (prec,recall,fscore))
    print("----------------")

    
    metadata_columns = ['source_id','target_id','pair_id', 'agg_score','source','target', 'label','datasource_pair', 'unsupervised_label']
    train_X = pairs_fv_train.drop(metadata_columns, axis=1)
    train_y = pairs_fv_train['label']

    test_X = pairs_fv_test.drop(metadata_columns, axis=1)
    test_y = pairs_fv_test['label']
    
    model = getClassifier('rf', random_state=1)
    model.fit(train_X,train_y)
    predictions = model.predict(test_X)
    prec, recall, fscore, support  = precision_recall_fscore_support(test_y, predictions, average='binary')

    print("Passive learning results micro: %f P, %f R, %f F1" % (prec,recall,fscore))
    
    index_values = ['micro', 'macro']+ list(set(pairs_fv_train['datasource_pair']))
    passive_results= pd.DataFrame(columns=['Precision','Recall','F1'], index = index_values)
    
    passive_results.at['micro', 'Precision'] = prec
    passive_results.at['micro', 'Recall'] = recall
    passive_results.at['micro', 'F1'] = fscore

    p_per_task = dict()
    r_per_task = dict()
    f1_score_per_task = dict()

    for ds_pair in set(pairs_fv_train['datasource_pair'].values):
        test_X_task = pairs_fv_test[pairs_fv_test.datasource_pair==ds_pair].drop(metadata_columns, axis=1)
        test_y_task = pairs_fv_test[pairs_fv_test.datasource_pair==ds_pair]['label']
        predictions_task = model.predict(test_X_task)
        prec_task, recall_task, fscore_task, support_task  = precision_recall_fscore_support(test_y_task, predictions_task, average='binary')
        f1_score_per_task[ds_pair] = fscore_task
        p_per_task[ds_pair] = prec_task
        r_per_task[ds_pair] = recall_task
        
        passive_results.at[ds_pair, 'Precision'] = prec_task
        passive_results.at[ds_pair, 'Recall'] = recall_task
        passive_results.at[ds_pair, 'F1'] = fscore_task
    
    macro_f1=np.mean(list(f1_score_per_task.values()))
    macro_p = np.mean(list(p_per_task.values()))
    macro_r = np.mean(list(r_per_task.values()))

    passive_results.at['macro', 'Precision'] = macro_p
    passive_results.at['macro', 'Recall'] = macro_r
    passive_results.at['macro', 'F1'] = macro_f1
    print("Passive learning results macro: %f P, %f R, %f F1" % (macro_p,macro_r, macro_f1))
    passive_results.to_csv(directory+"passive_results.csv", index=True)