import pandas as pd
from tqdm.auto import tqdm, trange
import string
import re
import os, os.path
import copy
import numpy as np

#simulates blocking: gets all positive pairs and all constructed good negatives
def get_pairs_for_fv(path, id_attr):
    all_sources = list([os.listdir(path+"sources/")][0])
    all_sources.sort()
                       
    pairs_per_task = dict()

    for i in range(len(all_sources)):
        for j in range(i+1,len(all_sources)):
            pairs_of_task= list()

            task_left = str(all_sources[i]).replace('.csv', '')
            task_right = str(all_sources[j]).replace('.csv', '')

            task_left_data = pd.read_csv(path+"sources/"+all_sources[i])
            task_right_data = pd.read_csv(path+"sources/"+all_sources[j])
            
            #convert id to string
            task_left_data[id_attr] = task_left_data[id_attr].astype(str)
            task_right_data[id_attr] = task_right_data[id_attr].astype(str)


            #add all matches
            matches = get_matches(task_left_data, task_right_data,id_attr)
            pairs_of_task.extend(matches)

            #add all constructed negatives
            non_matches_added = get_added_negatives(task_left_data, task_right_data,id_attr) 
            pairs_of_task.extend(non_matches_added)                              

            #add random negatives so that the ratio positives/negatives is 1/3
            how_many_random = len(matches)*3-len(non_matches_added)
            if how_many_random>0:
                non_matches_random = get_random_negatives(task_left_data, task_right_data, how_many_random, id_attr)                              
                pairs_of_task.extend(non_matches_random)
            pairs_per_task[task_left+'_'+task_right] =     pd.DataFrame(list(set(pairs_of_task)), columns =['source_id', 'target_id', 'matching'])
     
    write_all_pairs_after_blocking(path, pairs_per_task)
    
def get_random_negatives(task_left, task_right, how_many, id_attr):
    random_non_matches = list()
    idx_left = task_left[id_attr].values
    idx_right = task_right[id_attr].values
    for i in np.arange(how_many):
        random_left_idx = np.random.choice(idx_left,1)[0]
        random_right_idx = np.random.choice(idx_right,1)[0]
        while random_left_idx==random_right_idx:
            random_left_idx = np.random.choice(idx_left,1)[0]
            random_right_idx = np.random.choice(idx_right,1)[0]
        random_non_matches.append((random_left_idx, random_right_idx, False))
    return random_non_matches

def get_matches(task_left, task_right,id_attr):
    matches = list()
    idx_left = task_left[id_attr].values
    idx_right = task_right[id_attr].values
    overlapping_idx = list(set(idx_left) & set(idx_right)) 
    matches = [(x,x,True) for x in overlapping_idx]                                       
    return matches

def get_added_negatives(task_left, task_right,id_attr):
    added_negatives = list()
    added_idx_left = task_left[task_left[id_attr].str.contains('x')][id_attr].values
    added_idx_right = task_right[task_right[id_attr].str.contains('x')][id_attr].values
    for added_left in added_idx_left:
        orig_ =  added_left.split('x')[0]
        if orig_ in task_right[id_attr].values:
            added_negatives.append((added_left,orig_, False))
                                          
    for added_right in added_idx_right:
        orig_ =  added_right.split('x')[0]
        if orig_ in task_left[id_attr].values:
            added_negatives.append((orig_,added_right, False))
                                          
    return added_negatives

def write_all_pairs_after_blocking(path, pairs_per_task):
    if not os.path.exists(path+'blocked_pairs/'):
        os.makedirs(path+'blocked_pairs/')
    for matching_task in list(pairs_per_task.keys()):
        pairs_per_task[matching_task].to_csv(path+'blocked_pairs/'+matching_task+'.csv', index=False)
    
#unused                                          
def searcher(ix, search_data):
    
    pairs_after_blocking=list()
    for i in tqdm(range(search_data.shape[0])):
        query_string = normalize_string(search_data.content.iloc[i])
        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema)
            parser.add_plugin(FuzzyTermPlugin())

            query= parser.parse(query_string+'~10')
            results = searcher.search(query, terms=True)
    
            for r in results:
                if (search_data.iloc[i].id!=r["path"]):
                    import pdb;pdb.set_trace();
                pairs_after_blocking.append((search_data.iloc[i].id,r["path"]))
    return  pairs_after_blocking 



#unused
def normalize_string(str_):
    str_=normalize('NFD',str_)
    str_=str_.lower()
    str_=re.sub(' +', ' ', str_)
    str_=str_.translate(str.maketrans('', '', string.punctuation))
    return str_  
                                          
#unused                                          
def indexer(source_data, path_to_index, source_name):

    schema = Schema(content=TEXT(stored = True), path=ID(stored=True))    
    ix = index.create_in(path_to_index, schema)
    writer = ix.writer()
    
    for i in range(source_data.shape[0]):
        writer.add_document(content=normalize_string(str(source_data.content.iloc[i])), path=str(source_data.id.iloc[i]))
    writer.commit()
    
    source_to_index[source_name]=copy.copy(ix)                                          