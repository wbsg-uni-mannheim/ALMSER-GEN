import sys

from datautils import*
import os
import os.path as path
from learningutils import *
from sklearn import tree
from matching_task import *
import time
import glob


matching_tasks_summary = pd.DataFrame(columns=['Dataset', '#records_source', '#records_target', 'count_record_pairs', '#match', '#non-match',
                                               'count_attr','#short_string_attr', '#long_string_attr', '#numeric_attr','avg_density_all'])
matching_tasks_baseline_rf_results = pd.DataFrame(columns=['Dataset', 'precision','recall','f1','f1_std','proba_scores',
                                                          'proba_scores_std','x-val f1','x-val f1 sigma'])
matching_tasks_baseline_svm_results = pd.DataFrame(columns=['Dataset', 'precision','recall','f1','f1_std','proba_scores',
                                                      'proba_scores_std','x-val f1','x-val f1 sigma'])

matching_tasks_profiling = pd.DataFrame(columns=['Dataset','F1_xval_max', 'F1_xval_top_matching_relevant_features', 
                                                 'matching_relevant_features', 
   'matching_relevant_attributes','matching_relevant_attributes_density','matching_relevant_attributes_count',
 'matching_relevant_attributes_datatypes','top_matching_relevant_features','top_relevant_attributes', 
 'top_relevant_attributes_count','top_relevant_attributes_datatypes', 'top_relevant_attributes_density',
'avg_length_tokens_top_relevant_attributes','avg_length_words_top_relevant_attributes','corner_cases_top_matching_relevant_features'])
 
# Use the flags below to indicate which results should be calculates
summaryFeatures=True
profilingFeatures = True                                                           


source_folder="sources/"
fv_folder = "feature_vector_files/"

#add the correct separators of the source sets and the gold standard
sep_for_source_files= ','
gs_sep = ','
train_test_val=False # otherwise nested x-validation for baseline experiments
fv_name_split = "_"
#change for allowing multithreading
threads=-1

def get_matching_task_profile_info(main_path):
     
    dat_counter = 0
    for f in glob.glob(main_path+fv_folder+"/*"):
        dataset_name = f.split("/")[-1].replace(".csv","")
        ds1_name = dataset_name.split(fv_name_split)[0]
        ds2_name = dataset_name.split(fv_name_split)[1]
        #print("Reading ",f)
        feature_vector = pd.read_csv(f)
        if ('cosine_tfidf' in feature_vector.columns):
            feature_vector.drop(columns=['cosine_tfidf'], inplace=True)

        gs = feature_vector[['source_id','target_id','label']].copy()
        gs.rename(columns={'label':'matching'}, inplace=True)

        ds1= pd.read_csv(main_path+source_folder+"/"+ds1_name+".csv", sep =sep_for_source_files, engine='python')
        ds2= pd.read_csv(main_path+source_folder+"/"+ds2_name+".csv", sep =sep_for_source_files, engine='python')
        
        #ds1.drop(columns=['cluster_id'], inplace=True)

        ds1.rename(columns={'id':'subject_id'}, inplace=True)
        #ds2.drop(columns=['cluster_id'], inplace=True)
        ds2.rename(columns={'id':'subject_id'}, inplace=True)

        if not ds1.empty and not ds2.empty and not gs.empty:
            ds1['subject_id'] = ds1['subject_id'].apply(str)
            ds2['subject_id'] = ds2['subject_id'].apply(str)


            gs['source_id'] = gs['source_id'].apply(str)
            gs['target_id'] = gs['target_id'].apply(str)

            common_attributes = [value for value in ds1.columns if (value in ds2.columns and value!='subject_id')]
            matching_task = MatchingTask(ds1, ds2, gs, feature_vector, common_attributes)

            if (summaryFeatures):
                matching_task.getSummaryFeatures()
                #correspondes features
                summary_features = matching_task.dict_summary
                summary_features['Dataset'] = dataset_name

                for key in matching_tasks_summary.columns:
                    matching_tasks_summary.loc[dat_counter, key] = summary_features.get(key)

            if(profilingFeatures):
                matching_task.getProfilingFeatures()
                ident_features_profile =  matching_task.dict_profiling_features
                ident_features_profile['Dataset'] = dataset_name
                for key in matching_tasks_profiling.columns:
                    matching_tasks_profiling.loc[dat_counter,key] = ident_features_profile.get(key)                                  


            dat_counter+=1

    


      
def importantProfilingDimensions(main_path):
    profiling_dimensions = pd.DataFrame(columns=['Dataset'])
    profiling_dimensions['Dataset'] = matching_tasks_summary.Dataset
    profiling_dimensions['Size'] = matching_tasks_summary.count_record_pairs
    profiling_dimensions['Match#'] = matching_tasks_summary['#match']
    profiling_dimensions = pd.merge(profiling_dimensions, matching_tasks_profiling)
    
    for index, row in profiling_dimensions.iterrows():
        relev_attr = row['matching_relevant_attributes']
        
        top_relev_attr = row['top_relevant_attributes']
        
        format_relev_attr = []
        for ra in relev_attr:
            if ra in top_relev_attr: format_relev_attr.append(ra+"*")
            else: format_relev_attr.append(ra)
        profiling_dimensions.loc[index,'matching_relevant_attributes']=format_relev_attr
        
    columns = ['Dataset pair', 'F1-xval_all_attr', 'Relevant Attributes', 'Top Features', 'Schema Complexity', 'Textuality', 'Sparsity', 'Size', 'Match#', 'Corner Cases']
    profiling_dimensions.rename(columns={'Dataset':columns[0], 'F1_xval_max':columns[1],'matching_relevant_attributes':columns[2], 'top_matching_relevant_features':columns[3],
                                       'matching_relevant_attributes_count':columns[4], 'avg_length_words_top_relevant_attributes':columns[5],
                                        'matching_relevant_attributes_density':columns[6], 'corner_cases_top_matching_relevant_features':columns[9]}, inplace=True)
    
    profiling_dimensions= profiling_dimensions[columns]
    profiling_dimensions['Sparsity'] = 1-profiling_dimensions['Sparsity']
    profiling_dimensions.sort_values(by=['Dataset pair'], inplace=True)  
    
    display(profiling_dimensions)
    profiling_dimensions.to_csv(main_path+"profiling.csv", index=False)
    
def getImportantFeatures(feature_vector):
    
    X = feature_vector.drop(['source_id', 'target_id', 'pair_id', 'label', 'agg_score','unsupervised_label', 'source', 'target'], axis=1)
    y =  feature_vector['label'].values
    clf = RandomForestClassifier(random_state=1)
    model = clf.fit(X,y)     
    features_in_order, feature_weights = showFeatureImportances(X.columns.values,model,'rf',display=False) 

    xval_scoring = {'precision' : make_scorer(precision_score),'recall' : make_scorer(recall_score),'f1_score' : make_scorer(f1_score)}         
    
    max_result = cross_validate(clf, X, y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring)

    max_f1_score = round(np.mean(max_result['test_f1_score']),2)
    #gather features that are relevant for 95% of the max f1 score
    sub_result = 0.0
    for i in range(1,len(features_in_order)+1):
        results_subvector = cross_validate(clf, X[features_in_order[:i]], y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring)
        sub_result = round(np.mean(results_subvector ['test_f1_score']),2)
        if (sub_result>0.95*max_f1_score): break; 


    important_features = features_in_order[:i]
    
    return important_features