import copy
import pandas as pd
import random
from numpy.random import default_rng
import numpy as np
import difflib

def set_pattern(setting, pattern_level):
    #create the groups of datasources that will share the same patterns
    groups_w_same_pattern = list()
    group_size = int(setting.sources_count/pattern_level)
    
    #define for how many and for which entities, negative examples should be generated
    rng = default_rng()

    for i in range(0, len(list(setting.sources_data.keys())), group_size):
        groups_w_same_pattern.append(list(setting.sources_data.keys())[i:i + group_size])
    
    #groups_w_same_pattern=setting.groups_w_same_transf
    
    for i in range(len(groups_w_same_pattern)):
        pattern = setting.original_pattern
        #pattern = setting.patterns[i]
        sources_to_inject_pattern = groups_w_same_pattern[i]
        
        print("Inject pattern ",pattern)
        print("In data sources: ", sources_to_inject_pattern)
        reduntant_att = [i for i in pattern if i.startswith('REDUNT_')]
        if len(reduntant_att)>0: 
            reduntant_att=reduntant_att[0].replace('REDUNT_','')
            pattern = [reduntant_att if x=='REDUNT_'+reduntant_att else x for x in pattern]
        else: reduntant_att=None
        
        all_indices_of_group = []
        for source_id in sources_to_inject_pattern:
            all_indices_of_group.extend(setting.sources_data[source_id][setting.id_attr].values)

        ind_to_activate_pattern=rng.choice(list(set(all_indices_of_group)),int(0.2*len(set(all_indices_of_group))), replace=False)
        #for source_of_pattern in sources_to_inject_pattern:
        inject_pattern(setting, ind_to_activate_pattern, sources_to_inject_pattern, pattern, reduntant_att)
        
        
#the source should only have ids
def inject_pattern(setting, ind_to_activate_pattern, sources_of_pattern, pattern, reduntant_att):     

    for source_id in sources_of_pattern:
        setting.sources_data[source_id] = setting.sources_data[source_id].join(setting.original.set_index('id'), on='id')
        #pass the values of the attributes in the pattern of the original dataset to the mutated dataset
        setting.sources_data[source_id] = setting.sources_data[source_id][[setting.id_attr]+setting.original_pattern]

    
    added_rows_in_group=activate_pattern_with_examples(setting, ind_to_activate_pattern, sources_of_pattern, pattern, reduntant_att)
    for source_id in added_rows_in_group.keys():
        setting.sources_data[source_id] = setting.sources_data[source_id].append(added_rows_in_group[source_id], ignore_index=True)



def activate_pattern_with_examples(setting, ind_to_activate_pattern, sources_of_pattern, pattern, red_attribute=None):
    #in order to activate a pattern add examples in the datasource by replicating
    #the values of some of the attributes of the pattern
    #mutate all values of the pattern attributes but NOT of the reduntant ones    
    added_rows = dict()
    
    for source_in_group in sources_of_pattern:
        added_rows[source_in_group] = pd.DataFrame()
        
    attributes_to_mutate = copy.copy(pattern)
    #attributes_to_mutate = list(set(setting.original_pattern)-set(pattern))
    if red_attribute in pattern:
        attributes_to_mutate.remove(red_attribute)
 
    for i in ind_to_activate_pattern:        
        for att in range(len(attributes_to_mutate)):
            #examples_w_att[attributes_to_mutate[att]]+=1

            mutated_attribute_of_pattern = attributes_to_mutate[att]
            copy_columns = setting.original_pattern+[setting.id_attr]
            
            new_row = copy.copy(setting.original[setting.original[setting.id_attr]==i][copy_columns])

            select_source_to_add = np.random.choice(sources_of_pattern)                       
            #for select_source_to_add in sources_to_add:
            new_id = str(i)+'x'+str(select_source_to_add)+str(att)

            new_row[setting.id_attr] = new_id

            #now change the value
            old_value = new_row[mutated_attribute_of_pattern].values[0]
            
            vocab_rest = set(list(setting.vocabulary[mutated_attribute_of_pattern])).difference(set([old_value]))
            similar_new_values = difflib.get_close_matches(old_value,vocab_rest )
            
            if len(similar_new_values)==0:
                random_new_value = random.choice(list(setting.vocabulary[mutated_attribute_of_pattern]))


                #the new value needs to be different from the old one
                while random_new_value==old_value:
                    random_new_value = random.choice(list(setting.vocabulary[mutated_attribute_of_pattern]))
                new_row[mutated_attribute_of_pattern] = random_new_value

                added_rows[select_source_to_add] = added_rows[select_source_to_add].append(new_row, ignore_index=True)
            else:
                new_row[mutated_attribute_of_pattern] = similar_new_values[0]

                added_rows[select_source_to_add] = added_rows[select_source_to_add].append(new_row, ignore_index=True)

    return added_rows
    
def calculate_pattern_levels(setting):

    #we should be able to compare the records of one source to another
    #e.g. given 6 sources we can build the following groups of sources sharing the same pattern
    # 6x1 (all sources share the same pattern), 3x2 (two groups of 3 sources each, each groups shares the same pattern), 2x3 (three groups of 2 sources each)
    pattern_levels = list()
    pattern_levels.append(1)
    for i in range(2,setting.sources_count+1):
        if setting.sources_count%i==0: pattern_levels.append(i)

    return pattern_levels