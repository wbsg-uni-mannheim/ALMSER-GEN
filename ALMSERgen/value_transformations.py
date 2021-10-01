from numpy.random import default_rng
from collections import Counter
from country_abbreviations import *
from string import ascii_letters
import random
import numpy as np
import copy
import numbers
from pandas.api.types import is_numeric_dtype
import pandas as pd
import re
#all value transformations are inspired and adjusted from existing data generator tools
# like LANCE :https://github.com/jsaveta/Lance
# SWING
# EMBENCH

rng = default_rng()
country_abbr = get_country_abbr()

#string_transformations_w_severity = ['addBlankChars', 'blankCharsDeletion', 'randomCharsAddition', 'randomCharsDeletion', 'randomCharsModifiers', 'deleteWord', 'swapAndModifyWords']
string_transformations_w_severity = []
string_transformations_wo_severity = []
numeric_transformations_wo_severity = []

severity_levels = list(np.arange(0.3,0.7, 0.05))

def initialize_transformations():
    global string_transformations_w_severity
    global string_transformations_wo_severity
    global numeric_transformations_wo_severity
    string_transformations_w_severity = ['randomCharsAddition', 'randomCharsDeletion', 'randomCharsModifiers','swapAndModifyWords','insertNoiseWords'] #'deleteWord',, 'insertNoiseWords'
    string_transformations_wo_severity = ['tokenAbbreviation']
    numeric_transformations_wo_severity = ['add5percent', 'add10percent', 'add20percent','sub5percent', 'sub10percent', 'sub20percent', 'set0']

def initialize_count_value_transformations(setting):
    for i in string_transformations_w_severity:
        setting.count_value_transformations[i]=0
    for i in string_transformations_wo_severity:
        setting.count_value_transformations[i]=0
    for i in numeric_transformations_w_severity:
        setting.count_value_transformations[i]=0    
    return setting


#sets the same type of corner cases to the sources of the same group

#set the same type of transformations to ALL records of the same group of sources
#CC defines the severity of transformations (# columns and # severity weight)
def set_corner_cases_different_per_group(setting, cornercases_level, group_level):
    #first define the indices of entities to be mutated
    
    initialize_transformations()
    entities_to_cc_size = pd.DataFrame.from_dict(setting.original_entity_to_group_size, orient='index', columns=['cc_size'])
    entities_to_cc_size.sort_values('cc_size', ascending=False, inplace=True)
    cc_amount = int(len(setting.original[setting.id_attr].values)*cornercases_level)
    
    #all_entities_indices = rng.choice(list(setting.original[setting.id_attr].values), int(len(setting.original[setting.id_attr].values)*cornercases_level), replace=False) 
    
    all_entities_indices = list(setting.original[setting.id_attr].values)
        
    #in what size components do the selected entities belong?
    selected_entities_to_group_size = {key: value for key, value in setting.original_entity_to_group_size.items() if key in all_entities_indices}
    
    print("Selected entities to groups size:")
    print(Counter(selected_entities_to_group_size.values()))


    groups_transformations= list()

    print("Groups of sources with same transformations :",setting.groups_w_same_transf )
    group_id=-1
    for group_of_sources in setting.groups_w_same_transf:
        #the severity should be defined by the cc level
        #severity level of group
        #group_sev_level = np.random.choice(severity_levels)
        
        #print("Value transformations for group of sources:", group_of_sources)
        group_id+=1
        #get common entities of group where
        all_ids_in_group = []

        for source_ in group_of_sources:
            all_ids_in_group.extend(setting.sources_data[source_][setting.id_attr].values)
        
        only_injected = list(set(all_ids_in_group) - set(list(setting.original[setting.id_attr].values)))
        
        #index_of_entity_to_add_cc= rng.choice(list(only_injected), int(len(only_injected)*cornercases_level), replace=False)
        #index_of_entity_to_add_cc=index_of_entity_to_add_cc.tolist()
        index_of_entity_to_add_cc=list(only_injected)
        index_of_entity_to_add_cc.extend(all_entities_indices)
        index_of_entity_to_add_cc = set(index_of_entity_to_add_cc)
        
        
        
        print("index_of_entity_to_add_cc:", len(index_of_entity_to_add_cc))
        
        #assign one transformation to each column
        all_columns = list(set(setting.sources_data[group_of_sources[0]].columns) - set([setting.id_attr]))
        
        
        column_to_trans=dict()
                   
        #make sure each group is assigned a different transformation
        while (column_to_trans in groups_transformations) or (not column_to_trans):
            column_to_trans=dict()
            column_to_severity=dict() 
                        
            #select amount of columns to be transformed based on the CC level
            selected_col_for_group =  rng.choice(all_columns, max(int(len(all_columns)*cornercases_level)+1, 2), replace=False)
            
            #selected_col_for_group = all_columns
                 
            for column in selected_col_for_group:
                if is_numeric_dtype(setting.sources_data[group_of_sources[0]][column]):
                    column_to_trans[column] = random.sample(numeric_transformations_wo_severity,1)[0]
                    column_to_severity[column] = -1 
                else:
                   # w_severity = random.uniform(0.0,1.0)
                   # if w_severity>0.2:
                    try:
                        merged_dict ={k: v for d in groups_transformations for k, v in d.items()}
                        transf_of_group=random.sample(string_transformations_w_severity,1)[0]
                        #sev_of_group = group_sev_level 
                        sev_of_group = cornercases_level/2
                        if column in merged_dict:
                            iter_=0
                            while merged_dict[column] == transf_of_group and iter_<50:
                                transf_of_group=random.sample(string_transformations_w_severity,1)[0]
                                sev_of_group = cornercases_level
                                iter_+=1
                            if iter_==50:
                                print("Reached 50 iterations while searching for a new combination of column-value transf. Will stop.")
                    except:
                        import pdb;pdb.set_trace();
                   # else:
                        #transf_of_group=random.sample(string_transformations_wo_severity,1)[0]
                        #sev_of_group = -1
                    
                    
                    column_to_trans[column]=transf_of_group
                    column_to_severity[column] = sev_of_group
                        
        groups_transformations.append(column_to_trans)
        
        print("Will apply the transformations: ", column_to_trans)
        print("with severity: ", column_to_severity)

        print("To the group of sources: ", group_of_sources)
        #print(column_to_severity)
        for index_ in index_of_entity_to_add_cc:
            sources_w_entity = []
            for i in group_of_sources:
                if index_ in setting.sources_data[i][setting.id_attr].values:
                    sources_w_entity.append(i)

            if len(sources_w_entity)==0: continue

            for source_id in sources_w_entity:
                source_ = copy.copy(setting.sources_data[source_id])
                idx = source_[source_[setting.id_attr]==index_].index.values[0]

                all_columns = list(set(source_.columns) - set([setting.id_attr]))
                #column = np.random.choice(all_columns)
                for column in selected_col_for_group:
                    try:
                        if column_to_severity[column]==-1:
                            new_value = globals()[column_to_trans[column]](source_.at[idx, column])
                        else:
                            new_value = globals()[column_to_trans[column]](source_.at[idx, column],column_to_severity[column])

                    except: 
                        print("Problem here")
                        import pdb;pdb.set_trace();

                    source_.at[idx, column]=new_value
                    #print("New value: ", new_value)

                    setting.sources_data[source_id]=source_

def set_corner_cases(setting, cornercases_level):
    #first define the indices of entities to be mutated
    not_injected_ind = setting.original[setting.id_attr].values
    index_of_entity_to_add_cc = rng.choice(list(not_injected_ind), int(not_injected_ind.shape[0]*cornercases_level), replace=False)    
    for source_id in setting.sources_data:        
        for index_ in index_of_entity_to_add_cc:
            if index_ in setting.sources_data[source_id][setting.id_attr].values:
                all_columns = list(set(setting.sources_data[source_id].columns) - set([setting.id_attr]))
                for column in all_columns:
                    if is_numeric_dtype(setting.sources_data[source_id][column]):
                        transformation = random.sample(numeric_transformations_w_severity,1)[0]
                        severity = random.uniform(0.5, 1)
                    else:
                        w_severity = random.uniform(0.5, 1)
                        if w_severity>0.2:
                            transformation=random.sample(string_transformations_w_severity,1)[0]
                            severity = cornercases_level
                        else:
                            transformation=random.sample(string_transformations_wo_severity,1)[0]
                            severity = -1
                    source_ = copy.copy(setting.sources_data[source_id])

                    idx = source_[source_[setting.id_attr]==index_].index.values[0]
                    if severity==-1:
                        new_value = globals()[transformation](source_.at[idx, column])
                    else:
                        new_value = globals()[transformation](source_.at[idx, column],severity)
                    source_.at[idx, column]=new_value

                    setting.sources_data[source_id]=source_    
        
                     

def swapWords(value_):
    words = value_.split()
    random.shuffle(words)
    new_value = ' '.join(words)
    return new_value
 
def swapAndModifyWords(value_, severity):
    modifiers = ['addBlankChars', 'blankCharsDeletion', 'randomCharsAddition', 'randomCharsDeletion', 'randomCharsModifiers']

    words = value_.split()
    random.shuffle(words)
    new_value = ' '.join(words)

    transformation = random.sample(modifiers,1)[0]
    new_value = globals()[transformation](new_value,severity)

    return new_value

def addBlankChars(value_,severity):
    
    value_length = len(value_)
    how_many_blanks = int(severity*len(value_))
    how_many_blanks = max(1, how_many_blanks)

    index_to_add_blank = sorted(rng.choice(value_length+how_many_blanks, how_many_blanks, replace=False))
    new_value = value_
    for i in index_to_add_blank:
        new_value = new_value[:i]+' '+new_value[i:]
        index_to_add_blank = [x+1 for x in index_to_add_blank]
        
    return new_value


def replace_random(src, frm, to):
    matches = list(re.finditer(frm, src))
    replace = random.choice(matches)
    return src[:replace.start()] + to + src[replace.end():]

def blankCharsDeletion(value_, severity):
    try:
        if ' ' not in value_: return value_ 
        blankslength = Counter(value_)[' ']
        how_many_deletions = int(severity*blankslength)
        how_many_deletions = max(1, int(how_many_deletions))

        new_value = value_
        for _ in range(how_many_deletions-1):
            new_value =replace_random(new_value, r' ', '')
    except:
        import pdb;pdb.set_trace();
    return new_value


def countryAbbreviation(value_):
    new_value = value_.lower()
    #contains some country name?
    contained_country = None
    for country in list(country_abbr.keys()):
        if country in new_value:
            contained_country= country
            break;

    abbr = country_abbr.get(contained_country)
    if abbr is None: return new_value
    else: return new_value.replace(contained_country, abbr)
    
def tokenAbbreviation(value_):
    tokens = value_.split(' ')
    random_token_to_abbreviate_index = rng.choice(len(tokens), 1)[0]
    token_to_abbrev =  tokens[random_token_to_abbreviate_index]                               
    abbrev = token_to_abbrev[:1]+"."
    
    new_value = ''
    for i in range(len(tokens)):
        if i == random_token_to_abbreviate_index:
            new_value+=abbrev+' '
        else: new_value+=tokens[i]+' '
    
    return new_value

def randomCharsAddition(value_, severity):
    try:
        inds = [i for i,_ in enumerate(value_) if not value_.isspace()]

        how_many_additions = int(severity*len(value_))
        how_many_additions = max(1, how_many_additions)

        sam = random.sample(inds, how_many_additions)

        letts =  iter(random.choices(ascii_letters, k=how_many_additions))
        lst = list(value_)
        for ind in sam:
            lst[ind] = next(letts)

        new_value = "".join(lst)
    except: 
        import pdb;pdb.set_trace();
    return new_value

def insertNoiseWords(value_, severity):
    try:
        noise_words=['record','value','row','attribute','entry', 'check', 'top', 'generated', 'text']
        words_count=len(value_.split(' '))

        how_many_noise_words = max(1,int(severity*words_count))
        
        select_random_words = random.choices(noise_words,k=how_many_noise_words)

        new_value = " ".join(select_random_words)
        new_value = new_value + " " + value_
    except: 
        import pdb;pdb.set_trace();
    return new_value

def randomCharsDeletion(value_, severity):
    value_length = len(value_)
    
    index_of_chars = [i for i in range(value_length) if value_[i]!=' ']
    how_many_deletions = int(severity*len(index_of_chars))
    how_many_deletions = max(1, how_many_deletions)
    
    index_to_remove = rng.choice(index_of_chars, how_many_deletions, replace=False)
    temp = list(value_)
    for idx in index_to_remove:
        temp[idx] = ''
    new_value = ''.join(temp)
    
    return new_value


def randomCharsModifiers(value_, severity):
    value_length = len(value_)
    
    index_of_chars = [i for i in range(value_length) if value_[i]!=' ']
    how_many_modifications = int(severity*len(index_of_chars))
    how_many_modifications = max(1, how_many_modifications)
    index_to_modify = rng.choice(index_of_chars, how_many_modifications, replace=False)
    temp = list(value_)
    for idx in index_to_modify:
        temp[idx] = random.sample(ascii_letters, 1)[0]
    new_value = ''.join(temp)

    return new_value

#if high severity change first digits, if low change last digits
def changeNumber(value_, severity):
    number_as_string = str(value_)
    which_digit_to_change = len(number_as_string)-int(severity*len(number_as_string))-1
    
    current_digit = number_as_string[which_digit_to_change]
    while not current_digit.isnumeric():
        which_digit_to_change+=1
        current_digit = number_as_string[which_digit_to_change]

    temp = list(number_as_string)
    temp[which_digit_to_change] = str(random.sample(list(np.arange(9)), 1)[0])
    new_number_as_string = ''.join(temp)
    
    return float(new_number_as_string)

def add5percent(value_):
    return value_*1.05

def add10percent(value_):
    return value_*1.10

def add20percent(value_):
    return value_*1.20

def sub5percent(value_):
    return value_*0.95

def sub10percent(value_):
    return value_*0.90

def sub20percent(value_):
    return value_*0.80

def set0(value_):
    return 0

def deleteWord(value_, severity):
    tokens = value_.split(' ')
    remove_tokens =random.sample(list(tokens), int(severity*len(tokens)))
    new_value= copy.copy(value_)
    for rem_token in set(remove_tokens):
        new_value.replace(rem_token, '')
    return new_value

def deleteCompleteValue(value_):
    return None