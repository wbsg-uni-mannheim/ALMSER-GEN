import pandas as pd
from structure_transformations import *
from pattern_transformations import *
from value_transformations import *
import sys
import os

class MultiSourceTaskGenerator(object):
    
    def __init__(self, original_ds, sources, id_attr, pattern, config, other_meta_attributes=[]):
        
        self.sources_count = sources
        self.sources_data = dict()
        for i in range(1,self.sources_count+1):
            self.sources_data[i]=pd.DataFrame()

            
        self.schema = list(set(original_ds.columns)-set(other_meta_attributes)-set(id_attr))
        self.original = original_ds
        self.id_attr = id_attr
        #create vocabulary of each attribute. Will be needed during transformations.
        self.vocabulary = dict()
        for att_ in self.schema:
            self.vocabulary[att_] = set(x for x in self.original[att_].values if str(x)!='nan')
        
        self.config = config
        self.config['pattern']=1 # one pattern of identifying attributes for the whole domain. Not the same like VPO.
        #for negative examples
        self.original_pattern = pattern
        self.patterns = [pattern]
        
        #set id column to string so that we can perform later concatenations of new id values
        self.original[self.id_attr] = self.original[self.id_attr].astype(str)
        
        self.count_value_transformations = dict()
        self.original_entity_to_group_size = dict()
        self.groups_w_same_transf = self.groups_w_same_transf(self.config['groups'])
        
  
    
    def generate(self):
        self.apply_transformation_on_structure()
        self.apply_transformation_on_patterns()
        self.apply_transformation_on_corner_cases()
        print_cc_size_distribution(self)

    def apply_transformation_on_corner_cases(self):
        print("----Apply transformation on corner cases (value transformations)----")
        #set_corner_cases(self, self.config['corner_cases'])
        set_corner_cases_different_per_group(self, self.config['corner_cases'], self.config['groups'])
    
    def apply_transformation_on_structure(self):
        
        print("----Apply transformation on the structure (entity overlap)----")
        self.original_entity_to_group_size = set_structure(self, self.config['structure'])
        print_cc_size_distribution(self)
        
    def apply_transformation_on_patterns(self):
        
        print("----Apply transformation on patterns----")
        self.possible_pattern_levels = calculate_pattern_levels(self)
        
        if self.config['pattern'] not in self.possible_pattern_levels:
            print("The pattern level you have defined is out of range. Possible pattern levels are:", self.possible_pattern_levels)
            sys.exit(0)
        
        if self.config['pattern']>len(self.patterns):
            print("Not enough patterns to achieve the pattern level %i. Please add more patterns." %self.config['pattern'])
            sys.exit(0)
            
        set_pattern(self, self.config['pattern'])
          
    def write_sources(self, path):
        print("----Write sources----")
        if not os.path.exists(path+"sources/"):
            os.makedirs(path+"sources/")
        for source_name in list(self.sources_data.keys()):
            self.sources_data[source_name].to_csv(path+"sources/"+str(source_name)+".csv", index=False)
            
    def groups_w_same_transf(self, group_level):
        groups_w_same_transf = list()
        group_size = int(self.sources_count/group_level)

        #define for how many and for which entities, negative examples should be generated
        rng = default_rng()
        assigned_sources = []
        for i in range(0, len(list(self.sources_data.keys())), group_size):
            assigned_sources.extend(list(self.sources_data.keys())[i:i + group_size])
            groups_w_same_transf.append(list(self.sources_data.keys())[i:i + group_size])

        extra_groups = len(groups_w_same_transf)-group_level
        for i in range(extra_groups):
            extra_group = groups_w_same_transf[i]
            j=i+1
            for source_ in extra_group:
                groups_w_same_transf[j].append(source_)
            groups_w_same_transf.pop(i)
        return groups_w_same_transf
