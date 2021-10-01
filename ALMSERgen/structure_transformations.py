import random
from itertools import groupby
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
from numpy.random import default_rng


#iterate over the original source
#add with probability <structure level> an entity in more than two sources
#the distribution should be power low (many entities in few sources, less entities in all sources) even at high structuredness levels
def set_structure(setting, structure_level):
    sources_sorted = sorted(i for i in setting.sources_data.keys() if i > 2)
    sources_sel_probabilites = [1/x for x in sources_sorted]
    sources_sel_probabilites /= np.array(sources_sel_probabilites).sum()

    entity_to_group_size = dict()
    for index, row in setting.original.iterrows(): 
        flip_coin = random.uniform(0.0, 1.0)
        add_in_sources_count = 0
        if (structure_level)>flip_coin : 
            #get a power low distribution
            add_in_sources_count = np.random.choice(sources_sorted, 1, p=sources_sel_probabilites)[0]

        else: add_in_sources_count = 2

        entity_to_group_size[row[setting.id_attr]] = add_in_sources_count
        rng = default_rng()
        sources_to_add_row = rng.choice(list(setting.sources_data.keys()), add_in_sources_count, replace=False)
        for source_to_add_row in sources_to_add_row:
            setting.sources_data[source_to_add_row] = setting.sources_data[source_to_add_row].append({setting.id_attr: row[setting.id_attr]}, ignore_index=True)

    return entity_to_group_size

def print_cc_size_distribution(setting):
    #we keep the convention that same entities have the same id across data sources
    all_ids = list()
    for source_id in setting.sources_data:
        all_ids.extend(setting.sources_data[source_id][setting.id_attr].values)
    all_ids.sort()
    frequency_of_dist_ids = [len(list(group)) for key, group in groupby(all_ids)]
    group_sizes = Counter(frequency_of_dist_ids)
    print("Multi-source task cluster size distribution after setting the structuredness level")
    print(group_sizes)
    plt.bar(group_sizes.keys(), group_sizes.values())
    plt.xticks(np.array(list(set(frequency_of_dist_ids))))
    plt.show()
    #plt.savefig('%s.pdf' % (path+'cc_distribution'), bbox_inches='tight', format='pdf')
    #plt.savefig('%s.png' % (path+'cc_distribution'), bbox_inches='tight', format='png')