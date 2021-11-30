# ALMSER-GEN

This repository contains the code for reproducing the results of the paper "Active Learning for Multi-Source Entity Matching: How do the Characteristics of the Task Impact Performance?".

Due to the large size of files containing the data, we provide the generated multi-source matching tasks (584 MB) and benchmark tasks (103M) as zip files here:

http://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/almser_gen_data/

Each task, contains all necessary data for reproducing the results of the paper. Additionally, we provide in the results directory of every setting, the result files of ALMSER, ALMSERgroup and HeALER query strategies.
If you want to run any of the above strategies, please use the ALMSER-GB code located at:
https://github.com/wbsg-uni-mannheim/ALMSER-GB.git

# INSTALLATION OF REQUIRED PACKAGES:

The project runs with Python 3.7. 
Please install the packages found in the requirements.txt file.

# Functionalities ALMSER-GEN repo
You can find the following functionalities in this repository:
# 1. ALMSERgen code generator
Navigate to ALMSERgen/ALMSERgen_INIT.ipynb notebook. Run ALMSERgen as follows:

Step 1: Define configuration
  sources: amount of sources to be generated
  id_attr: identifying attributes of the domain
  main_path : path to output the generated tasks
  vpo_values : amount of groups of data sources with the same value pattern. Don't forget to normalize it before analysis!
  eo_values : entity overlap levels
  vh_values : value heterogeneity levels

Step 2: ALMSERgen will generate the tasks (sources and feature vectors)

Step 3: Along with the tasks the following files are generated:
  - train/test split using the cc of the complete graph
  - unsupervised results
  - passive learning results
  - distribution of connected components sizes (bar chart)
  - naive transfer learning heatmap as csv
  - profiling information on the task

# 2. Analysis of continuum tasks
Navigate to continuum_tasks_analysis.ipynb notebook. What you can do here?

1. Load continuum tasks results files (provided in the data/msmt_continuum.zip --see details above to download --> Please unzip the folder before using.)
2. Calculate winning method per task
3. Plot analysis results for patterns P1-P4

# 3. Analysis of benchmark tasks
Navigate to benchmark_tasks_analysis.ipynb notebook. What you can do here?

  1. Gets identifying attributes [1] per task (Benchmark tasks located in benchmark_tasks.zip --see details above to download --> Please unzip before using). 
  2. Produces results table
  3. Calculates profiling dimensions per task

  [1] Primpeli, Anna, and Christian Bizer. "Profiling entity matching benchmark tasks." Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020.

