# Quora Duplicate Question Pair Analysis

### Steps to run the project
* Install miniconda https://docs.conda.io/en/latest/miniconda.html
* Download the quora_project folder. The folder contains four files and project_artifacts empty folder.
* Download Quora data set from https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs. Extract and add it to this project_artifacts folder.
* Download the glove embedding (glove.6B.300d.txt) from http://nlp.stanford.edu/data/glove.6B.zip and place it in the project_artifacts project folder. 
* Update project_folder config in config.py with correct path to this project_artifacts folder
* The environment.yml file contains all the necessary libraries to run the project.
* Run the following command from mini conda prompt to create conda environment to run the package
	* _conda env create -f <path to the environment.yml file>_
* Now this conda environment can be used to run the python files associated with two tasks.
	* _quora_duplicate_question_answer.py – Task 1_
	* _quora_question_cluster_analysis.py – Task 2_
* From the command prompt which is currently running the new conda environment, run the python files one after the other to execute the tasks
	* _python <filename.py>_
