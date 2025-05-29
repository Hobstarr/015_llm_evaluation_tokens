### User Guide
#### Outline
This repository contains scripts and datasets for running an evaluation to understand what level of output can be achieved for the llm. Using the `databricks-dolly-15k` dataset, and the `gpt2` model, we carry out repeated completions with the aim of understanding performance.

#### Installation
Clone the repository and then install the following requirements:

llm libraries:  
`pip install transformers torch pydantic`

plotting libraries:  
`pip install matplotlib seaborn`

#### Usage
`python main.py` will run the script, this will feed in the instruction and context from the dataset and then invoke the model to create 5 completions. The time taken is monitored, and the results of each prompt evaluation stored in the `results.jsonl` file. Included here is an example run where 1279 prompts have been evaluated.

`python post_processing.py` will create a distribution plot of the tokens_per_second achieved by the model during the evaluation, and then print a summary of the evaluation including the number of prompts, number of completions, time taken, total tokens produced, and the average tokens per second achieved over all prompts. It will save the file as `results.jpg`. An example output plot is included here.