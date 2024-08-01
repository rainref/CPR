# C-P-R

A comprehensive guide to our project's structure, including the data generation from Language Model (LLM), the refinement process, and the evaluation of results.

## Directory Structure

The root directory includes three main folders:

### Dataset Folder
Due to size constraints, only the test set is shown. The complete dataset is available at [https://github.com/mciniselli/T5_Replication_Package](https://github.com/mciniselli/T5_Replication_Package).

### Generation Folder

This folder contains scripts to generate data from LLM and to evaluate the results. Each file in this directory serves an independent function:

- `gen_test.py`: Generates test set results from the GPT-4o API.
- `gen_train.py`: Generates training set results from the GPT-4o API.
- `gen_incoder.py`: Generates data from Incoder.
- `gen_santa.py`: Generates data from SantaCoder.
- `regen.py`: Performs a C-P cycle to regenerate cases with grammatical errors.
- `eval_cpr.py`: Evaluates the results files from the C-P-R process.

Each script is self-contained and responsible for a specific part of the data generation and regeneration pipeline.

### Refiner Folder

This folder is a key component of the complete project and includes the implementation, training, and prediction functionalities of Refiner. For detailed information on how to use Refiner and an explanation of its capabilities, please see the `ReadMe.md` file located within this folder.

## Getting Started

To get started with the project, clone the repository and navigate to the desired folder to run the operations you require.

Ensure you have all the necessary dependencies installed before running any script. You might need to have access to the GPT-3.5 API and any other models mentioned, like Incoder and SantaCoder.