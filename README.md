# Neural Network Cycle Count Prediction Model 

## Abstract
Machine learning is a relatively new technology that is being used in almost every industry today. However, one of the fields that fail to take advantage of its power is the processor design realm. In order to begin to bridge the gap between machine learning and processor design, we developed a machine learning model that predicts the performance of a CPU pipeline design using a simulator. To do so, we went through the process of collecting and generating data and developing a long short term memory (LSTM) model that predicts the total number of cycles taken for a given basic block. This model provided some accuracy using TensorFlows accuracy metric for models, but can be further improved upon using a more complex and optimized LSTM model, or modifying the way the data is being preprocessed. 

## Background
Research into using machine learning for computer architecture and its components is a new topic of study, but can inevitably be utilized to understand, research, and optimize implementations of microprocessors to its full potential. Machine learning has been changing every industry for the last decade, and will eventually also transform the way processors are designed and tested. For this reason, we decided to work on a project that coupled machine learning with CPU pipelines by creating a deep neural network to estimate the total number of execution cycles. As a result, we are attempting to advance the exploration and connection between computer architecture and machine learning. 

## Problem
CPU cycles and execution time are both very important metrics in determining the performance of a processor. Currently, benchmark computing tests that run a set of programs through a pipeline are used to determine a CPU’s performance because simply analyzing a computer's specifications is a difficult and inaccurate way to judge its true capabilities. Furthermore, while comparing CPUs by their specifications allows you to analyze the tradeoffs between each type of individual metric, there is still no concrete answer for which CPU will run faster and perform better for the system it is being used for without actually testing it. However, using a machine learning model to accurately predict CPU performance would make it easier and less costly to implement, test, and evaluate the performance of a microprocessor. If such a model existed, chip designers, companies, and researchers could determine whether their new innovative architectures and technologies were performing well without actually having to test the designs themselves; rather, the machine learning model would give a reasonable estimate of the CPU simulator’s capabilities and limitations.

## Architectural Solution
Our solution to this problem of expensive testing and implementation was to create a machine learning model that predicts the total numbers of cycles taken for a given CPU. The model would be trained on a large set of programs in assembly that were each mapped to the total cycle count associated with the CPU simulator it was run through. The ISA would be limited to RISC-V and the dataset would only contain basic blocks to eliminate looping, branching, and assembly code bugs that could occur in large scale programs. As a result of narrowing down the ISA specifications and getting rid of jump instructions, we would yield more accurate results for a specific ISA because the amount of instructions the model needs to recognize would be far less and the total cycle count would be directly related to the length of the basic block. Once trained on a specific CPU, the model would then be able to accurately output the total cycle count given a RISC-V program. 

The model itself consists of several layers. The first LSTM layer takes in a vector of mapped tokens with each index being an instruction from the given basic block. The LSTM output was fed into a Rectifier (ReLU) linear layer which removed all negative values from the first LSTM layer output of the model using a piecewise function. The second layer of the model is an LSTM layer that allows us to generate predictions based on the previous instructions that were executed in the basic block. This layer was then fed into two subsequent linear layers made up of a Softsign and a Scaled Exponential Linear Unit (SeLU) to flatten and normalize the prediction values from the LSTM. The output of this model was an array of cycle counts for each individual instruction. After summing all elements from this output array, the model was able to provide an estimate for the total cycle count per basic block. Figure 1 below demonstrates how the model processes one specific instruction of a basic block.

![alt text](http://url/to/img.png)


## Method of evaluation

## Implementation progress

## Results

## Installation
$ git clone https://github.com/abbywysopal/cs254_project

## Compile and Run
$ python run.py

Generate dataset to feed into ML model

$ python LSTM_model.py

Create and train ML model
