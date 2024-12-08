# NLP701_assignment2_subtask1

SemEval Team name: Malak


2 experiments were conducted. Forst with the finetuned model xlm-roberta then to improve the results which were probably caused by the small datasets, we inferenced gpt4o mini using zero and few shot. The code for each of the 2 experiments is found under the model name's method. The prediction results are under the folder called results for each model. XLM-roberta used Kaggle's GPU100 resources while GPT4o mini inferencing used Azure's API access.

# XLM-Roberta-based Multilingual Inference

This repository contains the experimental setup and results for evaluating the performance of XLM-Roberta (XLM-RoBERTa-base) on a multilingual dataset across different inference tasks. The goal is to assess the model's ability to perform in Zero-shot and Few-shot learning settings for multiple languages, specifically English, Hindi, Bulgarian, and Portuguese. The evaluation metrics include Exact Match Ratio (EMR), Micro Precision (P), Micro Recall (R), Micro F1 Score (F1), and Main Role Accuracy.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Experiment Setup](#experiment-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview

In this experiment, XLM-Roberta was tested on multiple languages to evaluate its performance in a multilingual setting. The inference was performed using the following configurations:
- **Zero-shot**: The model was tested without any task-specific training or prompting.
- **Zero-shot Prompted**: The model was prompted with additional context or instructions to guide the inference.
- **Few-shot**: The model was provided with a small number of labeled examples (200 and 512 examples).
- **Few-shot Prompted**: The model received prompted input along with a few labeled examples.

The task was evaluated on the following languages:
- English
- Hindi
- Bulgarian
- Portuguese

## Setup

### Requirements:
1. Python 3.7+
2. PyTorch
3. Hugging Face Transformers library
4. Datasets for English, Hindi, Bulgarian, and Portuguese

### Installation:
To run this experiment, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/xlm-roberta-multilingual.git
cd xlm-roberta-multilingual
conda create assignment2NLP
conda activate assignment2NLP
pip install -r requirements.txt



## Overview

This experiment evaluates GPT models in zero-shot and few-shot learning configurations. Specifically, the performance of GPT-based models was assessed using the following methods:
- **Zero-shot**: The model was tested without any task-specific training or prompting.
- **Zero-shot Prompted**: The model was prompted with additional context or instructions to guide the inference.
- **Few-shot 200**: The model was provided with 200 labeled examples.
- **Few-shot 512**: The model was provided with 512 labeled examples.
- **Few-shot 512 Prompted**: The model received prompted input along with 512 labeled examples.

The task was evaluated on multiple languages (e.g., English), and the following metrics were used to assess performance: Exact Match Ratio, Precision, Recall, F1 Score, and Main Role Accuracy.

## Setup

### Requirements:
1. Python 3.7+
2. OpenAI GPT API (for GPT-3 or GPT-4)
3. Datasets for evaluation (task-specific datasets)
4. Other dependencies from `requirements.txt`


```bash
git clone https://github.com/your-username/gpt-zero-shot-few-shot.git
cd gpt-zero-shot-few-shot
pip install -r requirements.txt
