# NLP701_assignment2_subtask1

**SemEval Team name:** Malak

Two experiments were conducted: 

1. The first experiment involved using the finetuned model XLM-Roberta.
2. To improve the results, which were likely affected by the small datasets, the second experiment used GPT-4 Mini for inference in zero-shot and few-shot configurations.

The code for each experiment is organized under the respective model name’s method. The prediction results are stored in a folder called `results`, with separate subfolders for each model. XLM-Roberta was run using Kaggle's GPU100 resources (https://www.kaggle.com/), while GPT-4 Mini inferencing was conducted via [Azure's API](https://ai.azure.com/).

### Installation

Run the following commands to set up the conda environment and install the necessary dependencies:

<pre>
<code>
  conda create --name assignment2NLP python=3.9
  conda activate assignment2NLP
  pip install -r requirements.txt
</code>
<button onclick="copyToClipboard(this.previousElementSibling.innerText)"></button>
</pre>

# XLM-Roberta-based Multilingual Inference

This repository contains the experimental setup and results for evaluating the performance of XLM-Roberta (XLM-RoBERTa-base) on a multilingual dataset across different inference tasks. The goal is to assess the model's ability to perform in Zero-shot and Few-shot learning settings for multiple languages, specifically English, Hindi, Bulgarian, and Portuguese. The evaluation metrics include Exact Match Ratio (EMR), Micro Precision (P), Micro Recall (R), Micro F1 Score (F1), and Main Role Accuracy.

## Overview

In this experiment, XLM-Roberta was tested on multiple languages to evaluate its performance in a multilingual setting. The task was evaluated on the following languages:
- English
- Hindi
- Bulgarian
- Portuguese

## Setup

### Requirements:
1. Python 3.7+
2. PyTorch
3. Hugging Face Transformers library
4. Datasets for English, Hindi, Bulgarian, Portuguese, and Russian
5. Access to GPU resources 

### Installation:
To run this experiment, clone this repository and install the required dependencies:

<pre>
<code>
  git clone https://github.com/your-username/xlm-roberta-multilingual.git
  cd xlm-roberta-multilingual
</code>
<button onclick="copyToClipboard(this.previousElementSibling.innerText)"></button>
</pre>

# GPT Inferencing

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

<pre>
<code>
  git clone https://github.com/your-username/gpt-zero-shot-few-shot.git
  cd gpt-zero-shot-few-shot
  pip install -r requirements.txt
</code>
<button onclick="copyToClipboard(this.previousElementSibling.innerText)"></button>
</pre>

### Microsoft Azure account
A Microsoft Azure account is required to set up and run the experiments in this project. Once you have set up your account, visit the [Azure AI portal](https://ai.azure.com/) to deploy your model.

You will need the **ENDPOINT URL** and **API key** for your deployed model. Add them to your environment variables: 
<pre>
<code>
  # prepare your Endpoint URL and API key (for linux)
  export ENDPOINT_URL="{ENDPOINT-URL}"
  export API_KEY="{API-KEY}"

  # prepare your Endpoint URL and API key (for windows)
  set ENDPOINT_URL="{ENDPOINT-URL}"
  set API_KEY="{API-KEY}"
</code>
</pre>

---

## Acknowledgments

We would like to express our sincere gratitude to the SemEval team for their contributions to the development and organization of the datasets used in this project. Their dedication to advancing natural language processing research is truly appreciated.
