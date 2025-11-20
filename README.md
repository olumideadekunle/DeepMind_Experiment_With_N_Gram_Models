## N-Gram Language Model Experiment
Project Overview
This project explores the fundamentals of N-gram language models, demonstrating their construction, application in text generation, and inherent limitations, particularly data sparsity. Using the Africa Galore dataset, we built and compared bigram and trigram models to understand how context length impacts model performance and coherence.



## Table of Contents
Introduction
Methodology
Key Findings
Model Comparison (Bigram vs. Trigram)
Limitations of N-gram Models
Conclusion and Next Steps
Usage
Installation
References





## Introduction
N-gram models are foundational in natural language processing (NLP) for estimating the probability of word sequences. This lab focused on building such models to predict the next token in a sequence, a core task in text generation. The primary objective was to gain a practical understanding of how n-grams capture language patterns and to identify the challenges associated with them.

## Methodology
Dataset: We utilized the Africa Galore dataset, a collection of synthetically generated paragraphs focusing on African culture, history, and geography.
Tokenization: Text was tokenized into word-like units using a simple space tokenizer.
N-gram Counting: Functions were developed to extract and count n-grams (unigrams, bigrams, trigrams) from the tokenized dataset.
Probability Estimation: Conditional probabilities were calculated based on these n-gram counts.


## Text Generation: An iterative process was implemented to generate new text by sampling the next token based on the estimated probabilities, given a preceding context.
Key Findings
N-gram Generation and Counting: Successfully implemented functions to generate and count n-grams of varying lengths.
Significant Data Sparsity: A high degree of data sparsity was observed, with 99.95% of possible bigrams and 99.98% of possible trigrams having zero occurrences in the dataset. This highlights a fundamental challenge for n-gram models.
Model Construction: Successfully built and tested n-gram language models that convert counts into conditional probabilities.


## Model Comparison (Bigram vs. Trigram)
Feature	Bigram Model	Trigram Model
Coherence & Grammatical Correctness	tended to produce less coherent and grammatically awkward continuations.	Generally produced more sensible and grammatically correct continuations due to considering a larger context (two preceding words). Example: "Jide was hungry so she went looking for a cup of Kenyan chai, rich and peanut-infused, or spicy" (more coherent) vs. "Jide was hungry so she went looking for the day in the bread are cooked with warm temperatures" (less coherent).
Data Sparsity	Lower sparsity (99.95% zero counts).	Higher sparsity (99.98% zero counts) makes it more prone to failing to find continuations for unseen contexts.
Failure Rate	is less prone to failing to find continuations.	More prone to failing (e.g., KeyError for unseen contexts like trigram_model['Their name']).
Data Sparsity Visualization
import matplotlib.pyplot as plt

labels = ['Bigram Model', 'Trigram Model']
sparsity_percentages = [99.94884329805882, 99.9752926948483] # Values from kernel state
colors = ['skyblue', 'lightcoral']

plt.figure(figsize=(8, 6))
plt.bar(labels, sparsity_percentages, color=colors)

plt.title('Data Sparsity in N-gram Models')
plt.xlabel('N-gram Model Type')
plt.ylabel('Percentage of Zero Counts (%)')

for i, percentage in enumerate(sparsity_percentages):
    plt.text(i, percentage + 0.5, f'{percentage:.2f}%', ha='center', va='bottom')

plt.ylim(0, 100)
plt.show()
Limitations of N-gram Models
Data Sparsity: The primary limitation, especially for higher-order n-grams, leading to zero probabilities for unseen sequences and an inability to generate continuations for many contexts.
KeyError Incidents: Directly linked to sparsity, where attempting to access probabilities for unobserved contexts results in errors.
Lack of Long-Range Dependencies: N-gram models only consider a fixed, local context, making them ineffective at capturing longer-range linguistic dependencies.
Fixed Context Size: The 'n' value is rigid, limiting adaptability to different linguistic phenomena.
Conclusion and Next Steps
N-gram models offer a simple yet powerful approach to language modeling, capable of capturing local word patterns. However, their inherent vulnerability to data sparsity limits their applicability, particularly with larger 'n' values or smaller datasets.

##Future Improvements:

Smoothing Techniques: Implement smoothing algorithms (e.g., Laplace smoothing, Kneser-Ney smoothing) to address zero probabilities for unseen n-grams and improve model robustness.
Back-off Models: Explore combining models of different 'n' values (e.g., backing off to a bigram model if a trigram is unseen) to provide more consistent predictions.
Comparison with Advanced Models: Investigate more sophisticated models, such as neural network-based language models (e.g., LSTMs, Transformers), which are better equipped to handle long-range dependencies and data sparsity.
Usage
To use the N-gram models developed in this notebook:

##Run all cells in sequence to define the necessary functions and build the models.
Utilize generate_next_n_tokens function with your desired prompt, n value (for bigram or trigram), and num_tokens_to_generate to create new text continuations.
# Example usage:
prompt = "Jide was hungry so she went looking for"
# For a bigram model (n=2)
generate_next_n_tokens(n=2, ngram_model=bigram_model, prompt=prompt, num_tokens_to_generate=10)

## For a trigram model (n=3)
generate_next_n_tokens(n=3, ngram_model=trigram_model, prompt=prompt, num_tokens_to_generate=10)
Installation
This project runs in a Google Colab environment. All necessary packages are installed via pip commands within the notebook (e.g., pandas, ai_foundations). Ensure your Colab environment is set up and all cells are run sequentially.

## References Credit To:
Ronen Eldan and Yuanzhi Li. 2023. Tiny Stories: How Small Can Language Models Be and Still Speak Coherent English. arXiv:2305.07759. Retrieved from https://arxiv.org/pdf/2305.07759.
