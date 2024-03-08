# Text-Based Data Analysis on RentTheRunway Dataset

## Overview

This project focuses on analyzing the RentTheRunway dataset to predict the appropriateness of clothing based on customer reviews. The main goal is to classify reviews into three categories: fit, large, and small. This analysis is critical for understanding customer satisfaction and improving product recommendations.

## Dataset

The dataset comprises reviews from 105,508 users, totaling 192,462 records. Each record includes a `user_id`, `review_text`, and a `fit` attribute indicating the clothing fit as either "fit", "large", or "small". The dataset is stored in a compressed JSON format file named `renttherunway_final_data.json.gz`.

## File Structure

- `data_analysis.ipynb`: Jupyter Notebook containing the analysis code.
- `renttherunway_final_data.json.gz`: The compressed dataset file.
- `fit.png`, `large.png`, `small.png`: Word cloud images generated for each clothing fit category.

## Installation

Before running this project, ensure you have the following third-party modules installed:

- pandas
- matplotlib
- wordcloud

You can install these modules using pip:

```bash
pip install pandas matplotlib wordcloud
