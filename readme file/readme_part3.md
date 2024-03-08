## Interaction-Based data analysis

## Overview

This project focuses on clothing_fit dataset to understand customer preferences, behaviors, and the overall rental trend. The aim is to derive insights that could help in improving the service and tailoring it more closely to customer needs. this part of code is included in `Interaction-Based data analysis.ipynb`.

## Dataset

The dataset, named `renttherunway_final_data.json.gz`, consists of customer reviews and metadata related to clothing rentals. It includes various attributes such as fit, user ID, item ID, body type, category, height, size, age, review date, rating, weight, bust size, and review text.

## Fit Distribution

Use different statistical charts based on previously completed data tables and analyze their corresponding relationships.

**Draws Histograms:** For each feature in the dataset df (‘height’, ‘age’, ‘body type’, ‘size’, and ‘category_grouped’), a histogram is drawn. The histogram shows the distribution of counts for different ‘fit’ categories across the various values of that feature. For example, the line sns.histplot(data=df, x='height', hue='fit', multiple='stack', bins=30) draws a histogram showing the distribution of counts for different ‘fit’ categories across various ‘height’ values.

**Calculates Percentages:** For each feature, the percentage of counts for different ‘fit’ categories out of the total counts across the various values of that feature is calculated. For instance, the line frequency_data['percentage'] = (frequency_data['count'] / frequency_data['total_count']) * 100 calculates the percentage of counts for different ‘fit’ categories out of the total counts across various ‘height’ values.

**Draws Line Plots:** For each feature, a line plot is drawn. The line plot shows the distribution of percentages for different ‘fit’ categories across the various values of that feature. For example, the line sns.lineplot(data=frequency_data, x='height', y='percentage', hue='fit') draws a line plot showing the distribution of percentages for different ‘fit’ categories across various ‘height’ values.

These charts can help us understand the relationship between the ‘fit’ category and these features, such as which values of ‘height’, ‘age’, ‘body type’, ‘size’, and ‘category_grouped’ are more likely to correspond to a certain ‘fit’ category. This could be very useful for further data analysis and model training. These charts also allow us to visually see the distribution of the data, so we can better understand the characteristics of the data. For example, we can see which values of ‘height’, ‘age’, ‘body type’, ‘size’, and ‘category_grouped’ have a particularly large number of counts for a certain ‘fit’ category, or a particularly small number of counts. 



