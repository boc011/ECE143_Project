# Data Visualization

## Overview

All data comes from the file "dataset renttherunway_final_data". We implement matplotlib.pyplot and seaborn libraries for all data visualizations in the file "Interaction-Based data analysis" with an "ipynb" extension.

## Histogram

- **Height Histogram**: We standardized the height column to reflect all heights in inches and checked the distribution of inches in height.
- **Size Histogram**: We display the size column to check the distribution of chosen cloth sizes.
- **Age Histogram**: We display the age column to check the distribution of customers' ages, noticing some fake info or mistyping that leads to outliers, for example, 'age': 116.
- **Rating Histogram**: Ratings range from 0 to 10. We display the rating column to check the overall rating distribution, regardless of different cloth types, noticing it only receives even inputs.
- **Weight Histogram**: We standardized the weight column to reflect all weights in lbs and checked the distribution of lbs weights.

## Correlation Matrix

- **Features**: 'user_id', 'item_id', 'height', 'size', 'age'
- We display the linear correlation between chosen features.
- We observe potential linear relationships between "height and size" and "age and size".

## Pie Charts

- **Distribution of Body Type**: We display proportions of different body types, highlighting the most and least common body types among customers.
- **Distribution of Top 6 Categories and Other**: We display proportions of cloth categories, highlighting the most and least popular garments among customers.

## Body Type Distribution by Fit Category

- **Plot One**: We display the actual amount of each Fit category for different body types.
- **Plot Two**: We display the percentage of each Fit category for different body types, presenting it in such a way to determine which body types may lead to a higher probability of picking the wrong size.
