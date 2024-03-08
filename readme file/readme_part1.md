## Interaction-Based data analysis

## Overview

This project focuses on clothing_fit dataset to understand customer preferences, behaviors, and the overall rental trend. The aim is to derive insights that could help in improving the service and tailoring it more closely to customer needs. this part of code is included in `Interaction-Based data analysis.ipynb`.

## Dataset

The dataset, named `renttherunway_final_data.json.gz`, consists of customer reviews and metadata related to clothing rentals. It includes various attributes such as fit, user ID, item ID, body type, category, height, size, age, review date, rating, weight, bust size, and review text.

## Data Preparation

### Cleaning and Transformation

- Store the unique user IDs from the dataset and Calculate the number of unique users.
- Define the columns to be analyzed.(fit', 'user_id', 'item_id', 'body type', 'category', 'height', 'size', 'age', 'review_date', 'rating', 'weight', 'bust size', 'review_text') 
  - **Height Conversion**: The height of users is converted from a string format (e.g., '5' 8"') to total inches.
  - **Weight Conversion**: The weight is transformed from a string (ending with 'lbs') to a numeric format in pounds.
  - **Data Type Conversion**: Fields such as age and size are converted to numeric types, and the review date is converted to a datetime format.
  - **Categorical Conversion**: Attributes like fit, item ID, body type, category, and bust size are converted to categorical data types for efficient analysis.\

### Saving Clean Dataset

The cleaned dataset is saved to `caius_data_clean.csv` for further analysis, avoiding the need to repeat the cleaning process.

## Analysis

### Basic Statistics

- **Data Records Count**: The total number of records in the dataset.
- **Missing Values Count & Percentage**: Counts and percentages of missing values for each attribute.
- **User and Item Analysis**: Identification of unique users and items, along with the average number of clothes rented per user.
- **Purchase Distribution**: Analysis of how many clothes users tend to rent, summarized in a distribution graph.

### Insights and Visualization

- Visualize tution of the number of clothing rentals. Label any values exceeding eight as "other," displaying them as a percentage on the chart, highlighting the rental behavior across different user segments.

  ![image-20240229212809811](/Users/yeqiuhan/Library/Application Support/typora-user-images/image-20240229212809811.png)

## Conclusion

The project provides a comprehensive analysis of the Rent the Runway dataset, offering insights into customer rental patterns and preferences. This analysis can be instrumental for business strategy and enhancing customer satisfaction.