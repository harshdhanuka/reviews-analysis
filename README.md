## reviews-analysis
This repository contains my work on a detailed analysis of a Women's Clothing company reviews, considering Clustering, Sentiment Analysis, Exploratory Analysis, Text Analysis, and various Rating Predictor Models. This was completed as part of a class group submission. All credits to be given to the owner.

### Description of the Data
This dataset is about the buyer’s reviews of woman’s clothing in e-commerce. The dataset contains the details of customer reviews based on 10 different variables. Along with the review written by the customer, there is demographic information as well. For example, the age of the customer. The specificity of the reviews i.e. the dimensions the customer reviews the clothing item (recommend or not, division name, department name), will make it easier to analyze the data on different dimensions.

This dataset includes 23,486 rows and 10 feature variables.

Data Source: Kaggle CC0: Public Domain, Owner - https://www.kaggle.com/nicapotato  
Data Link: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

### Description of the Variables
1.	Clothing ID: 
Clothing ID is an integer categorical variable that refers to the specific piece being reviewed. 

2.	Age: 
A positive integer variable of the reviewers age. The data contains ages of reviewers which range from 18 to 99-year-old women.

3.	Title: 
Title is a string variable for the title of the review. Titles of review are like “Perfect Fit!”, “Runs Big”, “Love it”. The title characters vary from 0 to 12 words. From the title, it is easier to figure out what the review text will be about.

4.	Review Text: 
The review text is string variable for the review body. The review text is a detailed description of the Title. In the review text the reviewer is explaining the title e.g. 
Review Title: Zipper is weird
Review Text: “I ordered the dress on-line. this dress looked pretty, the material was soft and comfortable, the length was perfect, but the zipper was totally out of place. it bulged out in a weird way which made it look like the dress was torn. I had to return the dress. I normally don't see a problem at this level in retailer clothes.”

5.	Rating: 
Rating in the dataset is positive ordinal integer variable for the product score granted by the customer from 1 Worst, to 5 Best. Based on the review, the customer allocates a rating. The table below contains the number of customers that gave a particular rating integer out of the total 23,486 customers:
 
6.	Recommended IND: 
Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended. In the dataset, out of the total customers, 4,172 customers did not recommend the product and 19,314 recommended the product.

7.	Positive Feedback Count: 
Positive Integer documenting the number of other customers who found this review positive. The positive feedback count ranges from 0 to 122. Hence, the maximum number of positive feedback count on a review is 122, the minimum being 0.

8.	Division Name: 
Categorical name of the product high level division. There are 3 divisions; general, general petite and intimates. The table below contains the details of division and the number of reviews in a particular division:

9.	Department Name: 
Department name is the categorical name of the product department name. There are 6 departments namely bottoms, dresses, intimate, jackets, tops and trend. 
 
10.	Class Name:
Class name is the categorical name of the product class name. There are a total of 20 classes for the company.

### Questions/Problem being Addressed:
By processing and analyzing the data, we developed graphical representations of the characteristics of the text reviews. Further, we built a predictive model, by splitting our data set into train and test and predicted the rating of the clothing items, on a scale of 1-5. The prediction of the rating of the clothing items will help the e-commerce website make marketing and sales strategy to improve the performance and increase the revenues. We also performed sentimental analysis and applied text mining techniques.
