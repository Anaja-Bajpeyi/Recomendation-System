# Recomendation-System

The recommendation system made for FoMC gives recommendation based on high number of purchases as well as recommendation to individual customers based on their purchase history. 
The recommendation system is developed using Anaconda Navigator that contains porting for all the popular python libraries that can be used in data science.  
The code is written in python using Jupyter Notebook’s editor from anaconda navigator and compiled using Jupyter Notebook’s terminal. The code is written in 4 files namely: R_sort_data, R_popular, R_personalised, Recommendation_system. The Recommendation_system code is the main code that has all the three codes imported along with the library numpy, pandas, time and sklearn. 

# Flow of the program:
The Recommendation_system program calls a function named ‘runrecommender’ to which one argument is passed. 
The argument is the csv file that is given when running the code on terminal. 
The csv file has data in columns first name, last name, sku and name of the item. 
The data needs to be arranged by joining the firstname and lastname together in one column and sku in another column. 
To process it further the data is then arranged as one column containing firstname + lastname and other all sku’s the customer has purchased separated by a delimiter(‘ | ’).This is done by calling R_sort_data by passing the file name to it for sorting the data to perform recommendation on it. 

# Popularity-Model 
The Popularity model takes the most popular items for recommendation.
These items are products with the highest number of sells across customers.
The Popularity model doesn’t give any personalizations, it only gives the same list of recommended items to every user by ranking list of items by their purchase count.
It can use context, user features, item features and purchase history. 

# Item-similarity based nearest neighbor recommendation:
This model Recommends items that are similar to the items the user bought. 
Similarity is based upon co-occurrence of purchases. The model gives personalized recommendation to every user. 
The co-occurrence matrix is constructed using Jaccard index. 
The Jaccard index, also known as Intersection over Union and the Jaccard similarity coefficient is a statistic used for comparing the similarity and diversity of sample sets.
