# tripleH - Project Overview


Hello, we are Year 1 CS Students from NTU! Welcome to our project for the course : Data Science and Artificial Intelligence 

Our project is based on IMDB 5000 dataset found on [kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

We wonder, **"If we were movie directors, how could we maximise our success rates?"**

## Step 1: Looking at the Dataset

|index|0|
|---|---|
|color|object|
|director\_name|object|
|num\_critic_for_reviews|float64|
|duration|float64|
|director\_facebook_likes|float64|
|actor\_3_facebook_likes|float64|
|actor\_2_name|object|
|actor\_1_facebook_likes|float64|
|gross|float64|
|genres|object|
|actor\_1_name|object|
|movie\_title|object|
|num\_voted_users|int64|
|cast\_total_facebook_likes|int64|
|actor\_3_name|object|
|facenumber\_in_poster|float64|
|plot\_keywords|object|
|movie\_imdb_link|object|
|num\_user_for_reviews|float64|
|language|object|
|country|object|
|content\_rating|object|
|budget|float64|
|title\_year|float64|
|actor\_2_facebook_likes|float64|
|imdb\_score|float64|
|aspect\_ratio|float64|
|movie\_facebook_likes|int64|


## Step 2: Choosing our variables 

Our ultimate goal is to find out what we should focus on to make our movie successful.

We define success based on: 
- monetary gains (read **Challenges Faced** section)
&ensp a. `gross`
&ensp b. `profit margin` 
- ratings 
&ensp a. `imdb_score`
- popularity
&ensp a. `num_critic_for_reviews`
&ensp b. `num_voted_users`
&ensp c. `movie_facbook_likes`


## Step 3: Our hypotheses 
After doing Univariate EDA, and Bivariate EDA, we have chosen particular variables as our potential predictors of success. These are: 


## Step 4: Some Interesting Questions 
- Does the length of the title affect movie success? 
- Are there any types of words that we should include into our movie title to increase our chances of success? 






