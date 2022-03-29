# tripleH - Project Overview


Hello, we are Year 1 CS Students from NTU! Welcome to our project for the course : Data Science and Artificial Intelligence 

Our project is based on IMDB 5000 dataset found on [kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

We wonder, **"If we were movie directors, how could we maximise our success rates?"**

## Foreword
- Sections/ Texts that are highlighted are additional/ extra materials that we learnt outside of our course. 


## Content Section 
- Introduction 
- Steps: 
  - 1 :  [Looking at the Dataset](#step-1-looking-at-the-dataset)
  - 2 : [Choosing our Variables](#step-2-choosing-our-variables)
  - 3 : [Our Hypotheses](#step-3-our-hypotheses)
- [Challenges Faced](#challenges-faced)

## Step 1: Looking at the Dataset
|variable|dtype||variable|dtype|
|---|---|---|---|---|
|color|object||actor\_3_name|object|
|director\_name|object||facenumber\_in_poster|float64|
|num\_critic_for_reviews|float64||plot\_keywords|object|
|duration|float64||movie\_imdb_link|object|
|director\_facebook_likes|float64||num\_user_for_reviews|float64|
|actor\_3_facebook_likes|float64||language|object|
|actor\_2_name|object||country|object|
|actor\_1_facebook_likes|float64||content\_rating|object|
|gross|float64||budget|float64|
|genres|object||title\_year|float64|
|actor\_1_name|object||actor\_2_facebook_likes|float64|
|movie\_title|object||imdb\_score|float64|
|num\_voted_users|int64||aspect\_ratio|float64|
|cast\_total_facebook_likes|int64||movie\_facebook_likes|int64|


## Step 2: Choosing our variables 

Our ultimate goal is to find out what we should focus on to make our movie successful.

We define success based on: 
- monetary gains (read [**Challenges Faced**](#challenges-faced) section)
   - `gross`
   - `profit margin` 
- ratings 
   - `imdb_score`
- popularity
   - `num_critic_for_reviews`
   - `num_voted_users`
   - `movie_facbook_likes`


## Step 3: Our hypotheses 
After doing Univariate EDA, and Bivariate EDA, we have chosen particular variables as our potential predictors of success. These are: 


## Step 4: Some Interesting Questions 
- Does the length of the title affect movie success? 
- Are there any types of words that we should include into our movie title to increase our chances of success? 


## Challenges Faced 
Through research, we found that:
1. The type of gross is not standardised, e.g: 
2. Extremely skewed data
   - Facebook likes data:
     - solution: use log transform 
   





