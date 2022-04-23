# How to Maximise Movie Success - Project Overview 

## Contributors: tripleH group 
* Sand***h***iya Sukumaran [@sandhiyaaa](https://github.com/imaginaryBuddy/imdbMoviesDSAI/commits?author=sandhiyaaa)
* Ko***h*** Zhi En [@zex3](https://github.com/zex3)
* Yap S***h***en Hwei [@imaginaryBuddy](https://github.com/imaginaryBuddy)
</n>

## Codes are in: 
1. [imdbFullAnalysis](https://github.com/imaginaryBuddy/imdbMoviesDSAI/blob/main/imdbFullAnalysis.ipynb)

2. [answeringInterestingQuestions](https://github.com/imaginaryBuddy/imdbMoviesDSAI/blob/main/answeringInterestingQues.ipynb)

![source: bespeaking.com](https://www.bespeaking.com/wp-content/uploads/2019/09/Movie-vocab.jpg)
  

Our project is based on IMDB 5000 dataset found on [kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

<!---We wonder, **"If we were movie directors, how could we maximise our success rates?"** --->



## Content Section 
- [Introduction](#introduction)
  - Problem Statement 
  - Motivation
- Steps: 
  - 1 : [Looking at the Dataset](#step-1-looking-at-the-dataset)
    - Our Hypotheses 
  - 2 : [Data Extraction & Cleaning](#step-2-data-extraction-and-data-cleaning)
  - 3 : [Exploratory Data Analysis](#step-3-eda)
  - 4 : [Machine Learning](#step-4-machine-learning)
- [Interesting Questions](#interesting-questions)
- [The Big Conclusion](#the-big-conclusion)
- [Beyond Our Course](#beyond-our-course)
- [Limitations and Discussion](#limitations-and-discussion)
- [Workload Delegation](#workload-delegation)
- [Our Video](#our-video)
- [References](#references) 

# Introduction 
Have you ever wondered why some movies are more successful than others? If you're a movie director, you've came to the right place! If you are not a movie director, of course, you can still read on to find out more!
  ## Problem Statement 
   > Identify which features contribute to the success of a movie.

  ## Motivation 
   > Give directors a better estimation on how to maximize the success rate of their movie
  
  

# Step 1: Looking at the Dataset
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

There are a total of 28 variables. 

  ### Our hypotheses : 
  - Duration will not affect IMDB scores
  - Variables related to popularity will have positive correlation with IMDB score
  - Budget will affect IMDB score


# Step 2: Data Extraction and Data Cleaning 

  ## Train:Test:Validation
  We split our dataset into 80:20 ratio, then used the 80% as our Train Dataset to further divide to obtain our Train: Validate for our Machine Learning Models in 80 : 20 ratio. 
  
  
  ## Cleaning the Train Dataset
  
  1. Issue with `gross`
     - We found out that `gross` was not standardized, as the dataset contained different types of `gross` for each movie.  (e.g: opening week gross, US&Canada Gross etc.)
     - An example of disparities between the types of gross: 
      ![gross](https://user-images.githubusercontent.com/81760484/163833790-a99fc1ac-5a63-47be-a33e-7665f4d3f9fc.png)
  2. Issue with `budget` 
     - Different movies from different countries had different currencies for their budget. 
     - Since the proportion of movies from other countries (besides US) was quite small, we decided to drop them. 
     - We only used movies from USA.
     - Needed to standardize budget base on 2016 inflation rates in the US. 
  3. Null Values 
     - When # of null values are small for the variable, we dropped them. 
     - Otherwise, for numerical data, we replaced them with median in scenarios such as during Machine Learning. 
     - For categorical data, we dropped the rows. 
  4. Train : Validate : Test
     - We followed the Train : Validate : Test scheme  
     - Split Train:Test in 80:20 ratio 
     - Used Train as our EDA 
     - Further split Train into Train:Validate in 80:20 ratio for Machine Learning 
  5. Binning `imdb_scores` 
     - We wanted to observe the correlation not just in a numerical manner but also in a categorical manner. 
     - Besides, since we couldn't really find any strong linear correlation (as you will read later on), we figured that it would be beneficial to split `imdb_score` into categories. 

```python
  # Bins to categorise the imdb_score ranges

  # Multi bins
  imdb_bins = [0, 3, 5, 7, 10]
  imdb_labels = ["horrendous", "ok", "good", "very good"]

  # Binary bins
  # 6.5 = 1-6.5 (Bad) 10 = 6.6-10 (Good)
  bins = (2, 6.5, 10)

```

---
  > **TLDR:**
  > * we only used movies from the US, and standardised the budget based on 2016 inflation rates.
  > * used Train : Validate : Test scheme
  > * removed gross entirely due to inconsistency 
  > * we binned the `imdb_score` into categories, and tried out different bins. 
---
  

# Step 3: EDA
In this section, we will look at univariate and bivariate EDAs concerning more significant/ interesting variables. 


### Choosing our response variable 
We have chosen `imdb_score` as our main response variable, for simplicity purposes. Initally, we wanted to use `gross`, but due to disparities, we decided not to.  

### 1. director_name
these are the most frequently appeared directors. 

|director_name|count|
|---|---|
|Steven Spielberg|22|
|Woody Allen|18|
|Clint Eastwood|17|
|Spike Lee |15|
|Ridley Scott |15|
|Martin Scorsese|15|
|Steven Soderbergh|12|
|Renny Harlin  |12|
|Robert Zemeckis  |12|
 
It is interesting to note that Steven Spielberg is also one of directors from the [Top20 performing movies](#What-are-the-personalities-of-directors-of-top-performing-movies?)



### 2. num_critic_for_reviews 
- a large proportion of movies receive close to 0 num_critic_for_reviews. 
- there is no significant linear correlation bewteen `num_critic_for_reviews` and `imdb_score`
- the table below shows the movies sorted based on their `num_critic_for_reviews`, it does seem to show that `imdb_score` falls in a range of > 7.0 for these 20 movies. 

<img src="https://user-images.githubusercontent.com/81760484/164884443-c08e3f60-516c-4781-9359-0e914ef7c222.png" width= 40% height=auto>&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/81760484/164884505-347d6c89-c932-4525-a889-70b4a0d01d3b.png" width= 45% height=auto>

- to be fair, it may be that there is some sort of indication for imdb_score based on num_critic_for_reviews (as shown on the table), perhaps due to the large proportion of data receiving close to 0 reviews, we couldn't observe a linear correlation. 


### 3. duration 
**duration vs imdb_score**
- We binned the `imdb_score` into categories to form `score_cat`
- There seems to be slight correlation based on the boxplot between `duration` and `score_cat`
<!---- ![duration eda](https://user-images.githubusercontent.com/81760484/163723231-f0a5995b-82c9-4a50-ab68-696089f279b5.png)--->
![duration vs score](https://user-images.githubusercontent.com/81760484/163723346-65235c7f-2ffb-4581-93e4-261db7619378.png)

### 4. director_facebook_likes
- Was extremely right-skewed even after removing the outliers, which is not unexpected, since "success" depends on outliers. 
<!---- ![dir_facebook_likes_distribution](https://user-images.githubusercontent.com/81760484/164615445-95c745ba-65ae-448a-9e0b-a0a3ebbc021b.png)--->
- Due to the skew structure, we used log transform to visualise the data. 
- Distribution after log transform: 
- ![dir_facebook_likes_distribution_log](https://user-images.githubusercontent.com/81760484/164615674-2bdea061-8d18-4e8e-a916-abc810d63a92.png)
- Binomial distribution, suggesting that there may be two different "clusters". 

**director_facebook_likes vs imdb_score**

Although it can't be confirmed that there is a correlation between them, the boxplots shows that the median values of imdb_score do vary for the different categories. 

![dir_likes_vs_imdb_score](https://user-images.githubusercontent.com/81760484/164624848-2f4563c1-d2b9-4220-981d-ff776b5574e2.png)

However, we note that the "good" and "very good" categories had relatively larger numbers of outliers, that had larger `director_facebook_likes`, this could possibly suggest that there is some correlation if we split them into subgroups to observe. (as we recall that there is binomial distribution) 


<!--- ### 5. actor_name 
We concatenated data from `actor_1_name`, `actor_2_name` and `actor_3_name`. Here are the top20 most frequently appeared actors/actresses! 
Any names that you know of? 

|actor\_name|num of appearances| |actor\_name|num of appearances|
|---|---|---|---|---|
|Robert De Niro|36||Matthew McConaughey|21|
|Bruce Willis|30||Harrison Ford|21|
|Morgan Freeman|29||Robert Downey Jr\.|21|
|Steve Buscemi|26||James Franco|20|
|Johnny Depp|26||Brad Pitt|20|
|Will Ferrell|26||Matt Damon|20|
|Bill Murray|24||Sylvester Stallone|19|
|Denzel Washington|23||Meryl Streep|19|
|Nicolas Cage|22||Robin Williams|19|
|J\.K. Simmons|21||Julia Roberts|19|
--->

### 5. genres 
We had to split the strings into individual genres. 

![Screenshot 2022-04-22 at 2 56 12 PM](https://user-images.githubusercontent.com/81760484/164620520-6e76816f-401f-4361-a1e0-00c9c79d3a5a.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![genreFreq](https://user-images.githubusercontent.com/81760484/164620806-17a03280-1762-4774-a01d-5abbd2c3a51e.png)
<!---```python
# get the genre frequencies 
from collections import Counter
genreDi = Counter()

for strGenre in imdb["genres"] :
  wds = strGenre.split("|")
  for w in wds :
    if w in genreDi:
      genreDi[w] = genreDi[w] + 1
    else:
      genreDi[w] = 1 

print(genreDi)
``` 

```python 
# convert the dictionary into pandasdataframes + sort in descending order 

genreFreq = pd.DataFrame.from_records(genreDi.most_common(), columns = ["Genre", "Count"])

genreFreq.head(n=10)
```
--->


**Observations** 
- Most common genre : Drama 

- Is it because it is the most profitable?

- this formed our hypothesis that: assuming that the movies industry follows demand and supply, there is high demand for Dramas, 
so this genre will be the most popular with the highest ratings amongst the other genres.

**genres vs mean imdb_scores**

We calculated the mean imdb_scores for each genre. 

The results : 

&nbsp;&nbsp;&nbsp;&nbsp;![genre_vs_score](https://user-images.githubusercontent.com/81760484/164622775-e9f8010f-2613-46b5-95cb-7522ae5e9c4a.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![cat_plot](https://user-images.githubusercontent.com/81760484/164623638-6af3b9b5-e09b-49d9-a8aa-b1a7064cbca1.png)

It seems that Film-Noir has the highest `imdb_score`, however, this is inaccurate, as later, we find out that there were only 5 Film-Noir movies contributing to this observation. 
As noted here: 

![164624141-ac8d8373-3cff-4170-afb1-ffb19079eb77](https://user-images.githubusercontent.com/81760484/164884265-1b35b970-8ee2-4acd-b7b5-3f4c7808f3e7.png)



### 6. budget 
We decided to use only movies produced in the USA, so we could standardize the budget based on CPI (referenced: https://aarya1995.github.io/) 

We performed web scraping using BeautifulSoup to obtain CPI data. Then, we updated the budget column of the whole dataset.
```python
from bs4 import BeautifulSoup
import requests
```
<img src="https://user-images.githubusercontent.com/81760484/164617316-3b00dec4-fc21-4825-b01c-8493cabe749c.png" width="300" height="300">

**budget vs imdb_score**

Initially, we couldn't really see any pattern with only 2 and 4 imdb_score bins. 
So we split into 5 bins and saw a clearer picture. 
It does seem like higher budget can influence imdb_score. However, for the "horrendous" category, it seems like the budget used on them is higher. 

This could mean that although budget does follow a certain trend as imdb_score increases, we ought to be careful with our budget as there is still a risk of the movie turning out to be "horrendous" 

```python
# the new bins (5 categories) we used 
# bins = [1,3,4,6,9,10], labels = ["horrendous", "very bad", "bad", "ok", "good"]

```
<img src="https://user-images.githubusercontent.com/81760484/163830896-41f5615a-ab65-498c-85d1-292c307c58ee.png" width="500" height="250">

### 7. num_voted_users 

- Positively-skewed, large proportion had no number of voted users 
- Not much linear correlation either : with a correlation of -> 0.470567

<img src="https://user-images.githubusercontent.com/81760484/164625453-d1feef57-1c50-44ea-a3b5-eea9f8cf59a3.png" width=40% height=auto> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/81760484/164625754-609acd65-b334-44d5-b4e6-6a21a6842547.png" width=40% height=auto>



### 8. imdb_score 
<img src="https://user-images.githubusercontent.com/81760484/164626388-4422ef1a-71de-401e-a0cd-c0ea813dc8b6.png" width=50% height=aito>

<!---<img src="https://user-images.githubusercontent.com/81760484/164626481-f8250bfb-9b1e-49e1-a235-8545d1c0fcca.png" width="600" height="300">
--->

A large proportion has imdb score of around 5-8.

The median of imdb_score is 6.5, which is why we chose one of our bins to be [0. 6.6, 10] (i.e. 0-6.5 will be classfied as "bad" and 6.6-10 as "good") 

## Multivariate EDA 
<img src="https://user-images.githubusercontent.com/81760484/164652576-56979110-8b18-4e20-9403-3cc86f84a1d2.png" width=50% height=aito>
The heat map shows that some variables affecting imdb_score are:

- `num_critic_for_reviews`
- `duration`
- `num_voted_users`
- `num_user_for_reviews`
- `movie_facebook likes`


# Step 4: Machine Learning 
We explored several ML Models, the best-performing ML Model for our dataset turned out to be ..... Random Forest! 

The list of models we used were: 
1. Linear Regression 
2. Logistic Regression 
3. K-Means 
4. Decision Tree
5. Random Forest (Main)

### Linear Regression 
- As expected, since our dataset is highly categorically-inclined, linear regression for both bivariate and multivariate LR had low R<sup>2</sup> and MSE scores.
- Below shows the scores of some bivariate LR that we attempted 
- <img src="https://user-images.githubusercontent.com/81760484/164642331-dd029105-6dbe-4168-a941-46c4196f0361.png" width= 90% height=auto>
- Multivariate LR 
- <img src="https://user-images.githubusercontent.com/81760484/164642857-70678a7a-390f-46f9-b0c5-ea35162e182f.png" width= 60% height=auto>

### Logistic Regression 
- Logistic Regression showed slightly better results, however, accuracy scores were not that high either. 
- This implies that there is not a "clear cut" between the datas. 
- Since there is some improvements, maybe a decision tree would show better results. 
- We performed Multivariate and Multiclass Logistic Regression 
<img src="https://user-images.githubusercontent.com/81760484/164646260-88e32140-5c0e-4c76-8154-322a5c2efdec.png" width= 75% height=auto>


- For mulvariate logR, scaling improved the accuracy score from 0.51 to 0.66 
- K-folds also improved: 
  - multivariate accuracy scores from 0.51 to 0.63
  - multiclass accuracy scores from 0.65 to 0.66 (very slightly) 

- Multiclass logRegression showed better scores than binomial logRegression. 
- We also used other metrics like F1 scores and precision to observe our model. 


### K-Means 
K-means is an unsupervised machine learning model. 
We found out that the optimal number of clusters is 3 (using elbow method) 

<img src="https://user-images.githubusercontent.com/81760484/164647997-d473b284-c84b-4281-a6ac-555285d8ec2d.png" width=50% height=auto>

The 2-D Grid, Parallel Coordinates Plot and Boxplot all show that `budget` is a huge determinant in influencing the split between the clusters! 

<img src="https://user-images.githubusercontent.com/81760484/164648027-51a1331c-df06-4a7c-9420-e5898f8ab352.png" width="300" height="300">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/81760484/164648133-1a9c1421-39a0-40d2-bfd2-f9b8c470c9a6.png" width="300" height="300">

<img src="https://user-images.githubusercontent.com/81760484/164648197-3e92f65c-5bd6-445e-8142-756df6bbf1e9.png" width=60% height = auto>


### Decision Tree 

Train vs Validation results: had relatively good performance with 0.7 - 0.83 accuracy. This further confirms that our dataset is highly categorically inclined. However, train data had slightly better accuracy compared to validation, indicating that there may be slight overfitting issues. 

Nevertheless, since the performance was good, we decided to use dectree on our Test dataset. Below shows the results. 

<img src="https://user-images.githubusercontent.com/81760484/164649648-56195b2b-3bd9-4d36-ae67-f84a0fb0a7b5.png" width=40% height = auto>

### Random Forest 

Random Forest was the best! (Although, again, there may be slight overfitting for the same reasons as dectree + it took quite long to load) 

Accuracies of :
- Train Data = 0.96 
- Validation Data = 0.82 
- Test Data = 0.99 

Feature importance in random forest shows how important each feature is in determining the decision the tree makes
Below shows the feature importance for determining `imdb_score`. 

![feature important](https://user-images.githubusercontent.com/81760484/164649061-77fa421d-19e2-41c5-8cd9-6124070ec14e.png)

Turned out that `num_voted_users`, `duration`, `num_user_for_reviews`, `num_critic_for_reviews` and `budget` are the top5 determinants. 

It is interesting to note that the variables that indicate popularity are : `num_voted_users`, `num_user_for_reviews`, and `num_critic_for_reviews`, and it is not unexpected for them to be determinants of success (imdb_scores). 



# Interesting Questions 
### Does the movie title length affect imdb scores? 
Unfortunately, as much as we wanted to see some correlation, our bivariate EDA tells us that there isn't any correlation. See the boxplot below! 
However though, there is a somewhat normal distribution in the data, with a median length of 13 

![distribution of title_length](https://user-images.githubusercontent.com/81760484/163716854-952900d4-3d06-4663-a46a-fbc05ccea7d3.png)
![title_length vs imdb_goodbad](https://user-images.githubusercontent.com/81760484/163716860-90a62849-47be-415a-a393-78ba62124d43.png)
![title_length vs imdb_cat](https://user-images.githubusercontent.com/81760484/163716865-4f491560-15e5-4f92-b431-aa53e2110eb0.png)

### What are the personalities of directors of top performing movies? 
We looked into the Top20 imdb_score movies and searched for their personalities online. 
Here are the results ! 
|index|director\_name|personality type||index|director\_name|personality type|
|---|---|---|---|---|---|---|
|0|Frank Darabont| INFP ||10|David Fincher| INTJ |
|1|Francis Ford Coppola| INTJ ||11|Christopher Nolan| INTJ |
|2|John Stockwell| INFP ||12|Peter Jackson| ENFP |
|3|Christopher Nolan| INTJ ||13|Irvin Kershner| INTP |
|4|Francis Ford Coppola| INTJ ||14|Mitchell Altieri| n/a | 
|5|Peter Jackson| ENFJ ||15|Lana Wachowski| ENFP |
|6|Sergio Leone| n/a | |16|Cary Bell| n/a |
|7|Steven Spielberg| ISFP ||17|Fernando Meirelles| INFP |
|8|Quentin Tarantino| ENTP | |18|Milos Forman| INTP |
|9|Robert Zemeckis| ENFP | |19|Akira Kurosawa| INFJ |


Observations: almost all of them (except for one - Steven Spielberg) have "N" in their personalities, which is the intuitive element. 

Do you, as a movie director, have these personality traits too? 

# The Big Conclusion: 

1. Our outcomes show that decision tree and random forest are the most suitable machine learning models for our data set. 
2. This may be due to our dataset having skewed and imbalance data. Also, our dataset does not have very good linear relationships.
3. Duration and budget of the movie are the top 5 features affecting imdb score.  
4. Popularity of the director and cast plays a role in determining imdb score.
5. The top 3 genres affecting imdb score is drama, comedy and action. This aligns with our bi-variate eda as drama is one of the most representation genres affecting imdb_score.

So a movie director should pay close attention to the aforementioned factors. 

Generally, based on our EDA and ML, movies with the following attributes will do better on the imdb rating score:
- Higher duration
- Higher budget 
- More popular director and cast 
- Movies with the genres of drama, comedy and/or action 



# Beyond our Course: 
- Standardising budget to 2016 inflation rate as the latest movies only go up to 2016 
- Web scraping 
- Visualisations:
  - 3D scatter plot & word cloud
 
   <img src="https://user-images.githubusercontent.com/81760484/164884715-cc90c437-bcc6-4e97-9fb2-669ff25a7499.png" width=30% height = auto>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/81760484/164654849-ff16eddb-20ff-43c2-b2f3-9bcff4fd1a20.png" width=60% height = auto>


- Machine Learning: 
 - K-modes & K-means 
 - Logistic Regression
    - using Scaler() from sklearn 
 - Random Forest 
  - Feature Importance 
 - Metrics

# Limitations and Discussion:
1. Analysis of personalities of the directors may be biased because they may be classified as those personalities based on their careers. Therefore, it may not be an accurate representation. However, it is still interesting to note their personalities! 
2. Further analysis can be done on other variables that indicate success through popularity or movie like `director_facebook_likes`, `num_critic_for_reviews`, `num_voted_users` 
3. Our dataset is quite imbalanced and skewed, therefore a larger dataset may help. 


# Workload Delegation: 
### 1. Koh Zi En
  - ML : KMeans, Decision Tree 
  - Presentation 
  - EDA
  - Codes for EDA : [EDA on last 9 Variables](https://github.com/imaginaryBuddy/imdbMoviesDSAI/blob/main/EDA_on_last_9.ipynb)
  - Data Visualisation 
### 2. Sandhiya Sukumaran 
  - ML : Random Forest, Linear Regression 
  - Presentation 
  - EDA 
  - Codes for EDA : [EDA on mid 9 Variables](https://github.com/imaginaryBuddy/imdbMoviesDSAI/blob/main/EDA_on_mid_9.ipynb)
  - Data Visualisation 
### 3. Yap Shen Hwei 
  - ML : Logistic Regression
  - Presentation 
  - EDA
  - Codes for EDA : [EDA on first 9 Variables](https://github.com/imaginaryBuddy/imdbMoviesDSAI/blob/main/EDA_on_first_9.ipynb)
  - Github 
  - Answering Interesting Questions : [Codes here](https://github.com/imaginaryBuddy/imdbMoviesDSAI/blob/main/answeringInterestingQues.ipynb)

# Our Video: 

[![Alt text](https://user-images.githubusercontent.com/81760484/163825344-c985b805-d06b-4934-98ff-c517a6541d48.png)](https://youtu.be/CTV7WypIIf0)

# References:
- https://aarya1995.github.io/
- https://www.kaggle.com/code/carolzhangdc/predict-imdb-score-with-data-mining-algorithms
- https://www.kaggle.com/code/niklasdonges/end-to-end-project-with-python/notebook
- https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397
- https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/
- http://rstudio-pubs-static.s3.amazonaws.com/342210_7c8d57cfdd784cf58dc077d3eb7a2ca3.html#conclusion
- https://scikit-learn.org/stable/modules/impute.html
- https://www.datacamp.com/community/tutorials/wordcloud-python
- https://machinelearningmastery.com
- https://www.bespeaking.com/wp-content/uploads/2019/09/Movie-vocab.jpg
