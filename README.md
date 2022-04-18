# How to Maximise Movie Success - Project Overview

by tripleH 


### Contributors:
* Sand***h***iya Sukuraman 
* Ko***h*** Zhi En [@zex3](https://github.com/zex3)
* Yap S***h***en Hwei [@imaginaryBuddy](https://github.com/imaginaryBuddy)
</n>

![source: bespeaking.com](https://www.bespeaking.com/wp-content/uploads/2019/09/Movie-vocab.jpg)
  
Hello, we are Year 1 CS Students from NTU! Welcome to our project for the course : Data Science and Artificial Intelligence 

Our project is based on IMDB 5000 dataset found on [kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

We wonder, **"If we were movie directors, how could we maximise our success rates?"**

## Foreword
- Sections/ Texts that are highlighted are additional/ extra materials that we learnt outside of our course. 

## Motivation 
- Give directors a better estimation on how to maximize the success rate of their movie

## Content Section 
- Introduction 
- Steps: 
  - 1 : [Looking at the Dataset](#step-1-looking-at-the-dataset)
  - 2 : [EDA] (#eda) 
    - [Uni-variate EDA] (#uni-variate-eda)
    - [Bi-variate EDA] (#bi-variate-eda)
    - [Multi-variate EDA] (#multi-variate-eda)
  - 3 : [Choosing our Variables](#step-3-choosing-our-variables)
  - 4 : [Our Hypotheses](#step-3-our-hypotheses)
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

  #### Dataset info: 
  


  #### Train:Test:Validation 
  We split our dataset into 80:20 ratio, then used the 80% as our Train Dataset to further divide to obtain our Train: Validate for our Machine Learning Models in 80 : 20 ratio. 
  
  
  #### Cleaning the Train Dataset
  - drop_duplicates()
  
  #### Initial Thoughts 
  - popularity of cast,
## Step 3: Choosing our variables 

Our ultimate goal is to find out what we should focus on to make our movie successful.

We define success based on: 
- ratings 
   - `imdb_score`
initially, we wanted to use `gross` as a predictor of success as well, however, we faced some challenges due to inconsistent data [view challenges faced section](#challenges-faced)
   
## Step 2: EDA
In this section, we will look at univariate and bivariate EDAs concerning the respective variables. 
 
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
 
It is interesting to note that Steven Spielberg is also one of directors from the [Top20 performing movies](###What-are-the-personalities-of-directors-of-top-performing-movies?)



### 2. num_critic_for_reviews 
there is no significant linear correlation. 

### 3. duration 
- we binned the `imdb_score` into categories to form `score_cat`
- there seems to be slight correlation based on the boxplot between `duration` and `score_cat`
![duration eda](https://user-images.githubusercontent.com/81760484/163723231-f0a5995b-82c9-4a50-ab68-696089f279b5.png)
![duration vs score](https://user-images.githubusercontent.com/81760484/163723346-65235c7f-2ffb-4581-93e4-261db7619378.png)

### 4. budget 
We decided to use only movies produced in the USA, so we could standardize the budget based on CPI (referenced: https://aarya1995.github.io/) 

We performed web scraping using BeautifulSoup 
```python

from bs4 import BeautifulSoup
import requests
url = "https://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/"

r = requests.get(url)
data = r.text
soup = BeautifulSoup(data, 'html.parser')

table = soup.find('table')
rows = table.tbody.findAll('tr');

years = []
cpis = []
```
``` python
for row in rows:
    year = row.findAll('td')[0].get_text()
    if year.isdigit() and int(year) < 2017:
        years.append(int(year))
        cpis.append(float(row.findAll('td')[13].get_text()))

cpi_table = pd.DataFrame({
    "year": years,
    "avg_annual_cpi": cpis
})

cpi_table.tail()
```
``` python 
# replace na values with 0
imdbData["budget"].fillna(0, inplace = True)
imdbData["title_year"].fillna(0, inplace = True)

# only consider movies made in the USA. Drop all other rows
imdbData = imdbData[imdbData['country'].str.contains('USA') == True]

imdbData.drop(imdbData[(imdbData["budget"] == 0) | (imdbData["title_year"] == 0)].index, inplace=True)
```
```python
CPI_2016 = float(cpi_table[cpi_table['year'] == 2016]['avg_annual_cpi'])
real_budget_values = []

# must transform gross and budget values into real 2016 dollar terms
for index, row in imdbData.iterrows():
    budget = row['budget']
    year = row['title_year']
    cpi = float(cpi_table[cpi_table['year'] == int(year)]['avg_annual_cpi'])
    
    real_budget = get_real_value(budget, cpi, CPI_2016)
    real_budget_values.append(real_budget)   

# place the converted budget values into the original budget column
imdbData["budget"] = real_budget_values
```
**budget vs imdb_score**

Initially, we couldn't really see any pattern with only 2 and 4 imdb_score bins. 
So we split into 5 bins and saw a clearer picture. 
It does seem like higher budget can influence imdb_score. However, for the "horrendous" category, it seems like the budget used on them is higher. This could mean that although budget does follow a certain trend as imdb_score increases, we ought to be careful with our budget as there is still a risk of the movie turning out to be "horrendous" 

![budget_vs_imdb_score](https://user-images.githubusercontent.com/81760484/163830896-41f5615a-ab65-498c-85d1-292c307c58ee.png)


### 4.

### 5. 
## Step 3: Our hypotheses 
After doing Univariate EDA, and Bivariate EDA, we have chosen particular variables as our potential predictors of success. These are: 


## Step 4: Some Interesting Questions 
- Does the length of the title affect movie success? 
- Are there any types of words that we should include into our movie title to increase our chances of success? 


## Challenges Faced 
Through research, we found that:
1. The type of gross is not standardised, e.g
2. Large proportion of null values
3. Budget needed to be adjusted for inflation  
4. Extremely skewed data
   - Facebook likes data:
     - solution: use log transform 
5. Random Forest took a very long time to load 
6. Extremely low Linear Regression Correlation  
   - solution: 
7. Genres and Plot keywords came 

# Machine Learning 
1. Linear Regression 
2. Logistic Regression 
3. K-Modes 
4. K-Means 
5. Random Forest (Main)


# Interesting Observations 
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
|index|director\_name|personality type|
|---|---|---|
|0|Frank Darabont| INFP |
|1|Francis Ford Coppola| INTJ |
|2|John Stockwell| INFP |
|3|Christopher Nolan| INTJ |
|4|Francis Ford Coppola| INTJ |
|5|Peter Jackson| ENFJ |
|6|Sergio Leone| NA | 
|7|Steven Spielberg| ISFP |
|8|Quentin Tarantino| ENTP | 
|9|Robert Zemeckis| ENFP | 
|10|David Fincher| INTJ | 
|11|Christopher Nolan| INTJ |
|12|Peter Jackson| ENFP | 
|13|Irvin Kershner| INTP | 
|14|Mitchell Altieri| NA | 
|15|Lana Wachowski| ENFP |
|16|Cary Bell| NA |
|17|Fernando Meirelles| INFP |
|18|Milos Forman| INTP |
|19|Akira Kurosawa| INFJ |

Observations: almost all of them (except for one - Steven Spielberg) have "N" in their personalities, which is the intuitive element. 
# Results 

# Beyond our Course: 
- Standardising budget to 2016 inflation rate as the latest movies only go up to 2016 
- Web scraping 
- Visualisations:
 - 3D scatter plot 
 - Word Cloud 
- Machine Learning: 
 - K-modes & K-means 
 - Logistic Regression
    - Scaler from sklearn 
 - Random Forest 
  - Feature Importance 
 - Metrics

# Limitations and Discussion:
1. Analysis of personalities of the directors may be biased because they may be classified as those personalities based on their careers. Therefore, it may not be an accurate representation. However, it is still interesting to note their personalities! 
2. Further analysis can be done on other variables that indicate success through popularity or movie like `director_facebook_likes`, `num_critic_for_reviews`, `num_voted_users` 
3. Our dataset is quite imbalanced and skewed, therefore a larger dataset may help. 


## Workload Delegation: 
1. Koh Zi En 
  - ML : KModes, KMeans, Random Forest 
  - Presentation 
  - Exploratory Data Analysis
  - Data Visualisation 
2. Sandhiya Sukuraman 
  - ML : Decision Tree, Linear Regression 
  - Presentation 
  - EDA 
  - Data Visualisation 
3. Yap Shen Hwei 
  - ML : Logistic Regression
  - Presentation 
  - Exploratory Data Analysis 
  - Github READMe 
## Our Video : 

[![Alt text](https://user-images.githubusercontent.com/81760484/163825344-c985b805-d06b-4934-98ff-c517a6541d48.png)](https://youtu.be/CTV7WypIIf0)

## References 
- https://aarya1995.github.io/
- https://www.kaggle.com/code/carolzhangdc/predict-imdb-score-with-data-mining-algorithms
- https://www.kaggle.com/code/niklasdonges/end-to-end-project-with-python/notebook
- https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397
- https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/
- http://rstudio-pubs-static.s3.amazonaws.com/342210_7c8d57cfdd784cf58dc077d3eb7a2ca3.html#conclusion
- https://scikit-learn.org/stable/modules/impute.html
- https://www.datacamp.com/community/tutorials/wordcloud-python

