
# Motivation
The cult of sneaker is an interesting derivation of the basketball sports. As a fervid basketball fan, I came across the [StockX's Data Contest dataset](https://stockx.com/news/the-2019-data-contest/). I decided to apply some of the skills I learned from the Center of Data Science of NYU on this dataset, and hopefully find some interesting patterns.

Knowledge applied:

- Math:
  - Perron-Frobenius Theorem
  - Google's PageRank Algorithm
- Python:
  - Numpy, Pandas, Matplotlib

# Data Description
Quote from the StockX team:

  <em>"The data in this sheet consist of a random sample of all U.S. Off-White x Nike and Yeezy 350 sales from between 9/1/2017 and 2/13/2019. 
  
  To create this sample, we took a random, fixed percentage of StockX sales (X%) for each colorway, on each day, since September, 2017. So, for each day the Off-White Jordan 1 was on the market, we randomly selected X% of its sale from each day. (It’s not important to know what X is; all that matters is that it’s a random sample, and that the same fixed X% of sales was selected from every day, for every sneaker). The sample was limited to U.S. sales only.
  
  We’ve included 8 variables for you to work with: Order Date, Brand, Sneaker Name, Sale Price ($), Retail Price ($), Release Date, Shoe Size, and Buyer Region (the state the buyer shipped to). You can use whatever variables you want in the analysis; you can use 1 variable, or you can use all 8. "</em>
 
I am curious about the frequency of appearance of each model in this dataset.

```python
import pandas as pd
import numpy as np
data = pd.read_excel('StockX-Data-Contest-2019-3.xlsx', sheet_name = 1, header = 0)
freq = data.groupby('Sneaker Name').count()
freq = freq.sort_values(by = 'Brand',ascending = False)
```

From this we know that "adidas-Yeezy-Boost-350-V2-Butter" is the most frequent sneaker in this dataset, while "Nike-Air-Force-1-Low-Virgil-Abloh-Off-White-AF100" is the least frequent sneaker.

# Plan
I am thinking of a ranking problem.
- To use PageRank Algorithms to rank the importance of each sneaker. An updated version of this implementation can include the factors of prices. Compare this with a simple rank given by frequency.
  - Compare the ranking given by the transition matrix and the rankings of simple frequency
  - Plot the variance and mean of the net profit (sale price - retail price) and see if the *importance* has any relationship with the profit 
  - Evaluation of ranking: how well does it predict the profitability?
  
# Task: Ranking

I came across the PageRank algorithms in my "Optimization and Computational Linear Algebra" course during my first semester at NYU. I implemented a variation of PageRank algorithms to rank the NBA teams based on their records against each other, and later take the scores in the game in account. I realized that such idea may also be applied to the Sneakers dataset on hand: **Can I rank these sneakers based on this idea, and also later add prices in my analysis?**

The PageRank algorithm imagines a 'drunk' surfer randomly clicking the pages, and the importance of a webpage is quantified by *How much time he spends on each website*. I mimiced this idea in my analysis by imagining a "drunk sneaker buyer" randomly purchasing sneakers. There are 50 unique models in total. When the buyer bought model A after buying model B, I defined that there is a *link* from model A to model B.

Going through the data table row by row, the drunk buyer is constantly creating a link between two models. Matrix $\textbf{M}$ records these links. <img src="https://latex.codecogs.com/gif.latex?M_{i,j}" title="M_{i,j}">represents number of times the buyer buys *j* after buying *i*.

Here are codes for generating *M*.

```python
i = 0
M = np.zeros((nbshoes,nbshoes))
# rows: current one, columns: next one
# for example: 1,2 means how many times there is a purchase of 2 followed by 1
current = shoes_code[data.iloc[0]['Sneaker Name']]
for i in range(len(data.index)):
    try:
        nextone = shoes_code[data.iloc[i+1]['Sneaker Name']]
        # link from current to nextone += 1
        M[current,nextone] += 1.0
        current = nextone
    except IndexError as e:
        break
```
The number of links from sneaker *i* is defined as the degrees of *i*. If a sneaker is pointed to by more sneakers, then it is considered as a 'more important' sneaker. How many links the first sneaker, 'Adidas-Yeezy-Boost-350-Low-V2-Beluga', has?

```python
print(M[0,:])
array([112.  17.   9.  14.  15.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 331.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.])
```

It has 6 links in total, becauase there is 6 non-zero entries in its row. Following this logic, we show the links of every sneaker:

```python
M_edge = np.count_nonzero(M, axis = 1)
M_edge
array([ 6,  4,  5,  4,  3,  2,  4, 12,  9, 13, 11, 13, 10, 15,  7,  7,  3,
        6,  3, 15, 11,  4, 13, 11,  6,  7,  7, 11, 13, 11,  4,  4,  5,  6,
        9,  4,  2,  8,  2,  6,  3,  4,  2,  2,  5,  4,  6,  8,  6,  6])
```

We can hereby define the transition matrix *P*，where <a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_{i,j}" title="P_{i,j}" /></a> means the probability that the buyer transition from sneaker j to sneaker i.

```python
# define P
p = np.zeros((nbshoes,nbshoes))
for i in range(nbshoes):
    for j in range(nbshoes):
        if M[j,i] != 0:
            p[i,j] = 1/M_edge[j]
        if M[j,i] == 0:
            p[i,j] = 0
```

The key of understanding is that <a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_{i,j}" title="P_{i,j}" /></a> only gets a value if there is a non-zero value in <a href="https://www.codecogs.com/eqnedit.php?latex=M_{j,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M_{j,i}" title="M_{j,i}" /></a>. **In other words, if a sneaker *k* has more incoming links from another sneaker *s*, when we multiply an *n-vector* to the matrix *P*, sneaker *k* will have a greater entry on its position than sneaker *s*.**

Since this is a transition matrix, each column should sum up to one, and it does.

```python
np.sum(p,axis = 0)
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

Google uses a parameter <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> to make sure the matrix satisfies the conditions for *Perron-Frobenius Theorem*. In my case, the parameter doesn't change the ranking by too much.

By *Perron-Frobenius Theorem*, we can then find an invariant measure of matrix *g*.
```python
# define g
g = np.zeros((nbshoes,nbshoes))
alpha = 0.85
g = np.add(alpha*p, ((1-alpha)/nbshoes)*np.ones((nbshoes,nbshoes)))
# find invariant measure of g
x0 = np.random.rand(50,1)
g_new = np.matmul(g,g)
nruns = 500
for time in range(nruns):
    g_new = np.matmul(g,g_new)
mu = np.matmul(g_new,x0)
mu_transpose = mu.transpose()
mu_transpose[0]
array([1.01212683, 0.6617741 , 0.83437669, 1.10138629, 1.38773063,
       1.52071462, 1.96912188, 1.29168915, 0.45933716, 0.34406051,
       0.52035946, 0.4583473 , 0.18320918, 0.36881516, 0.52160361,
       0.33363046, 0.67315751, 0.45826603, 0.69011184, 0.22810528,
       0.37750283, 0.15399109, 0.28587159, 0.35324224, 0.9533496 ,
       0.2403898 , 0.60811125, 0.21517931, 0.35685355, 0.19855182,
       0.61668735, 0.39617817, 0.21935017, 0.23120464, 0.44507964,
       0.35885118, 0.33776711, 0.24581009, 1.2001945 , 0.11913678,
       0.43807686, 0.34333117, 0.40898765, 0.43471565, 0.16514622,
       0.15369885, 0.137682  , 0.10701729, 0.14762665, 0.1130744 ])
```

Now let us visualize the rankings, and compare it to the rankings by simple frequencies of sneakers.

```python
# ranking by pagerank
rank = dict(zip(mu_transpose[0], shoes_code.keys()))
sorted_rank = sorted(rank.keys(),reverse = True)
sorted_rank_name = [rank[i] for i in sorted_rank]
sorted_rank_shoe_code = [shoes_code[j] for j in sorted_rank_name]
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,5))
rank_bar = plt.bar(height = sorted_rank, x = sorted_rank_name)
plt.xticks(ticks = sorted_rank_name, rotation = 90)
plt.show()

# ranking by frequency
freq_dict = dict(zip(freq.index, freq['Brand']))
fig2 = plt.figure(figsize=(20,5))
freq_bar = plt.bar(height = list(freq_dict.values()), x = list(freq_dict.keys()))
plt.xticks(ticks = list(freq_dict.keys()), rotation = 90)
plt.show()

```
**Ranking by PageRank**
![alt text](https://github.com/ZichangYe/SneakersAnalytics/blob/master/Rank_PageRank.png "Ranking By PageRank")

**Ranking by Frequency**
![alt text](https://github.com/ZichangYe/SneakersAnalytics/blob/master/Rank_Simple_Frequency.png "Ranking By Frequency")

Obvious they are different. We can make some observations:

- The ranking given by the PageRank is smoother than the simple frequency.
- The Top 10 is definitely not exactly the same. 

Which one is more useful? *Thinking back to our original question*, I am trying to find a way to predict and identify the sneakers that give highest net profit in a stable way. Therefore, the next step is going to bring the metric of 'net profit = sale price - retail price', and see whether the sneaker with higher net profit on average will have a higher ranking. We will visualize this first.

## Visualizing the rankings with mean and standard deviation of net profits

**Mean of Net Profits**
![alt_text](https://github.com/ZichangYe/SneakersAnalytics/blob/master/mean_profit_pagerank.png "Mean Profit by PageRank")
![alt_text](https://github.com/ZichangYe/SneakersAnalytics/blob/master/mean_profit_freq.png "Mean Profit by Frequency")

- **A general trend is: if the ranking is higher, then the mean profit is lower.** This is especially visible with the ranking given by frequency. This ironically is the opposite of my intention, but is intuitive because the more frequent the sneaker is being traded, the most likely it will return to a market-clearing price, driving the profit to zero. It is the 'rare' sneaker that gives the highest return.

**Standard Deviation of Net Profits**

![alt_text](https://github.com/ZichangYe/SneakersAnalytics/blob/master/std_profit_pagerank.png "Standard Deviation of Profit by PageRank" )
![alt_text](https://github.com/ZichangYe/SneakersAnalytics/blob/master/std_profit_freq.png "Standard Deviation of Profit by Frequency")

- **The PageRank ranking gives a better signal of low price volatility than the ranking given by frequency.** Sneakers with the highest rankings and the sneakers with the lowest rankings are the ones with most stable net profit. This is also reasonable. The prices for the most popular and the most common sneakers are settled down already, while the net profit for the least available sneakers are  stably high because there isn't enough stocks up there. 
- **The sneakers in the middle will turn out to be relatively more risky.** Sometimes the prices will rocket, driving the net profit very high.

# Conclusion

The PageRank is not very informational in terms of telling us the average profitability of each sneaker, but it can help me find a stably profitable sneaker. 

To really explain this, we need to go back to where PageRank was used at the first place. In the original setting, the webpage with the higher ranking is the one that has more links directed into. In our problem, the sneaker is ranked as higher because it has more different shoes purchased one row before itself own record. In long run, the most important sneaker is the sneaker where the 'drunk sneaker buyer' (remember him?) spends the most time on.

This is not necessarily a bad assumption. If we have absolutely no information about a new customer, it is reasonable to assume that he or she to have a random purchasing pattern. Thus, it gives us a guidance about what the consumer might purchase when we know nothing about him or her.

The ranking also illustrates some information about the stability of net profit for each sneaker. Compared to the ranking given by simple frequency, it fits better with our intuition. **Assume that these sneakers are all demanded by the customers**, we can make these statements:
  - The more available the good is, the more stably low the net profit is, i.e. market is clearing.
  - The more unavailable the good is, the more stably high the net profit is, i.e. market is craving.

# Looking Forward

Few future updates can be implemented:
- Update the rankings with the factor of price
- Estimate the demand of each sneaker


2019 Copyright@Zichang Ye
