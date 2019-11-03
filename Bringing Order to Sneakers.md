markdown: kramdown

kramdown:
  input: GFM
 
# Motivation
The cult of sneaker is an interesting derivation of the basketball sports. As a fervid basketball fan, I came across the [StockX's Data Contest dataset](https://stockx.com/news/the-2019-data-contest/). I decided to apply some of the skills I learned from the Center of Data Science of NYU on this dataset, and hopefully find some interesting patterns.

# Data Description
Quote from the StockX team:

  <em>"The data in this sheet consist of a random sample of all U.S. Off-White x Nike and Yeezy 350 sales from between 9/1/2017 and 2/13/2019. 
  
  To create this sample, we took a random, fixed percentage of StockX sales (X%) for each colorway, on each day, since September, 2017. So, for each day the Off-White Jordan 1 was on the market, we randomly selected X% of its sale from each day. (It’s not important to know what X is; all that matters is that it’s a random sample, and that the same fixed X% of sales was selected from every day, for every sneaker). The sample was limited to U.S. sales only.
  
  We’ve included 8 variables for you to work with: Order Date, Brand, Sneaker Name, Sale Price ($), Retail Price ($), Release Date, Shoe Size, and Buyer Region (the state the buyer shipped to). You can use whatever variables you want in the analysis; you can use 1 variable, or you can use all 8. "</em>

# Plan
I am thinking of two sub-projects.
- To use PageRank Algorithms to rank the importance of each sneaker. An updated version of this implementation will include the factors of prices. Compare this with a simple rank given by a weighted average between frequency and mean prices.
  - Compare the ranking given by the transition matrix and the rankings of simple frequency
  - Plot the variance and mean of the net profit (sale price - retail price) and see if the *importance* has any relationship with the profit
- To break the datasets into two parts, and try to predict the prices.

# Task 1: Ranking

I came across the PageRank algorithms in my "Optimization and Computational Linear Algebra" course during my first semester at NYU. I implemented a variation of PageRank algorithms to rank the NBA teams based on their records against each other, and later take the scores in the game in account. I realized that such idea may also be applied to the Sneakers dataset on hand: **Can I rank these sneakers based on the frequencies of purchases, and also later add prices in my analysis?**

The PageRank algorithm imagines a 'drunk' surfer randomly clicking the pages, and the importance of a webpage is quantified by *How much time he spends on each website*. I mimiced this idea in my analysis by imagining a "drunk sneaker buyer" randomly purchasing sneakers. There are 50 unique models in total. When the buyer bought model A after buying model B, I defined that there is a *link* from model A to model B.

Going through the data table row by row, the drunk buyer is constantly creating a link between two models. Matrix $\textbf{M}$ records these links. $M_{i,j}$ represents number of times the buyer buys $j$ after buying $i$.

# Task 2: Price Prediction
