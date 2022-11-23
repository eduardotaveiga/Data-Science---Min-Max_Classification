[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<div class="alert alert-block alert-info">
<b><h2><center>Data Science Project - Buy/Hold/Sell prediction</center></h2></b>
</div>

Trying to predict whether we should buy, sell or hold a specific stock.

*First things first, a brief disclaimer: it is not my intention to give any kind of investment recommendation and I do not guarantee that you will have any benefits by using or replicating what is shown here. Everything was done only considering educational purposes.*

This project is part of my data science learning journey. Feel free to send me a message regarding tips of any kind, like statistics, machine learning, visualizations, or even english grammar, I will be grateful. I hope that you find something useful here and learn with me too!

---
## Roadmap
*This work is currently in progress*
- [x] Min/Max algorithm
- [x] Basic ML models 
- [ ] LSTM model
- [ ] Evaluate model profitability
    - [x] Random Forest
    - [x] Logistic Regression
    - [ ] LSTM
- [ ] Monte Carlo simulations 
- [ ] Model deployment
    - [ ] Web application using Dash
---

## About the project

A very simple strategy (at least, conceptually) in the investment world is to buy when prices are low and sell when they are high. At extreme, we could get the maximum returns by buying at the minimum value and selling at the maximum. But, things are not that easy and, if predicting the stock market direction is already a really hard task, then predicting the absolute prices is nearly impossible. Here, we are almost trying to do the latter.

## Data Preparation

By using the Yahoo Finance API (in Python, with the aid of yfinance package), we have access to the historical price of different stocks. Consider the following image about the historical (close) price of a brazilian stock

![newplot](https://user-images.githubusercontent.com/76738265/203628939-b4561977-7e6b-4350-b206-3135ab050a27.png)

![newplot](https://user-images.githubusercontent.com/76738265/197007110-2725f834-ba14-4f3d-aa5e-55c23e5c3ec4.png)

Generally, the first idea that comes to mind when we think about using Machine Learning to trade in the stock market is to build a complicated regression model, like a neural network, using the close price as a target. But here, our task is to build a *classification* model to predict whether a specific day is a "buy" oportunity, a "sell" spot, or neither ("hold"). 

For that, the first step is to split our dataset using the first 70% data to compose the training set and the last 30% to create the test set. Splitting at the very beginning avoids data leakage. Then, implement an algorithm that searches for local minimum and maximum, labeling them as 0 or 1, respectively. Every other point is labeled as 2, which means that you should "hold" the stock at that day.

![newplot](https://user-images.githubusercontent.com/76738265/197007316-13ea0053-4319-4656-88f3-45959f4e93a5.png)

After finding our targets, we can create some new features to enhance the dataset, like a normalized price, defined as following:

$Normalized price = \frac{Close-Low}{High-Low}$

The advantage of this feature is that it contains the information of three types of prices (Low, High and Close) and not only the close one. We can also compute the rolling mean and angular coefficients of a fitted linear regression for 5, 10 and 21 days.

![newplot](https://user-images.githubusercontent.com/76738265/203630046-515f91b0-1e12-4a09-aba4-de73f63a5752.png)

## Model Training

We proceed to train a Machine Learning model. At this point, it's useful to do some feature selection, using ANOVA F-value, Mutual Information value, or any other useful metric, following by a hyperparameter optimization. For the prediction step, it's possible to do some threshold manipulation in order to ask our model to be at least 75% sure that the current data point is whether 0 or 1 (buy or sell), otherwise it's a 2 (hold). We end up missing some true opportunities, but we also avoid a lot of false 0s and 1s. 

![newplot](https://user-images.githubusercontent.com/76738265/197007540-68c4e69f-f90c-432c-bcb6-7950cbb1ccd3.png)

## Strategy Evaluation

With the predictions at hand, it's time to analyse the evolution of R$1000.00 (local currency) invested using those models.

![newplot](https://user-images.githubusercontent.com/76738265/197359583-656394e8-4954-4670-9dac-2c6db60c30ef.png)

![newplot](https://user-images.githubusercontent.com/76738265/197359588-1ed4c6ac-dd81-4f7b-b3d7-7bb2c3395d92.png)


|Stock|Train_set_LR|Test_set_LR |Train_set_RF|Test_set_RF|Train_set_LSTM|Test_set_LSTM|
|-----|------------|------------|------------|-----------|--------------|-------------|
|PETR4|6.417416e+09|1.647811e+07|1000.000000 |1000.0     |              |             |
|VALE3|5.258041e+08|3.245516e+06|1000.000000 |1000.0     |              |             |
|ITUB4|2.590539e+08|9.683918e+04|1000.000000 |1000.0     |              |             |
|MGLU3|4.627618e+05|1.228008e+05|1000.000000 |1000.0     |              |             |
|PRIO3|8.793738e+04|1.024000e+05|1000.000000 |1000.0     |              |             |
|BBAS3|3.562421e+09|4.293427e+05|1524.762499 |1000.0     |              |             |
|BBDC4|2.003126e+06|2.649287e+04|4578.484050 |1000.0     |              |             |
|ELET3|1.232198e+10|8.434031e+06|5276.491156 |1000.0     |              |             |
|B3SA3|1.152574e+07|1.949376e+05|2453.195532 |1000.0     |              |             |
|RENT3|9.864012e+07|7.540797e+04|1000.000000 |1000.0     |              |             |

*Note: results may vary due to randomness.*

Interestingly, the Random Forest algorithm wasn't able to capture useful patterns, while Logistic Regression, a simple linear model, got some noteworthy results.

|Stock|Test_set_LR |Returns %|
|-----|------------|---------|
|PETR4|1.647811e+07|1647700  |
|VALE3|3.245516e+06|324451.6 |
|ITUB4|9.683918e+04|9583.918 |
|MGLU3|1.228008e+05|12180.08 |
|PRIO3|1.024000e+05|10140    |
|BBAS3|4.293427e+05|42834.27 |
|BBDC4|2.649287e+04|2549.287 |
|ELET3|8.434031e+06|843303.1 |
|B3SA3|1.949376e+05|19393.76 |
|RENT3|7.540797e+04|7440.797 |

The question is: are these results reliable? We may want some statistics to ensure that we could trust our money to the algorithm, which leads us to Monte Carlo simulations. 

*Work in progress*

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Eduardo Veiga - [LinkedIn](https://www.linkedin.com/in/eduardo-veiga-0728221a6/)

Email - etaveiga@gmail.com

Project Link: [https://github.com/eduardotaveiga/Data-Science---Min-Max_Classification](https://github.com/eduardotaveiga/Data-Science---Min-Max_Classification)

## Acknowledgments
* See the *1. Project planning* notebook for more information!
* I would like to thank Kaneel Senevirathne for writing [this post] on medium which was my main source of inspiration for this project!


[this post]:https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/eduardotaveiga/Data-Science---Min-Max_Classification/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/eduardotaveiga/Data-Science---Min-Max_Classification/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/eduardotaveiga/Data-Science---Min-Max_Classification/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/eduardo-veiga-0728221a6/
