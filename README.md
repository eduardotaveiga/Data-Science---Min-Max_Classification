# Data-Science---Min-Max_Classification
Trying to predict whether we should buy, sell or hold a specific stock.

*This work is currently on progress*

First things first, a brief disclaimer: it is not my intention to give any kind of investment recommendation and I do not guarantee that you will have any benefits by using or replicating what is shown here. Everything was done only considering educational purposes.

This project is part of my data science learning journey. Feel free to send me a message regarding tips of any kind, like statistics, machine learning, visualizations, or even english grammar, I will be grateful. I hope that you find something useful here and learn with me too!

A very conceptually simple strategy in the investment world is to buy when prices are low and sell when they are high. At extreme, we could get the maximum returns by buying at the minimum value and selling at the maximum. But, things are not that easy and predicting the stock market direction is a really hard task, which means that predicting the absolute prices is impossible. Here, we are precisely trying to do the latter.

Consider the following image about the historical price of a brazilian stock:

![newplot](https://user-images.githubusercontent.com/76738265/197007110-2725f834-ba14-4f3d-aa5e-55c23e5c3ec4.png)

Our task is to build a Machine Learning model to predict whether a specific day is a "buy" oportunity, a "sell" spot, or neither ("hold"). For that, we need to implement an algorithm that searches for local minimum and maximum, labeling them as 0 or 1, respectively. Every other point is labeled as 2, which means that you should "hold" the stock at that day.

![newplot](https://user-images.githubusercontent.com/76738265/197007316-13ea0053-4319-4656-88f3-45959f4e93a5.png)

After finding our targets, we create some new features to enhance the dataset and train a model, selecting features, optimizing its hyperparameters and putting it to predict.  

![newplot](https://user-images.githubusercontent.com/76738265/197007540-68c4e69f-f90c-432c-bcb6-7950cbb1ccd3.png)

With the predictions at hand, it's time to analyse the evolution of R$1000.00 (local currency) invested on this strategy.

![newplot](https://user-images.githubusercontent.com/76738265/197359583-656394e8-4954-4670-9dac-2c6db60c30ef.png)

![newplot](https://user-images.githubusercontent.com/76738265/197359588-1ed4c6ac-dd81-4f7b-b3d7-7bb2c3395d92.png)


|Stock|Train_set_LR|Test_set_LR |Train_set_RF|Test_set_RF|Train_set_SVC|Test_set_SVC|
|-----|------------|------------|------------|-----------|-------------|------------|
|PETR4|6.417416e+09|1.647811e+07|1000.000000 |1000.0     |             |            |
|VALE3|5.258041e+08|3.245516e+06|1000.000000 |1000.0     |             |            |
|ITUB4|2.590539e+08|9.683918e+04|1000.000000 |1000.0     |             |            |
|MGLU3|4.627618e+05|1.228008e+05|1000.000000 |1000.0     |             |            |
|PRIO3|8.793738e+04|1.024000e+05|1000.000000 |1000.0     |             |            |
|BBAS3|3.562421e+09|4.293427e+05|1524.762499 |1000.0     |             |            |
|BBDC4|2.003126e+06|2.649287e+04|4578.484050 |1000.0     |             |            |
|ELET3|1.232198e+10|8.434031e+06|5276.491156 |1000.0     |             |            |
|B3SA3|1.152574e+07|1.949376e+05|2453.195532 |1000.0     |             |            |
|RENT3|9.864012e+07|7.540797e+04|1000.000000 |1000.0     |             |            |

Note: results may vary due to randomness.

See the *1. Project planning* notebook for more information!
Also, I would like to thank Kaneel Senevirathne for writing [this post] on medium which was my main source of inspiration for this project!

*This work is currently on progress*
Next steps:
Get SVC results
Monte Carlo simulations
Train LSTM neural network

[this post]:https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2
