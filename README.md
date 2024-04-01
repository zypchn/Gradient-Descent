# Gradient-Descent

## Introduction 
Gradient Descent is an optimisation algorithm which helps you find the optimal weights for your model. It does it by trying various weights and finding the weights which fit the models best (minimises the cost function). Lets's breakdown this statement.

A "model" is basically a matematical hypothesis with parameters as coefficents and data points as variables. An example of a very simple model:
![image](https://github.com/zypchn/Gradient-Descent/assets/144728809/be3cc096-2706-4dfd-a0a5-f8b4eed39435)

Cost function can be defined as the difference between the actual output and the predicted output. Below is an example of a cost function mean squared error, but with a slight change:

![Screenshot 2024-03-31 134755](https://github.com/zypchn/Gradient-Descent/assets/144728809/03e27d46-793c-4a63-aa6a-0e7d5ea16262)

Gradient descent uses the partial derivatives of cost function and an arbitrary value called learning rate to update parameters. Image down below shows two functions: partial derivative of cost functin *J* with respect to *θ0* and partial derivative of cost function *J* with respect to *θ1*.

![Screenshot 2024-04-01 100614](https://github.com/zypchn/Gradient-Descent/assets/144728809/795b70f7-9880-4579-8853-939203f94fa9)

The aim of this project is to write a linear regression algoritm. Contrary to what is usually done, we will write our own cost function as well as our own Gradient Descent algorithm; instead of using the algorithms already available on several machine learning libraries. We used a slightly changed mean squared error for our cost function (a common approach), which can be seen above.


## Methods
The first step of any machine learning project is creating the dataset. Our dataset is "Advertising Dataset" from Kaggle, which is defined as: "analyses the relationship between TV advertising, Radio advertising, Newspaper advertising and Sales". This tells us that "TV", "Radio" and "Newspaper" values are all features whereas "Sales" is our targeted value. We downloaded the dataset from Kaggle, and converted it into a pandas dataframe using *pd.to_csv()*. Below is the first 5 entries of our raw dataset:

![Screenshot 2024-03-31 194203](https://github.com/zypchn/Gradient-Descent/assets/144728809/e50d4d27-b0a4-492f-8a11-e95dd4a024a7)

As we can see, the first column is not necessary for our model, it is just an indexing column so we dropped it. And the "sales" column must also be dropped, given that it contains our target values. With these 2 columns dropped, we have our **X** data. For **y** we only need the target values, so all we had to do was extract the *sales* column. We also converted dataframes to numpy arrays for computational reasons and scaled **X** and **y** with *StandardScaler()* from *sklearn.preprocessing* to prevent overflow.

Then, we created our weights and bias using *np.random.randn()* and *rd.random()* respectively.
With weights and bias initialized, we then wrote the *predict()* function with 3 parameters:
*x* (features), *w* (weights) and *b* (bias). The function makes predictions by computing the dot product (using *np.dot()*) of *x* and *w* and adding bias the the result. With *predict()* method written, we created a new data called **y_pred**.

Next step was to write our own cost function. Instead of using *mean_squared_error* from *sklearn.metrics* we wrote our own. Their difference is that our cost function divides the sum to *2×m*, whereas *mean_squared_error* only divides to *m*. We then computed our cost with our own function.

For our own gradient descent algorithm we first wrote our own update function for parameters, namely *update_params()* method. Since Python does not offer a derivative function itself, we had to write the result of the derivative. What that means is that if we change our cost function, we also have to change our *update_params()* method, since it is not a generic method for all cost functions. We wrote our method exactly as shown above on the *Introduction* part (image no: 3). It takes 6 parameters: *X*, *y*, *y_pred*, *theta*, *bias* and *lr*. First 5 parameters were already initialized and explained above. As for learning rate the definition is: a tuning parameter in an optimization algorithm (gradient descent in our case) that determines the step size at each iteration while moving toward a minimum of a loss function.
It is also important to choose the optimal learning rate. If the learning rate is set too high, the model may update the weight too drastically, leading to overshooting the optimal values, but a too low learning rate will either take too long to converge or get stuck in an undesirable local minimum, wheras our goal is to find the global minimum. A common apporach is to use negative powers of 10 multiplied by a coefficent, for example 3e-5 means 0.00003.
