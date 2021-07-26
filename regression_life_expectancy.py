# -*- coding: utf-8 -*-
"""
Created on Fri Oct 9  12:14:26 2020

@author: Vanessa Khoo
"""

"""
In this assignment we create a Python module
to perform some basic data science tasks. While the
instructions contain some mathematics, the main focus is on 
implementing the corresponding algorithms and finding 
a good decomposition into subproblems and functions 
that solve these subproblems. 

To help you to visually check and understand your
implementation, a module for plotting data and linear
prediction functions is provided.

The main idea of linear regression is to use data to
infer a prediction function that 'explains' a target variable 
of interest through linear effects of one 
or more explanatory variables. 

Part I - Univariate Regression

Task A: Optimal Slope

-> example: price of an apartment

Let's start out simple by writing a function that finds
an "optimal" slope (a) of a linear prediction function 
y = ax, i.e., a line through the origin. A central concept
to solve this problem is the residual vector defined as

(y[1]-a*x[1], ..., y[1]-a*x[1]),

i.e., the m-component vector that contains for each data point
the difference of the target variable and the corresponding
predicted value.

With some math (that is outside the scope of this unit) we can show
that for the slope that minimises the sum of squared the residual

x[1]*(y[1]-a*x[1]) + ... + x[m]*(y[m]-a*x[m]) = 0

Equivalently, this means that

a = (x[1]*y[1]+ ... + x[m]*y[m])/(x[1]*y[1]+ ... + x[m]*y[m])

Write a function slope(x, y) that, given as input
two lists of numbers (x and y) of equal length, computes
as output the lest squares slope (a).


Task B: Optimal Slope and Intercept

To get a better fit, we have to consider the intercept b as well, 
i.e., consider the model f(x) = ax +b. 
To find the slope of that new linear model, we \centre the explanatory variable 
by subtracting the mean from each data point. 
The correct slope of the linear regression f(x)=ax + b is the same 
slope as the linear model without intercept, f(x)=ax, calculated on the 
centred explanatory variables instead of the original ones. 
If we have calculated the correct slope a, we can calculate the intercept as
b = mean(y) - a*mean(x).

Write a function line(x,y) that, given as input
two lists of numbers (x and y) of equal length, computes
as output the lest squares slope a and intercept b and
returns them as a tuple a,b.


Task C: Choosing the Best Single Predictor

We are now able to determine a regression model that represents 
the linear relationship between a target variable and a single explanatory variable.
However, in usual settings like the one given in the introduction, 
we observe not one but many explanatory variables (e.g., in the example `GDP', `Schooling', etc.). 
As an abstract description of such a setting we consider n variables 
such that for each j with 0 < j < n we have measured m observations 

$x[1][j], ... , x[m][j]$. 

These conceptually correspond to the columns of a given data table. 
The individual rows of our data table then become n-dimensional 
data points represented not a single number but a vector.

A general, i.e., multi-dimensional, linear predictor is then given by an n-dimensional 
weight vector a and an intercept b that together describe the target variable as

y = dot(a, x) + b 

i.e., we generalise y = ax + b by turning the slope a into an n-component linear weight vector
and replace simple multiplication by the dot product (the intercept b is still a single number).
Part 2 of the assignment will be about finding such general linear predictors. 
In this task, however, we will start out simply by finding the best univariate predictor 
and then represent it using a multivariate weight-vector $a$. %smooth out with the text that follows.

Thus, we need to answer two questions: (i) how do we find the best univariate predictor, 
and (ii) how to we represent it as a multivariate weight-vector. 

Let us start with finding the best univariate predictor. For that, we test all possible
predictors and use the one with the lowest sum of squared residuals.
Assume we have found the slope a^j and intercept b^j of the best univariate predictor---and assume it 
uses the explanatory variable x^j---then we want to represent this as a multivariate 
slope a and intercept b. That is, we need to find a multivariate slop a such that dot(a, x) + b 
is equivalent to a^jx^j + b^j. Hint: The intercept remains the same, i.e., $b = b^j$.

Task D: Regression Analysis

You have now developed the tools to carry out a regression analysis. 
In this task, you will perform a regression analysis on the life-expectancy 
dataset an excerpt of which was used as an example in the overview. 
The dataset provided in the file /data/life_expectancy.csv.


Part 2 - Multivariate Regression

In part 1 we have developed a method to find a univariate linear regression model 
(i.e., one that models the relationship between a single explanatory variable and the target variable), 
as well as a method that picks the best univariate regression model when multiple 
explanatory variables are available. In this part, we develop a multivariate regression method 
that models the joint linear relationship between all explanatory variables and the target variable. 


Task A: Greedy Residual Fitting

We start using a greedy approach to multivariate regression. Assume a dataset with m data points 
x[1], ... , x[m] 
where each data point x[i] has n explanatory variables x[i][1], ... , x[i][m], 
and corresponding target variables y[1], ... ,y[m]. The goal is to find the slopes for 
all explanatory variables that help predicting the target variable. The strategy we 
use greedily picks the best predictor and adds its slope to the list of used predictors. 
When all slopes are computed, it finds the best intercept. 
For that, recall that a greedy algorithm iteratively extends a partial solution by a 
small augmentation that optimises some selection criterion. In our setting, those augmentation 
options are the inclusion of a currently unused explanatory variable (i.e., one that currently 
still has a zero coefficient). As selection criterion, it makes sense to look at how much a 
previously unused explanatory variable can improve the data fit of the current predictor. 
For that, it should be useful to look at the current residual vector r,
because it specifies the part of the target variable that is still not well explained. 
Note that a the slope of a predictor that predicts this residual well is a good option for 
augmenting the current solution. Also, recall that an augmentation is used only if it 
improves the selection criterion. In this case, a reasonable selection criterion is 
again the sum of squared residuals.

What is left to do is compute the intercept for the multivariate predictor. 
This can be done as


b = ((y[1]-dot(a, x[1])) + ... + (y[m]-dot(a, x[m]))) / m

The resulting multivariate predictor can then be written as 

y = dot(a,x) + b .



Task B: Optimal Least Squares Regression

Recall that the central idea for finding the slope of the optimal univariate regression line (with intercept) 
that the residual vector has to be orthogonal to the values of the centred explanatory variable. 
For multivariate regression we have many variables, and it is not surprising that for an optimal 
linear predictor dot(a, x) + b, it holds that the residual vector is orthogonal to each of the 
centred explanatory variables (otherwise we could change the predictor vector a bit to increase the fit). 
That is, instead of a single linear equation, we now end up with n equations, one for each data column.
For the weight vector a that satisfies these equations for all i=1, ... ,n, you can again simply find the 
matching intercept b as the mean residual when using just the weights a for fitting:

b = ((y[1] - dot(a, x[1])) + ... + (y[m] - dot(a, x[m])))/m .

In summary, we know that we can simply transform the problem of finding the least squares predictor to solving a system of linear equation, which we can solve by Gaussian Elimination as covered in the lecture. An illustration of such a least squares predictor is given in Figure~\ref{fig:ex3dPlotWithGreedyAndLSR}.
"""

from math import inf, sqrt

'''
Name: Vanessa Ming Yi Khoo
Student ID: 31417493
'''

# Task A
def slope(x, y):
    """
    Computes the slope of the least squares regression line
    (without intercept) for explaining y through x.
    
    Input: Two lists of numbers (x and y) of equal length representing data of 
           an explanatory and a target variable
    Output: The optimal least squares slope(a) with respect to the given data
        
    For example:
    >>> slope([0, 1, 2], [0, 2, 4])
    2.0
    >>> slope([0, 2, 4], [0, 1, 2])
    0.5
    >>> slope([0, 1, 2], [1, 1, 2])
    1.0
	>>> slope([0, 1, 2], [1, 1.2, 2])
	1.04
    
    
    4) What challenges does the function solve? Overall Approach?
    This is a problem that processes two lists where there is an arithmetic operation 
    to be done on each element of the two input lists, meaning loops are needed to iterate 
    over all elements.
    This function helps us to easily compute & find the optimal slope 'a' to model 
    the linear relationship between 1 explanatory variable and the target variable y, based on
    'm' observations. This slope 'a' minimises the sum of squared residuals.
    According to the specifications, the function should use the concept of 
    residual vectors, where we find the sum of squared residuals (r dot r), thus this again
    implies that a loop is required to iterate through each element of in x and y.

    
    5) Specific programming techniques choices:
    Since the equation given in the specifications is a dot product, a loop is needed to iterate 
    over all elements. Thus, I use a for loop to loop over each explanatory variable observation
    in x and each target variable in y to sum each product of x & y to a variable 'z', 
    which represents the dot product of x & y. 
    Also to loop over over each explanatory variable observation in x, and 
    sum the square of each observation to a variable 'd', which represents the dot product 
    of x & x.
    When accumulating each value, I use the augmented assignment statement += so that
    there isn't a need to create a new object to re-assign to each variable.
    These 2 variables 'z' and 'd' will be used after the for loop to compute the slope   
    After computing 'z' and 'd' after the for loop, I utilise the equation as stated in 
    the specifications to compute slope 'a' (a = z/d)
    In my implementation, I used a FOR loop to do my calculation, as I could
    iterate the body of code multiple times over using a for loop, and because there 
    would be a defined range of values to iterate over (length of list x). Also, I can 
    make sure that all values in my list 'x' is used (each iteration)
    This solution is very concise and contains code that very much relates to the problem and
    the equations given in the specifications.
    
    """  
    z = 0    # (x dot y)
    d = 0    # (x dot x)
    for i in range(len(x)):
        z += x[i]*y[i]
        d += x[i]**2
    a = z/d
    return a

# Additional function
def optimal_intercept(a, x, y):
    '''
    Function to compute the optimal intercept of a line that has a slope 'a' 
    based on centred data vector (x')
    
    Input: An optimal slope 'a', two lists of numbers (x and y) of equal length, 
           where one of the lists contains each observation's value of explanatory variable, 
           and the other containing the corresponding target variables.
    Output: The optimal intercept 'b' for this line (which is the average residual)
        
    For example:
    >>> a = 0.5
    >>> x = [0,1,2]
    >>> y = [1,1,2]
    >>> optimal_intercept(a,x,y)
    0.8333333333333334
    
    >>> a = 0.5
    >>> x = [2,1,2]
    >>> y = [3,5,4]
    >>> optimal_intercept(0.5,x,y)
    3.1666666666666665


    4) What challenges does the function solve? Overall Approach? 
    This is also a problem that processes two lists where there is an arithmetic operation 
    to be done on each element of the two input lists of same length (and slope 'a'), meaning 
    loops are needed to iterate over all elements. 
    I followed the equation given in the specifications for calculation 
    on the optimal intercept. The function is the result of decomposition of my line(x, y) 
    function. This so that my line(x, y) function will not have too many for loops/ lines of 
    code. This function helps easily derive the optimal intecept 'b' of the line that has a 
    slope 'a' based on the centred data vector. 
    
    5) Specific choices of programming techniques:
    In my implementation, I used a FOR loop to loop over each observation of the 
    explanatory variable and its corresponding target variable, to find the target value 
    minus the fitted value for each observation, and sum the result of this subtraction to 
    a variable 'b', which represents the total sum of residuals.
    I used a FOR loop for this, as I can iterate the body of code multiple times over using 
    a for loop, and because there would be a defined range of values to iterate over 
    (length of list x)
    Also, When summing to 'b' in each loop, I use the augmented assignment statement += 
    so that there isn't a need to create a new object to re-assign to each variable.
    To find the average residual, the variable 'b' will be divided by the number of
    observations.The function then returns the variable 'b'.
    This solution is very concise and contains code that very much relates to the problem.
    
    '''
    b = 0    # Declare variable
    for i in range(len(x)):
        b += (y[i] - a*x[i])
    b /= len(x)       # Divide by length to get average residual
    return b

# Task B
def line(x, y):
    """
    Computes the least squares regression line (slope and intercept)
    for explaining y through x.
    
    Input: Two lists of numbers (x and y) of equal length representing data of 
    an explanatory and a target variable.
    Output: A tuple (a, b) where 'a' is the optimal least squares slope and 'b'
    is the optimal intercept with respect to the given data.

    For example:
    >>> a, b = line([0, 1, 2], [1, 1, 2])
    >>> round(a,1)
    0.5
	>>> round(b,2)
	0.83
    

    4) What challenges does the function solve? Overall Approach?
    This is also a problem that processes two list where there is an arithmetic operation 
    to be done on each element of the two input lists of the same length, meaning loops are needed to iterate 
    over all elements. I followed the equation given in the specifications for arithmetic 
    calculation on the optimal intercept. This function helps to use orthogonality to 
    centred version of the data (X' dot r = 0) (x' is the centred data vector defined in
    the specifications), to find the optimal slope 'a'. 
    Also, this function helps us find a intercept 'b' so that our line is not forcefully 
    passing through the origin (with help from an extra function I wrote: optimal_intercept(a,x,y), 
    which returns the intercept 'b')
    
    
    5) Specific programming techniques choices:
    In my implementation, I used an additional function I wrote: mean_of_data(x), to find 'u', 
    whcih is the mean of the explanatory data.
    After computing u, I use a FOR loop to Loop over each explanatory variable observation 
    and each target variable observation, and sum the dot product (x' dot y) of each centred
    data (x + u) and y, and store the result in 'n1'
    Also, to sum the square of each centred data (x')^2, and store the result in 'd1''
    I used a FOR loop for this, as I can iterate the body of code multiple times over 
    using a for loop, and because there would be a defined range of values to iterate over 
    (length of list x). Also, this makes sure that all the values in my list is used to 
    compute the required values.
    When summing to 'n1' and 'd1' in each loop, I use the augmented assignment statement += 
    so that there isn't a need to create a new object to re-assign to each variable.
    I decomposed my function (created a new function optimal_intercept()), so that the code
    is shorter and more easily readable.
    This optimal_intercept() function returns an intercept 'b' based on its inputs
    (This function is explained in further detail in its pwn docstring).
    Thus, with 2 for loops, and use of another funciton, this solution is very concise and 
    contains code that very much relates to the problem and the equations given in 
    the specifications.
    
    """
    # constant u for x' (x' -> centred data vector)
    u = mean_of_data(x)    # mean_of_data function used here
    
    # Find optimal slope a - using centred data vector
    n1 = 0 # n1 is the numerator (x' dot y)
    d1 = 0 # d1 is the denominator (x' dot x')
    for i in range(len(x)):
        n1 += (x[i] - u)*y[i]
        d1 += (x[i] - u)**2
    a = n1/d1
    
    b = optimal_intercept(a, x, y) # Find b (optimal intercept)
  
    return a, b

def mean_of_data(x):
    '''
    Computes the mean of explanatory data (list x).
    
    Input: A list X representing the data for observations of an explanatory variable.
    Output: The mean (u) of the explanatory data 'x'.
    
    For example:
    >>> x = [25.97, 18.5, 18.94, 25.69, 25.71, 20.99, 36.61, 22.99, 17.84, 35.33, 18.41, 10.62, 20.7, 45.57, 23.22, 16.27, 11.84, 40.34, 26.31, 25.81]
    >>> u = mean_of_data(x)
    >>> u
    24.383
    
    >>> x = [6.8, 1.8, 5.9, 2.7, 4.2, 1.6, 1.9, 1.2, 4.5, 2.4, 3.0, 7.3, 2.0, 1.8, 6.4, 6.7, 2.2, 5.1, 1.0, 8.0]
    >>> u = mean_of_data(x)
    >>> u
    3.825
    
    4) What challenges does this function solve? Overall Approach?
    This problem processes a lists of unknown size, where we have to perform arithmetic
    operation (sum) with each element in the list. This implies that a loop is needed to iterate 
    through all elements and compute the result of the operation. 
        
    5) Specific programming techniques choices:
    In my implementation, I used a FOR loop, to loop over each observation for the particular 
    explanatory variable, and add each data value to a variable 'u'.
    When summing each value, I use the augmented assignment operator += 
    so that there isn't a need to recreate the object. 
    After all summation, to return 'u' as the mean of the data, I will divide
    'u' by the number of values/ elements in the list x (which is the length of list x)
    This solution is concise and quick, and provides as a good function to be used in other 
    functions.
    
    6) Complexity:
    O(n), n == len(x)
    '''
    u = 0 # constant u for x' (x' -> centred data vector)
    for i in range(len(x)):
        u += x[i]
    u /= len(x)  # divide by length as u is the mean value of the explanatory data
    return u

# Task C
def best_single_predictor(data, y):
    """
    Computes a general linear predictor (for multi-dimensional data points) by picking
    the best univariate prediction model from all available explanatory variables.
    
    Input:  Table data with m > 0 rows and n > 0 columns representing data of n explanatory
            variables and a list y of length m containing the values of the target variable
            corresponding to the rows in the table.
    Output: A pair (a, b) of a list 'a' of length 'n' and a number 'b' representing a linear
            predictor with weights 'a' and intercept 'b' with the following properties:
            a) There is only one non-zero element of 'a', ie: there is one i in range(n) such that
               a[j]==0 for all indices j!=i.
            b) The predictor represented by (a, b) has the smallest possible squared error among
               all predictors that satisfy property a)
    
    For example:
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, b = best_single_predictor(data, y)
    >>> weights[0]
    0.0
	>>> round(weights[1],2)
	-0.29
	>>> round(b,2)
	2.14
    

    4) What challenges does the function solve? Overall approach?
    This problem is a problem that processes data in a table and a list, where a comparison
    between a characteristic of each column and the target is carried out.
    This characteristic is the univariate predictor, which can be computed with functions within
    the same module. This implies a loop has to be carried out to get each 
    column's values to compute its predictor. Thus, this function helps us to easily 
    compute to get the best univariate predictor from all available explanatory variables
      
    
    5) Specific programming techniques choices:
    In my implementation, I used decomposition and wrote another function min_SSR_predictors() 
    which basically does all the looping for me and returns the index of the column of the 
    explanatory variable that gives the best predictor, and also returns the complete list of 
    univariate predictors corresponding to each explanatory variable.
    (I explain what techniques I use for the looping in the docstring of said function)
    
    Once I get the index for the explanatory variable which gives the univariate predictor
    with the lowest SSR, I use this index to find the univariate predictor that corresponds
    to that explanatory variable.
    
    Since according to the specifications, ie:
      a) There is only one non-zero element of 'a', ie: there is one i in range(n) such that
         a[j]==0 for all indices j!=i.
    I use a FOR loop to append n '0's integers to the weight 'a', and make sure to put the slope
    of the best predictor at the correct index in my weight vector.
    
    This solution is very clear and contains code that very much relates to the problem and
    the equations given in the specifications.
        
    """
    n = len(data[0]) # number of variables
        
    min_SSR_index, univariate_predictors = min_SSR_predictors(data, y) # min_SSR_predictors function called here
        
    # Best univariate predictor found - univariate predictor with smallest squared residuals
    best_up = univariate_predictors[min_SSR_index]
    
    # Find multivariate predictor - since univariate predictor is a list [a,b]
    b = best_up[1]
    a1 = best_up[0]
    
    # Getting the weight vector
    a = [0.0 for i in range(n)]
    a[min_SSR_index] = a1        # Putting a1 (best slope) at the correct index in my weight vector

    return a, b

# Additional function
def min_SSR_predictors(data, y):
    '''
    Computes and return the index of the column of the explanatory variable that gives the best 
    predictor, as well as returns the list of computed univariate_predictors, with use of functions
    also in this module. (is: line function, SSR function)
    
    Input: Table data with m > 0 rows and n > 0 columns representing data of n explanatory
           variables and a list y of length m containing the values of the target variable
           corresponding to the rows in the table.
    Output: The index 'min_SSR_ind' of the column of the explanatory variable that gives the best predictor,
            and a list 'univariate_predictors' containing all univariate predictors of the data
    
    For example
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> min_SSR_predictors(data, y)[0]
    1
    
    >>> data = [[1,3],[4,6],[5,2]]
    >>> y = [3,2,4]
    >>> min_SSR_predictors(data, y)[0]
    1
        
    4) What challenges does this function solve? Overall Approach?
    This problem is a problem that processes data in a table and a list, where a comparison
    between a characteristic of each column and the target is carried out.
    This would be the univariate predictor, which can be computed with functions within
    the same module(line(x,y)). Thus, this implies a loop has to be carried out to get each 
    column's values to compute its predictor. Thus, this function helps us to easily 
    compute to get the best univariate predictor from all available explanatory variables
    
    
    5) Specific programming techniques
    I use a FOR loop to loop over:
    each explanatory variable's data (which is each column in data table given) for 
    each iteration. I then use another FOR loop (nested) to 'invert' each column 
    (observation for each explantory variable) so that each value in that column is put 
    into a list 'x' (for each iteration), to be used in the line(x,y) function later.
    After creating list 'x' containing list of observations for the specific explanatory 
    variable, I find the univariate preedictor for each iteration, using the line(x, y) function. 
    This univariate predictor is then appended to the univariate_predictors list using 
    the augmented assignment statement += so that there is no need for re-creation of an object.
    After that, I find the Sum of Squared Residuals for the univariate predictor that is just 
    found (for each iteration), using the SSR function I wrote (details on this found in its docstring) 
    and this SSR is appended to the SSR_list list using the augmented assignment statement 
    += as well to avoid any re-creation of objects.
    
    I used a FOR loop (outerloop), for my iterations, as I can iterate the body of code 
    multiple times over using a for loop, and because there would be a defined range of values 
    to iterate over (number of variables/ columns)
    This solution takes into account each univariate predictor and makes a final comparison
    (comparing their SSR) to compute the final result.

    '''
    n = len(data[0])
    # Declare lists for univariate predictors and SSRs for each univariate predictor
    SSR_list = []
    univariate_predictors = []
    
    # Finding which explanatory variable gives the best predictor
    # Looping over each column/explanatory variable (i here would represent each column)
    for i in range(n):
        x = [] # Create inverted list x, to put each column in 'data' table to become a list 
               # representing the observations of each explanatory variable in the data set
        for j in range(len(data)): # Looping over each row (j here would represent each row)
            x += [data[j][i]]    # Append observation of each explanatory variable here
            
        a, b = line(x, y)                 # Find the slope & intercept of each explanatory variable
        univariate_predictors += [[a,b]]  # Append a & b (univariate predictor) of explanatory variable to list
        
        # SSR function used here
        SSR_list += [SSR(a, b, x, y)]  # Find SSR of each univariate predictor and append it to list of SSRs (for all predictors)
        
    # Get index of explanoty varianle with the least SSR
    min_SSR_ind = SSR_list.index(min(SSR_list))
    
    return min_SSR_ind, univariate_predictors

# Additional function
def SSR(a, b, x, y):
    '''
    Function to get the sum of squared residuals of a univariate predictor
    
    Input: a float 'a' representing the slope of the predictor, an intercept/bias 'b' 
           of the predictor, a list 'x' representing the observations (values) for each 
           explanatory variable, a list 'y' representing the target variable for each explanatory
           variables
    Output:  The sum of squared residuals for the univariate predictor
    
    >>> a = 0.07142857142857129
    >>> b = 1.5000000000000007
    >>> x = [1, 2, 4]
    >>> y = [2, 1, 2]
    
    >>> SSR(a,b,x,y)
    0.6428571428571428
    
    >>> a = -0.2857142857142858
    >>> b = 2.1428571428571432
    >>> x = [0, 3, 2]
    >>> y = [2, 1, 2]
    
    >>> SSR(a,b,x,y)
    0.2857142857142857
        
    
    4) What challenges does this function solve? Overall Approach?
    This problem processes 2 lists and 2 float values, where an arithmetic operation is
    to be done for each element of the input lists of sizes of the same size. This implies
    that a loop is needed to iterate through all elements and compute the result of
    the operation (SSR). 
        
    5) Specific programming techniques choices:
    In my implementation, I used a FOR loop, to loop over each observation for the particular 
    explanatory variable, get the squared residual for each observation (according to the 
    equation given Task A's specifications).
    When summing each squared residual, I use the augmented assignment operator += 
    so that there isn't a need to recreate the object. This solution is concise and quick.
    
    '''
    res = 0
    for i in range(len(x)):
        res += ((y[i]) - ((a)*x[i] + b))**2
    return res

# Assignment: Part 2
# Additional Function
def transpose(data):
    """
    This function helps return the transposed version of the input data given.
    
    Input: A list of lists representing data.
    Output: A transposed version of the data, with the columns put as the rows and
            the rows as the columns.
      
    For example:
    >>> data = [[1, 0, 1], [5, 7, 8], [9, 1, 3]]
    >>> transposed = transpose(data)
    >>> transposed
    [[1, 5, 9], [0, 7, 1], [1, 8, 3]]
    
    >>> data = [[3,4,5,6],[7,8,9,12],[1,2,5,4]]
    >>> transposed = transpose(data)
    >>> transposed
    [[3, 7, 1], [4, 8, 2], [5, 9, 5], [6, 12, 4]]
    
        
    4) What challenges does this function solve? Overall Approach?
        Challenge: Many functions coded in this assignment module requires
        the data to be transposed. And in assignment part 1, to transpose my data, I used 
        2 for loops in each situation when the data needed to be tranposed. This was
        not a good way/ style of coding as it produced more lines of code many times.
        Thus, I coded this function.
        
        How this function solves the challenge: 
        Many functions in this assignment module I coded requires the data to 
        be transposed in order for me to implement my functions.
        Note that in my assignment part 1 functions, I had used a not 'clean' way of 
        transposing my data with for loops (meaning: not utilising list comprehension)
        Thus, in assignment part 2, I utilised list comprehension and implemented this in this
        function, so that there will shorter lines of code with list comprehension.
        
        
    5) Specific programming techniques choices:
        I used list comprehension to create a list instead of using 2 for loops. 
        Thus, compared to using 2 for loops with no list comprehension which 
        would result in more than 2 lines of code; so by using list comprehension there 
        is only one line in this function. Of course, the complexity of this function is
        the same as if we used 2 for loops without list comprehension.
    
    6) Complexity Analysis:
        Let n = length of data (rows), m = length of a row of data (columns).
        The outer for loop iterates 'm' number of times, and for each outer loop iteration, the
        inner loop iterates 'n' number of times. The inner loop will
        Therefore, the total number of iterations for adding an element to the result 
        list is m*n.
        Thus, this function is in O(n*m).
    """
    return [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]

# Assignment: Part 2
# Additional Function 
def multivariate_intercept(a, x, y):
    """
    This function helps compute the multivariate intercept of the multivariate 
    predictor with weights 'a' and observations 'x' and target 'y'.
    The calculation is based on equation (6) in the assignment specifications.
    
    Input:  A list of weights 'a' of a multivariate predictor, list of lists of all 
    observations for each explanatory variable 'x', and a list of the target values 'y'.
    Output: A multivariate intercept 'b' obtained based on 'a', 'x', and 'y'. 
            intercept = average_target value - average_fitted value (or average residual)
    
    For example:
    >>> a = [0.23432, -1.2345]
    >>> x = [[1,3,5], [5,6,2]]
    >>> y = [3,4,1]
    >>> b = multivariate_intercept(a, x, y)
    >>> round(b, 2)
    7.31
    
    >>> a = [0.21428571428571436, -0.2857142857142858]
    >>> x = [[1, 2, 4], [0, 3, 2]]
    >>> y = [2, 1, 2]
    >>> b = multivariate_intercept(a, x, y)
    >>> round(b, 2)
    1.64
    
    4) What challenges does this function solve? Overall Approach?
        Challenges it solves:
        - Solves the problem of calculating the multivariate intercept after finding the weights
          of a predictor in every function in this module that is required to return 
          the multivariate intercept b.
        - Solves the hassle of recoding the same for lines of code every tinme. 
          (in different functions)
        Overall approach:
        - To compute the intercept, some kind of loop (nested loop) would be needed to loop over each 
        x and a value for calculation. Since I need to compute a result 'intercept', the sum in each loop
        needs to accumulated onto the variables holding the value of the 'intercept' ('b' in my case)
         
    
    5) Specific programming techniques choices:
        - Im my implementation,
        - I used 2 for loops to loop through the data, to compute the required output.
        - 1 inner for loop is used to compute the dot product of the weights & the data 
          (this is the fitted value)
        - The outer for loop is used to find the target value - the fitted value for
          each observation, and then sum each to the variable 'b'
        - After the looping, I just divided the sum of all target values- fitted values
          by the number of lines/ explanatory variables to get the average residual.
          
        - Since this function is based on for loops, I utilisd
          list comprehension (for one of the loops) so that I can shorten my code and make it
          cleaner & clearer.
        - Also, since there is a need for an iteratively accumalted sum, 
          I used the function sum() to shorten the code & make it easier and readable.
        - I used += when computing for b in each outer loop, so that I would not need to re-assign 
          the variable b.
         
    
    6) Complexity analysis:
       - Let n = length of the list 'a', and m = length of the list 'y'
       - The outer for loop will iterate m times, and for each outer loop iteration,
         the inner for loop (list comprehension) will iterate n times in order to the create this 
         list -> O(m*n)
         The cost of sum() will also be in O(n), since it sums 'n' elements that are in the
         list. Thus, the list comprehension must be evaluated for it to be passed into the 
         sum function, the total cost would still be in O(n).
         
       - Therefore, the overall time complexity for this function is in O(m*n). (the dominating complexity)
       
    """
    m, n = len(y), len(a)  # num of observations, num of explanatory variables
    b = 0
    for i in range(m):
        q = sum([a[j]*x[j][i] for j in range(n)])  # find dot product for x*weights for each target observation
        b += (y[i] - q)                            # minus dot product from target value & sum to b
    b /= m      # Divide by number of explanatory variables to get average residual
    return b


# Assignment Part 2: Task A
def greedy_predictor(data, y):
    """
    This implements a greedy correlation pursuit algorithm. It computes a multivariate predictor using
    the greedy strategy similar to the one described in the Assignment specifications for Task A.
    Thus, this represents a greedy approach in computing the multivariate predictor.
    
    Input: A data table data of explanatory variables with m rows and n columns and a list of 
           corresponding target variables y.
    Output: A tuple (a,b) where a is the weight vector and b the intercept obtained by the greedy strategy.
        
    Examples:
    - I added 3 more examples for greedy_predictor (to ensure my algorithm chooses the
      correct predictor (local best) in each interation)
    - These examples were given by Ms Tan Hui Xuan & Mr Michael Kamp on ed discussion.
    - What these examples helped highlight was the possibility of more than 2 predictors
      having the same SSR (& thus both would be 'best' predictors locally). Thus, these 
      helped make sure my function chooses the best predictor in each iteration in these such cases.
      However, to note: In actual fact, the way I chose the predictors that have the same minmum SSR should not greatly affect
      the result.

    For example
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, intercept = greedy_predictor(data, y)
	>>> round(weights[0],2)
	0.21
	>>> round(weights[1],2)
	-0.29
	>>> round(intercept, 2)
	1.64
    
    >>> data = [[0, 0],
    ...         [1, 0],
    ...         [0, -1]]
    >>> y = [1, 0, 0]
    >>> weights, intercept = greedy_predictor(data, y)
    >>> round(weights[0],2)
    -0.5
	>>> round(weights[1],2)
	0.75
	>>> round(intercept, 2)
	0.75
    
    >>> data = [[0, 0, 1],[1, 0, 0],[0, -1, 0]]
    >>> y = [1, 2, 3]
    >>> weights, intercept = greedy_predictor(data, y)
    >>> weights, intercept
    ([0.7499999999999999, -1.4999999999999998, -0.37500000000000006], 1.375)
    
    
    >>> data = [[2, 0, 0, 2],[0, -2, 0, 3],[0, 0, 3, 0]]
    >>> y = [1, 2, 3]
    >>> weights, intercept = greedy_predictor(data, y)
    >>> weights, intercept
    ([-0.7499999999999999, 0.05357142857142854, -0.03571428571428564, -0.3214285714285714], 3.1071428571428563)
    
    >>> data = [[1, 3, 2],[2, 2, 1],[3, 4, 1]]
    >>> y = [3, 5, 7]
    >>> weights, intercept = greedy_predictor(data, y)
    >>> weights, intercept
    ([2.0, 0, 0], 1.0)
    
    4) What challenges does this function solve? Overall Approach?
       Challenges it solves:
       - Individual regressions (eg: best predictor in assignment part 1) will have a higher error
       compared to a multivariate regression model.
       - Thus, we can use greedy residual fitting to result with a multivariate regression model
         that will have a lower error.
      With a greedy approach, the problem is seen as a combinatorial optimization problem,
      where we have to pick the best predictors that optimise the multivariate predictor.
      Thus, there are three features of this problem:
           1) collection of objects (data, target)
           2) Feasibility constraint (predictors are evaluated from the dataset)
           3) Cost/Value function (SSR of predictor -> optimum = minimum SSR)
    
    
       Overall Approach:
       - This function utilises the greedy approach to greedily pick the best predictor (local best)
         and adds its slope to the list of predictors.
       - This algorithm then iteratively extends the partial solution (local solution) by a small augmentation 
         to optimise the selection criterion.
         This means that the algorithm iteratively picks a non-chosen explanatory variable / feature that best 
         fits the current residual for that iteration. The best feature is the one that has the minimum SSR.
         Note that: This predictor is only chosen & added to the list of predictors (making it non-zero in weight vector) 
         if this augmentation can improve the selection criterion. By improving the criterion, it means that after
         the new feature / weight is used to augment the predictor, the SSE (criterion) should be smaller than
         the previous SSR. Else, if the current iteration's best predictor's has a higher SSR compared to the 
         previous SSR, then this feature will NOT be used to augment the current predictor.
         
         Thus there is a need to evaluate over the total number of explanatory variables
         to produce the optimal weight vector. Therefore, this means that a loop is required to find
         the best predictor at each iteration, and choose whether or not to use it to augment the current predictor.
         
         Also, an inner loop would then be needed to compute each predictor and it's SSR to compare
         and find the the current best predictor for each outer loop iteration.
         
         Also, to compute the residual y to be used for the next iteration, another for loop would
         be needed to do this to generate each new residual after selecting a feature.
         
         After choosing the predictors, this function then computes the intercept for the multivariate
         predictor. Another looping would be then be needed to compute this intercept.
       
    5) Specific programming techniques choices:
       Copying:
       - I used list slicing like lst[:] to copy data.
       Transposing data:
       - I used an additional function transpose() that I wrote to transpose any data that needed to be 
       tranposed. The complexity of the transpose function is in O(n*m), where n == num of cols and
       m = num of rows of the data.
       List comprehension
       - When needing to create / compute lists, instead of using traditional for loops, I used
         list comprehension to shorten the lines of code and make the lines more readable.
      
        In my implementation for looping, I chose to iterate over the input data with a for-loop, since
        the number of iterations is known. I also used a for loop for the 1st inner loop, to compute each 
        predictor's SSR. In each inner loop iteration, to compute a particular predictor from the current
        data and current residual, I used the line() function as written in assignment part 1 to compute this.
        
        In my implementation in choosing the best predictor, I used the min() function to find the
        smallest SSR, and with that I can find the predictor that gave this smallest SSR. 
       
        If the predictor's SSR improves the selection criterion (is smaller than the previous SSR), 
        I used the usual indexing to update certain variables. 
        Also, I used the .pop() function to pop out the data of the predictor that has been used. 
        This is so that I won't be using this line's data in the next iteration to find the next iteration's 
        best predictor.
        Note: I did try to use sorted() on my current_data list based on SSR & use .pop(-1), as this 
              would reduce cost. However, to follow Michael Kamp's example input & output (that chose the 
              best_SSRi differently when 2 predictors have the same SSR), I had to adjust my code to 
              make it choose the first index of duplicate SSR predictors (2 predictors with same best SSR)
              when the list > 2, and the second index of the duplicate SSR predictors when the list == 2. 
              Thus, using best_SSRi to pop an unsorted current_data instead popping the last element of the
              sorted current_data, would be better in this case to keep track of my indexing. (even though cost/complexity
              of .pop(best_SSRi) in its worst-case [worst-case: when best_SSRi = 0], would be higher than .pop())
        
        Once the weights chosen have been finalized (after all iterations of OUTER for looping), 
        I utilise the multivariate_intercept() function I wrote from Assignment part 1 to compute
        the multivariate intercept b, based on the weights, original data and original targets.
        
        Therefore, 
        With this approach of using nested for loops and using decompostion thus using many of my written 
        functions to compute the result, this solution allows for greedy_approach to derive
        the multivariate predictor, with a clear tracking of indices and values.
    
    6) Complexity Analysis: 
        Let n = length of the input data's individual row (number of lines/variables)
        let m = length of input data's columns. (number of observations)
        Cost of tranpose(data) -> O(m*n)  (this was done twice in this function)
        
        Complexity of big for loop:
        The outer loop iterates 'n' times. 
        
        For the 1st inner for loop, in the worst case where every outer loop's best SSR is < previous SSR,
        (where all codes in last if statement (all updating) needs to be executed)
        the length of current_data reduces by 1 each time.
        
        For each 1st inner for loop iteration, the function line(current_data[i], y) will be executed, and
        the for loop in the line() function iterates for a total of 'm' times at each 1st inner loop iteration.
        For each 1st inner loop iteration, the function SSR(x, y, current_data[j], y) will be executed,
        and the for loop in the SSR function also iterates for a total of 'm' times at each 1st inner loop iteration.
        
        Therefore, the total cost of all 1st inner for-loop looping is 
        T(n) = n*2m + (n-1)*2m + (n-2)*2m +...(1)2m, where n == original size of current_data, m = number of observations
        T(n) = [n(n+1)/2]*(2m).
        Note/Example to understand the terms of the T equation above:
        for the term (n-1)*2m: (n-1) represents the length of current_data at current OUTER for loop iteration, 
        '2m' represents the cost of the line() and SSR() function at each of the (n-1) iterations of the inner loop. 
        Note: the size of current_data decreases at each outer iteration at the worst case, thus
        we can observe the part of the terms representing the size of current_data decreasing in the 
        following sequence: n*2m + (n-1)*2m +...(1)2m, where it decreases from n til a size of 1.
        
        Thus, since T(n) = [n(n+1)/2]*(2m), the complexity of the 1st inner for loop is in O(m*n^2).
        
        
        For the 2nd inner for loop within the main OUTER for loop,
        -> if it is not clear where the 2nd for loop is, it is:  
            [y[i] - (best_a*current_data[best_SSRi][i] + b)for i in range(len(y))]
        This inner loop will only happen if the if statement conditions hold true.
        So, the worst case is when new_SSR of every outer loop iteration is < SSR of previous iteration,
        thus this inner loop would iterate for every outer loop iteration in the worst case.
        When this happens, this inner loop will iterate 'n' times for every OUTER for loop iteration (n iterations)
        Thus, the cost of this line in the worst case is: T(n) = n*n.
        Therefore, the complexity of the 2nd inner for loop is in -> O(n^2).
        
        For this line in outer for loop, -> original_data.index(current_data[best_SSRi]), 
        This line will only execute if the if statement conditions hold true.
        So, the worst case is when the conditions hold true for every outer loop iteration.
        In this worst case, the worst case of .index() is when it has to iterate over the whole list to find the index.
        thus the number of times that it has to iterate depends on the size of current_data.
        For the worst case scenario of .index() happening at every outer loop iteration, 
        the size of current_data would decrease by 1 due to .pop(). Thus, the total cost 
        for this .index() is -> T(n) = n+(n-1)+(n-2)+...+1 = n(n+1)/2 = (n**2 + n) /2.
        Thus, the complexity for this line is in -> O(n**2), where n == original size of current_data/ number of features
        
        Similarly, for .pop() in outer for loop:
        This line will only execute if the if statement conditions hold true.
        Therefore, the worst case for .pop() is very similar to .index() as mentioned above. 
        Note that, in the worst case for .pop(best_SSRi), best_SSRi would be 0 for each outer loop
        iteration in the worst case, as it would have to iterate & update the entire list.
        Therefore, in the worst case, .pop() happens in all outer loop iterations and would give a total
        cost of T(n) = n+(n-1)+(n-2)+...+1 = n(n+1)/2 = (n**2 + n) /2.
        Thus, the complexity for this line is in -> O(n**2), where n == original size of current_data.
        
        
        Therefore: 
        The dominating complexity is then -> O(m*n^2), due to the 1st inner for-loop looping in the outer loop, 
        where m = number of observations, and n = number of lines/ explanatory variables.
 
    Conclusion:
        The worst case Big-O time complexity for this function is in O(m*n^2).
              
    """
    n = len(data[0])                # Number of lines/ variables
    weights = [0 for _ in range(n)] # Declare & initialise list for weight vector
    original_data = transpose(data) # Transposing the data - original of transposed is kept to find index
    current_data = transpose(data)  # Transposing the data - 2nd copy made to be changed when needed in for loop iteration
    original_y = y[:]               # Copy the original y
    current_SSR = sum([y[i]**2 for i in range(len(y))]) # Current SSR is the sum of the square of the result (error) of all observations
    
    # In each iteration, choose best predictor, find SSR, do updates if SSR is better then previous SSR
    for i in range(n):  # O(n)
        lst_lines = []; lst_SSR = []
        for j in range(len(current_data)):    # n + n-1 + n-2 +...1
            a, b = line(current_data[j], y)   # O(m*n^2)
            lst_lines.append(a)
            lst_SSR.append(SSR(a, b, current_data[j], y))  # O()
            
        new_SSR = min(lst_SSR)   # Finds minimum SSR
        # If there is more than 1 predictor with same SSR that is the minimum out of lst_SSR, and if the number of remaining lines to pick from == 2:
        if lst_SSR.count(new_SSR) > 1 and len(lst_SSR) <= 2: 
            best_SSRi = 0     # best_SSRi == 0 -> pick the first occurrence of best_SSR
        else:
            best_SSRi = lst_SSR.index(new_SSR)
            
        # If new_SSR < current_SSR:
        if new_SSR < current_SSR and current_SSR != 0: 
            current_SSR = new_SSR       # Update the current_SSR for next comparison
            best_a, index = lst_lines[best_SSRi], original_data.index(current_data[best_SSRi]) # Retrive corresponding slope best_a, and index of that slope
            weights[index] = best_a     # Update weight vector
            y = [y[i] - (best_a*current_data[best_SSRi][i] + b) for i in range(len(y))]        # Update y as new residual for next comparison
            current_data.pop(best_SSRi) # Pop the data with the just found lowest SSR
            
    b = multivariate_intercept(weights, original_data, original_y)  # Compute the multivariate intercept based on the computed weights
    return weights, b


# Assignment 2 - Task B
def centred_coeff(x1, x2, u):
    """
    Finds the specific coefficient for a term of particular data column.
    This is done by finding the dot product of the centred data vector of x1 and x2 inputs.
    This implements the multiplication of of x1 and x2 inputs.   
    
    Input:  x1 data input (not yet centred), and x2 data input (not yet centred)
    Output: A single coefficent represented by the dot product of x1 and x2 inputs.
    
    For example:
    >>> x1, x2, u = [0, 3, 2], [1, 2, 4], 1.6666666666666667
    >>> coeff = centred_coeff(x1, x2, u)
    >>> coeff = round(centred_coeff(x1, x2, u), 2)
    >>> coeff
    2.33
    
    >>> x1, x2, u = [0, 3, 2], [0, 3, 2], 1.6666666666666667
    >>> coeff = centred_coeff(x1, x2, u)
    >>> coeff = round(centred_coeff(x1, x2, u), 2)
    >>> coeff
    4.67
    
    """
    res = sum([(x1[i] - u)*(x2[i] - u) for i in range(len(x1))])
    return res

def equation(i, data, y):
    """
    Finds the row representation of the i-th least squares condition,
    i.e., the equation representing the orthogonality of the residual on data column i:

    (x_i)'(y-Xb) = 0
    (x_i)'Xb = (x_i)'y

    x_(1,i)*[x_11, x_12,..., x_1n] + ... + x_(m,i)*[x_m1, x_m2,..., x_mn] = <x_i , y>
    
    Thus, this function produces the coefficients and the right-hand side of the linear equation for explanatory
    variable i. 
    
    Input: Integer i with 0 <= i < n, data matrix data with m rows and n columns such that 
           m > 0 and n > 0, and list of target values y of length n.

    Output: Pair (c, d) where c is a list of coefficients of length n and d is a float 
            representing the coefficients and right-hand-side of Equation 8 for data column i.

    For example:
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
	>>> coeffs, rhs = equation(0, data, y)
	>>> round(coeffs[0],2)
	4.67
	>>> round(coeffs[1],2)
	2.33
	>>> round(rhs,2)
	0.33
	>>> coeffs, rhs = equation(1, data, y)
	>>> round(coeffs[0],2)
	2.33
	>>> round(coeffs[1],2)
	4.67
	>>> round(rhs,2)
	-1.33
    
    4) What challenges does this function solve? Overall Approach?
        Challenges it solves:
          - Finding the coefficients for a particular data column i.
          - Finding the right-hand side of the linear equation as shown in the equation above
          - solves the complex calculation for each data column in the least_squares_regression()
            function.
        Approach:
          - Using loops to iterate over data lists for calculation
          - Many loops are needed, hence, using additional functions to make calculation would be
            easier & less complex, and can help to reduce the amount of lines of code
        This is a list processing & arithmetic problem as well, where the equation of a particular data column
        needs to be computed. This data column is first transposed into a list (row).
        Then, to compute each element's (of the data column) coefficient, 
        I would need to use a loop, to find the dot product of two centred vectors, for each of 
        the data column's elements according to equation (8) in the assignment specifications.
        
        To compute the RHS of the equation, I would also need to use a loop, to get the
        dot product of 'y' with the specific centered data column x, and accumulate the sum 
        with the sum() function with list comprehension.
    
    5) Specific programming techniques choices:
       In my implementation:
       To transpose any data / data column needed, I used the transpose() function I had written.
       This is for simplicity of my code.
       Fo any of the lists to be computed (with arithmetic calculation) with loops, 
       like 'centred' and 'coeffs', I chose to use for-loops to iterate over the specific data
       column & thus use list comprehension to avoid having to write extended lines of code.
       
       Other than the tranpose() function, I also used another additionally written function
       called 'centred_coeff' which calculates a single specfic coefficient for a particular term
       in the LHS of the equation (of equation (8)).
       Thus, I used this function in a for-loop to compute a list containing each of the 
       coefficients of each term in equation (8). This for-loop was implemented with list
       comprehension as well.
       
       To sum up the dot-product of the centred data & 'y' to compute the RHS of the equation,
       I chose to use a for-loop to loop over each of the elements in y & the centred data.
       However, to accumlate each arithmetic result, I first put the result in a list with list 
       comprehension, then used sum() to sum the elements in the created list.
       This is such that I would not need to accumulate the values with a += statement.
      
       Note: I only centred & transposed the specific data column needed, 
       instead of centering the entire data.
       
       Therfore, this solution, provides a clear and concise way of computing
       the RHS & Coefficients of the equation that represents the orthogonality of
       the residual on a specific data column. 
    
    6) Complexity analysis:
       The complexity for this function is in O(n*m), where n = len(data[0]) (number of explanatory variables),
       and where m = len(y) (number of observations)
       
       1st for-loop: -> [centred_coeff(x, transposed_x[j], mean) for j in range(n)]
           The cost of the 1st for loop is T(n) = n*m.
           
           This is because the complexity for centred_coeff() function is from a for loop that iterates 
           m times where 'm' == len(transposed[i]) (the number of observarions).
           Thus, since the centred_coeff() call executes once per OUTER iteration of the OUTER for loop in equation(), 
           and there are 'n' number of outer iterations, and for each centred_coeff() execution, 
           the for loop within the centred_coeff() function will iterate 'm' times.
           Therefore, the complexity of the line:
               [centred_coeff(x, transposed_x[j], mean) for j in range(n)]
           is in -> O(n*m).
       
       2nd for-loop: -> [(data[j][i]-mean) for j in range(len(y))]
           The cost of the 2nd for loop is T(n) = m*1.
           This is because it iterates only m times, and in each iteration, the code being executed 
           is in constant time. Thus, complexity for this line is in -> O(m).
       
       3rd for-loop: -> sum([centred[j]*y[j] for j in range(len(y))]) 
           The total cost of the 3rd for-loop is T(n) = 2*m.
           This is because the line: [centred[j]*y[j] for j in range(len(y))], 
           iterates for a total of m times, and in each iteration of this for loop, the code being executed is
           in constant time.
           Also, the sum() function used: sum([centred[j]*y[j] for j in range(len(y))])
           will have a cost of 'm', since it is summing 'm' number of elements from the list.
           Thus, the TOTAL cost for the 3rd for-loop is -> T(n) = m + m = 2m.
           Therefore, complexity for this line is in -> O(m).
       
       The cost of this slicing/ copying in the line: x = transposed_x[i][:],
           is T(n) = m, where m == len(transposed_x[i]) == len(y). 
           Thus complexity for this line, is in -> O(m).
           This is because the cost for slicing == length of slice.
       
       The cost of transpose(data):
           is T(n) = m*n, where m == len(data) == len(y), n == len(data[0]).
           This is because the transpose function has an outer loop that iterates n times, and for every
           outer loop iteration, the inner loop iterates m times.
           Thus, complexity for transpose(data) is in -> O(m*n).
       
       The cost of mean_of_data(transposed_x[i]), 
           is -> m, where n == len(transposed_x[i]) == len(y).
           This is because the for loop in mean_of_data() iterates m times.
       
    Conclusion:
       Therefore, the dominating complexity is in O(n*m) and so the worst case Big-O is in O(n*m).
       
    """
    n = len(data[0]) # Number of explanatory variables/ features
    transposed_x = transpose(data); mean = mean_of_data(transposed_x[i]) # Transposing the data & finding the mean of the specified data column.
    x = transposed_x[i][:] # Copy of data of specific data column i
    
    coeffs = [centred_coeff(x, transposed_x[j], mean) for j in range(n)] # Find coefficients of each term for specified data column.
     
    centred = [(data[j][i]-mean) for j in range(len(y))] # Centred then transposed data
    rhs = sum([centred[j]*y[j] for j in range(len(y))])  # Compute RHS with centred then transposed data
    
    return coeffs, rhs

from copy import deepcopy
# Additional Function for Task B (least_squares_regression)
def triangular(a, b):
    """
    CITATION/SOURCE: FIT1045, Lecture 18 p.48, Monash University.
    
    Computes equivalent system in upper-triangular form of input system (general system)
    of linear equations by means of forward elimination. 
    This function's code is the code given during lectures regarding the topic of Gaussian Elimination.
    
    Input:   n*n matrix 'a' in general form, n-dim vector 'b' represented RHS of equations.
    Output:  New n*n matrix 'u' in upper-triangular form, new n-dim vector 'c' in upper-
             triangular form.
    
    For example
    >>> a = [[1,2,1],[-5,-9,-2],[2,3,1]]
    >>> b = [2,-3,3]
    >>> u, c = triangular(a,b)
    >>> u
    [[1, 2, 1], [0.0, 1.0, 3.0], [0.0, 0.0, 2.0]]
    >>> c
    [2, 7.0, 6.0]
    
    >>> a = [[2,1],[6, 4]]
    >>> b = [7,22]
    >>> u, c = triangular(a,b)
    >>> u
    [[2, 1], [0.0, 1.0]]
    >>> c
    [7, 1.0]
    
    >>> a = [[3,-2,5],[9,-2,16],[3,-10,1]]
    >>> b = [-1,0,-5]
    >>> u, c = triangular(a,b)
    >>> u
    [[3, -2, 5], [0.0, 4.0, 1.0], [0.0, 0.0, -2.0]]
    >>> c
    [-1, 3.0, 2.0]
    
    """
    u,c = deepcopy(a), deepcopy(b)
    n = len(u)
    for j in range(n):
        k = pivot_index(u, j)
        u[j], u[k] = u[k], u[j]
        c[j], c[k] = c[k], c[j]
        for i in range(j + 1, len(a)):
            q = u[i][j]/u[j][j]
            u[i] = [u[i][l]-q*u[j][l] for l in range(n)]
            c[i] = c[i] - q*c[j]
    return u, c


# Additional Function for Task B (least_squares_regression)
def pivot_index(u, j):
    """
    Finds the pivot in the gaussian forward elimination. (has to be non-zero)
    The right pivot to be chosen is essential in the implementation of forward elimination.
    
    Input:  n*n matrix 'u' of current coefficients, and index 'j' representing the 
            current pivot row.
    Output: New pivot row index 'k' that represents the row the found pivot is in.
        
    For example
    >>> u = [[2, 1], [6, 4]]
    >>> j = 0
    >>> k = pivot_index(u, j)
    >>> k
    0
    
    >>> u = [[1, 2, 1], [0.0, 1.0, 3.0], [0.0, 0.0, 2.0]]
    >>> j = 2
    >>> k = pivot_index(u, j)
    >>> k
    2
    
    >>> u = [[3, -2, 5], [0.0, 4.0, 1.0], [0.0, 0.0, -2.0]]
    >>> j = 2
    >>> k = pivot_index(u, j)
    >>> k
    2
    
    """
    k = j
    while u[k][j] == 0 and k < len(u):
        k += 1
    return k


# Additional Function for Task B (least_squares_regression)
def solve_by_back_substitution(u, b):
    """
    CITATION: FIT1045, Lecture 18 p.29, Monash University.
    (note: function name in lecture notes was: solve_backsub)
    
    Solves linear system ux=b for a square matrix u in upper-triangular form.
    To do this, implements backward substitution to solve upper-triangular systems.
    This function's code is the code given during lectures regarding the topic of Gaussian Elimination.
    
    Input:  n*n matrix 'u' of coefficients, n-dim vector 'b'; 'u' & 'b' are in upper-
            triangular form.
    Output: n-dim vector 'x' such that ux = b.
    
    For example
    >>> u = [[2, 1], [0, 1]]
    >>> b = [7, 1]
    >>> x = solve_by_back_substitution(u, b)
    >>> x
    [3.0, 1.0]
    
    >>> u = [[3,-2,5], [0,4,1],[0,0,-2]]
    >>> b = [-1,3,2]
    >>> x = solve_by_back_substitution(u, b)
    >>> x
    [2.0, 1.0, -1.0]
    
    """
    n = len(b)
    x = n*[0]
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(n-1, i, -1):
            s += u[i][j] * x[j]
        x[i] = (b[i]-s)/u[i][i]
    return x


# Additional Function for Task B (least_squares_regression)
def Gaussian(a, b):
    """
    CITATTION: FIT1045, Lecture 18 p.45, Monash University. 
    (note: function name in lecture notes was -> solve_gauss_elim)
        
    Computes the n-dim vector x such that ax = b, for 'a' & 'b' matrix (converted 
    into triangular form).
    This function's code is the code given during lectures regarding the topic of Gaussian Elimination.
    
    Input:   n*n matrix 'a' of coefficients, n-dim vector 'b' (RHS of equations)  
    Output:  n-dim vector x such that ax = b.
    
    For example
    >>> a = [[1,2,1],[-5,-9,-2],[2,3,1]]
    >>> b = [2,-3,3]
    >>> x = Gaussian(a,b)
    >>> x
    [3.0, -2.0, 3.0]
    
    >>> a = [[2,1],[6, 4]]
    >>> b = [7,22]
    >>> x = Gaussian(a, b)
    >>> x
    [3.0, 1.0]
    
    >>> a = [[3,-2,5],[9,-2,16],[3,-10,1]]
    >>> b = [-1,0,-5]
    >>> x = Gaussian(a, b)
    >>> x
    [2.0, 1.0, -1.0]
    
    """
    u, b = triangular(a, b)   # Implement triangular 
    return solve_by_back_substitution(u, b)  # Implements solve_by_back_substitution


# Assignment Part 2: Task B
# Convex optimisation: 
def least_squares_predictor(data, y):
    """
    This function finds the optimal least squares predictor for the given data matrix and
    target vector.
    
    It finds the least squares solution by:
    - centering the variables (still missing)
    - setting up a system of linear equations
    - solving with Gaussian's elimination
    
    Input: Data matrix data with m rows and n columns such that m > 0 and n > 0.
    
    Output: Optimal predictor (a, b) with weight vector a (len(a) == n) and intercept b such that
            a, b minimise the sum of squared residuals.
        
    For example:
	>>> data = [[0, 0], [1, 0], [0, -1]]
    >>> y = [1, 0, 0]
    >>> weights, intercept = least_squares_predictor(data, y)
    >>> round(weights[0], 2)
    -1.0
	>>> round(weights[1], 2)
	1.0
	>>> round(intercept, 2)
	1.0
    
    >>> data = [[1, 0],[2, 3],[4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, intercept = least_squares_predictor(data, y)
	>>> round(weights[0],2)
	0.29
	>>> round(weights[1],2)
	-0.43
	>>> round(intercept, 2)
	1.71
    
    4) What challenges does this function solve? Overall Approach?
       To find the optimal slopes (weights), the residual vector needs to be orthogonal to the
       values of the centred explanatory variables.
       Challenges it solves:
       - It transforms the problem of finding the least squares predictor to a problem of 
         solving a system of linear predictors, which can be solved using Gaussian's Elimination.
       - It helps find the equations (row representation) of each explanatory variable.
         This is done based on centred data, so that the residual vectors are orthogonal to the
         values of the centred explanatory variables.
       - It then helps find for the least squares predictor weights by solving a system of these
         linear equations, with utilisation of Gaussian's elimination.
       Approach:
       - The function needs to find each row representation of each equation for each explanatory
         variable in the dataset given. Thus, this means that a loop is needed to iterate over
         each data column (each explanatory variable), to retrieve each equation.
       - After finding the list(s) of equations, Gaussian's elimination would then have to be
         implemented in order to solve the system of linear equations, to find the weights.
       - After this, we need to find the intercept b as well. And this means that a loop is again
         needed so that we can find the intercept 'b'. (using the weights computed)
    
    5) Specific programming techniques choices:
       In my implementation,
       To compute the coefficients, I chose to use a for-loop (as the number of iterations is
       already known), to iterate over each data column i, and so I utilized the function 
       equation() to find the list of coefficients of each data column. Thus, this creates a list 
       of lists of coeffiecients for each data column i.
       To compute the targets (RHS of each equation), I chose to use a for-loop again (as the number
       of iterations is known), to iterate over each data column i, and utilizing the function
       equation() once again, to find the list of coefficients of each data column. Thus,
       this creates a list of targets (RHSs of each equation) corresponding to each data column i.
       
       Note that I chose to use list comprehension in each case, so that I may reduce the lines
       of code, and increase readability of my code.
       
       Next, to compute the weights (using gaussian's elimination), I utilized an additional function
       called Gaussian(), which will help me solve the given system of linear equations
       with the concept of matrix multiplication, forward elimination into triangular form,
       and backward substitution.
       
       Lastly, to compute the multivariate intercept, I used an additional function I had written 
       called multivariate intercept (that uses a for loop), which will help me compute the 
       intercept based on the optimal weights computed.
       
       Thus, the solution to this least_squares_regression problem of centering the data, finding
       the equations, then solving for weights using Gaussian's elimination, is clear & concise.
    
    6) Complexity Analysis:
       
       The complexity of transposing the data is in O(m*n), where m == len(y), n == len(data[0])
           This is because in the tranpose function, the outer for loop iterates 'n'  number of times, 
           and for each outer loop iteration, it will loop 'm' number of times.
           Thus, the total cost T(n) = n*m.
           Therefore, the complexity is in -> O(m*n).
       
       The cost of the line: [equation(i, data, y)[0] for i in range(n)],
           is T(n) = m*n*n, when n == len(data[0]), m == len(y).
           This is because the equation function has a complexity of O(n*m). (refer to equation()
           docstring analysis) for more detailed explanation.
           Since the equation function is executed 'n' times in the for loop within this function, 
           the cost of this for loop is = m*n*n.
           Thus, the dominating big-O complexity for this for-loop is in -> O(m*n^2).
       
       The cost of the line: [equation(i, data, y)[1] for i in range(n)], 
           is also T(n) = m*n*n, when n == len(data[0]), m == len(y).
           This is because the equation function has a complexity of n*m. (refer to equation()
           docstring analysis) for more detailed explanation.
           Since the equation function is executed 'n' times in the for loop within this function, 
           the cost of this for loop is = m*n*n.
           Thus, the dominating big-O complexity for this for-loop is also in -> O(m*n^n).
       
       The cost of the line: Gaussian(coeffs_x, target) is 
           T(n) = (n*n*n*1) + (n*n*2) + n*n + (1+2+3+...(n-2)+(n-1)).
           
           This is because (n*n*n*1) + (n*n*2) + n*n is the cost of triangular() in Gaussian(),
           And, (1+2+3+...(n-2)+(n-1)) is the cost of solve_by_back_substitution in Gaussian().
           
           So, T(n) = (n*n*n*1) + (n*n*2) + n*(k) + (n)(n-1)/2.
              
           Explanation for T(n) of triangular():
           T(n) = (n*n*n*1) + (n*n*2) + n*n
           - There is a nested 3-layer for loop, where for each outer iteration, 
           the inner loop iterates n times, and for each inner-loop operation, the
           2nd inner loop iterates n times again. 
           The cost of the computations in each loop is in constant time, thus is in O(n^3).
           - Also: note that the term (n*n) from the above terms represents the cost of pivot_index() in
           triangular(), where for every i-th outer loop iteration in triangular(), the worst
           case cost for pivot_index is 'n'.
           
           Explanation for T(n) of solve_by_back_substitution():
           T(n) = (1+2+3+...(n-2)+(n-1))
           - There is a outer for loop in this function that iterates n-1 times.
           - The inner for loop will iterate 1 time in the first outer iteration, 2 times for the second iteration
           .... up to (n-1) times for the last iteration of the outer loop.
           Thus, since the inner for loop is dependant on the 'n' and the outer iteration:
           T(n) = 1+2+3+...(n-2)+(n-1) = (n)(n-1)/2 == O(n**2).
           
           Thus,between O(n**3) and O(n**2), the dominating complexity is still in O(n**3) for Gaussian().
       
       The cost of the multivariate_intercept() calling,
           is T(n) = m*n
           This is because in the multivariate_intercept function:
           The outer loop iterates m number if times, where m == len(y) (number of observations)
           The inner loop iterates n number of times for every outer iteration, where n == len(weights).
           Therefore, 
           the big-O complexity for multivariate_intercept function, is in -> O(m*n).
       
    Conclusion:
       Therefore, by analysing & adding up all the T(n)s of each line of code, 
       we can conclude that there are 2 possible cases for the dominating complexity:
           1) O(m*n^2) from [equation(i, data, y)[1] for i in range(n)] line.
           2) O(n^3) from Gaussian() line.
           where:
               n == len(data[0]) (number of explanatory variables/features).
               m == len(y)  (number of observations).
       It would be ambiguous to choose a single dominating complexity BECAUSE the
       size of the input data would affect the above costs & complexities and cause one to be greater than
       another in different cases (as concluded in the last conclusion below).
       
    Thus, 
       1) If m >> n, then the dominating complexity and Big-O for this function is in O(m*n^2).
       2) If m << n, then the dominating complexity and Big-O for this function is in O(n^3).
       note: '>>' here means 'much more than'.
       
    """
    n = len(data[0]) # number of lines / explanatory variables
    transposed_x = transpose(data) # transposed data
    
    # Centering the variables & getting the equations
    coeffs_x = [equation(i, data, y)[0] for i in range(n)]  # LHS 'matrix' 
    target = [equation(i, data, y)[1] for i in range(n)]    # RHS targets 'matrix'
    
    weights = Gaussian(coeffs_x, target)  # Computes weights based on the system of linear equations given
    b = multivariate_intercept(weights, transposed_x, y) # Computers the multivariate intercept based on the weights found, the original (transposed) data, and the original targets.
    return weights, b


# Assignment Part 1: Task D
def regression_analysis():		
    """
	The regression analysis can be performed in this function or in any other form you see
	fit. The results of the analysis can be provided in this documentation. If you choose
	to perform the analysis within this funciton, the function could be implemented 
	in the following way.
	
	The function reads a data provided in "life_expectancy.csv" and finds the 
	best single predictor on this dataset.
	It than computes the predicted life expectancy of Rwanda using this predictor, 
	and the life expectancy of Liberia, if Liberia would improve its schooling 
	to the level of Austria.
	The function returns these two predicted life expectancies.
	
	For example:
	>>> predRwanda, predLiberia = regression_analysis()
	>>> round(predRwanda)
	65
	>>> round(predLiberia)
	79
    

    4) What challenges does the function solve? Overall Apprach?
    This problem is a data processing problem where other functions in the same module are used
    to compute results of each country as needed in the analysis. This function is just a 
    function to analyze the life-expectancy dataset given. Since the data to be read is in CSV, 
    this implies that a loop is needed to loop through each row in the CSV.
    As before finding the expected life expectancy for Rwanda and Liberia, the data
    for the explanatory variable values for Rwanda has to be 'extracted', same with data
    for Liberia and Austria, which means that other than using a loop earlier, I should also
    use the property of indexing to find the specific row of data that is needed.
    Also, an arithmetic operation needs to be done on each element of each countries's
    explanatory variable in order to compute the prediction (a dot x) + b, which means another loop is needed
    to compute this.
    
    
    5) Specific programming techniques choices:
    In my implementation,
    I first opened my file, and read the data from my file to be read using readline.
    To get the data for the explanatory variables, a loop is needed to go through each line, and: 
    append a list of float values of each row to 'table' list, append each value of 
    the 1st col of each row (represents the country) to 'countries' list, and append
    the target value (last col of each row) to 'target' list.
    I use a nested loop (inner loop) to change each value in the CSV into a float
    I then use the best_single_predictor(data, y) function to get the best single predictor 
    (parsing 'table' as the data & 'target' as the y variable list)
    The weight vector & bias from the output of best_single_predictor(), 
    will be used find/predict the expected life expectancy for Rwanda and Liberia
    Before finding the expected life expectancy for Rwanda and Liberia, the data
    for the explanatory variable values for Rwanda is 'extracted', same with data
    for Liberia and Austria.
    I use the countries list to find the index of Rwanda's values in the table data,
    then get the data from table by using this index
    I repeat this to get Liberia's and Austria's values as well
    Once I get the required data in specific lists for my analysis, 
    to predict the life expectancy for Rwanda & Liberia (as specified in the 
    specifications), I use a FOR loop to carry out calculation on the basis of equation (7) 
    given in the specifications, and use the augmented assignment operator += 
    so that there isn't a need to recreate the object.
    
    After getting the dot product, I add the bias to each sum, 
    to produce the predicted result for:
          a) Rwanda
          b) Liberia with schooling of Austria
    
    6) ANALYSIS:
    On Rwanda:
    -> The actual life expectancy of Rwanda is 65.2
       while my result is 65.39390006812778
    -> After rounding, my life expectancy for Rwanda is 65
    The predicted result is close to the actual target value.
    
    On Liberia:
    -> The actual life expectancy of Liberia is 61.1
       while the predicted result of Liberia with Austria's schooling is 79.09274688024462
    -> Thus, we can see that with a better schooling (Austria's schooling is better than Liberia),
       we could expect that the life expectancy will be greater. 
    
    Overall:
    -> The purpose of this entire module for part 1 of the assignment, 
    is to find the best single predictor from the life expectancy dataset and use this
    predictor for prediction.
    -> The best predictor, or the explanatory variable that gives the best univariate
    predictor for this life expectancy dataset, is Schooling.
    -> Thus, the explanatory variable Schooling plays as an important explanatory variable 
    since we will use its predictor/(line) in prediction. 
    As we can see from this analysis, that with better schooling, Liberia
    will have a greater life expectancy.
    
    """
    # Opening the file  
    with open("life_expectancy.csv", 'r') as file:
         headers = list(file.readline().split(","))
         table = []      # Explanatory variables values (used mostly for prediction)
         target = []     # Target values
         countries = []  # Countries
         for line in file.readlines():
             row = list(line.split(","))  # Split each row's values
             countries += [row[0]]        # Append each country to countries
             
             for i in range(1, len(row)): # For each data value in each row (col 1 to end)
                 row[i] = float(row[i])   # Change the type to float for calculation
             table += [row[1:-1]]  # Append each row of explanatory data to table (col 1 to 2nd last col)     
             target += [row[-1]]   # Append each target value to target list
    
    original_table = deepcopy(table)
    best = best_single_predictor(table, target)  # The best_single_predictor function called here
    
    # Getting index of the explanatory variable that gives the best univariate predictor
    index = (list(min_SSR_predictors(table, target))[0]) + 1 # min_SSR_predictor function called here
    best_predictor = headers[index]  # Best-predictor is -> Schooling
    # The above is just a piece of code for analysis to show which explanatory variable,
    # is the best predictor.
      
    weight_vector = best[0]   
    bias = best[1]
    
    # Rwanda - getting its explanatory variables values
    predRwanda = 0
    index_Rw = countries.index('Rwanda')
    variables_Rw = table[index_Rw]

    # Liberia (with Austria's schooling value replacing its schooling value)
    # - getting its explanatory variables values 
    predLiberia = 0
    index_Lib = countries.index('Liberia')
    variables_Lib = table[index_Lib]
    original_Lib = variables_Lib[:]
    # Austria - getting its explanatory variables values
    index_Aus = countries.index('Austria')
    variables_aus = table[index_Aus]
    
    index_schooling = headers.index('Schooling')
    variables_Lib[index_schooling-1] = variables_aus[index_schooling-1]  
    # -1 index cuz variable rows excluded the first column (countries), but headers list still has that 'countries' column as its first element
    
    # Univariate Regression Prediction calculation
    c = 0  # counter
    for c in range(len(weight_vector)):
        predRwanda += weight_vector[c]*variables_Rw[c]   
        predLiberia += weight_vector[c]*variables_Lib[c]
    predRwanda += bias
    predLiberia += bias
    
    # Greedy Prediction on Life expectancy dataset:
    weights, intercept = greedy_predictor(original_table, target)
    # print(weights, intercept)
    predR_Greedy = sum([weights[c]*variables_Rw[c] for c in range(len(weights))]) + intercept
    predL_Greedy = sum([weights[c]*original_Lib[c] for c in range(len(weights))]) + intercept
    
    # Optimal Least Squares Regression Prediction on Life expectancy dataset:
    weights, intercept = least_squares_predictor(original_table, target)
    # print(weights, intercept)
    predR_Least = sum([weights[c]*variables_Rw[c] for c in range(len(weights))]) + intercept
    predL_Least = sum([weights[c]*original_Lib[c] for c in range(len(weights))]) + intercept
    
    # print(predR_Greedy, predL_Greedy)
    # print(predR_Least, predL_Least)
    
    return predRwanda, predLiberia

#regression_analysis()


if __name__=='__main__':
    import doctest
    doctest.testmod()


