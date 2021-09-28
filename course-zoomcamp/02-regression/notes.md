# Chapter 02 - Notes

<br />

This chapter focused on working with Linear Regression by using [Cars Dataset](https://www.google.com)

## Exploratory Data Analysis 

The first part of the lesson gives you an overview of Exploratory Data Analysis. 

### Highlights 

- For consistency and better addressing of columns, replace spaces or any other special character. 

```python
.lower()           # all lowercase 
.replace(' ', '_') # replace space by _
```

* Check data types, so you know what pipelines and transformations need to be created for data manipulation.

```python
df.dtypes # give you types of columns
```

* Check for unique values across your features. 

```python
for col in df.columns:
    df[col].nunique() # count amount of unique values
```

* Plot the distribution of your data. 

```python
sns.histplot(df.msrp[df.msrp < 100000], bins=50)
```

As the distribution is long tailed, 

## Validation Framework 

As previously stated, when building a model we would like to evaluate the model after training but we don't want to do it over the same traninig dataset. Because of that it's necessary to split the data. 

### Highlights
 
- Split the data into 3 sets, training, validation and test set. 

- Allocate a fair amount for traning set ( 60 % ) of your data

```python
df.iloc[:n_train] # grab indices from beginning to size of training set
```

- Shuffle the indices of data before doing the split.

```python
idx = np.arange(n)     # grab all the indices
np.random.seed(2)      # set a seed so it will be reproducible  
np.random.shuffle(idx) # shuffle the indices
```

- Reset the index after allocating the data for each set

```python
df_train.reset_index(drop=True)
```

## Linear Regression

On this section, Linear regression form was shown for a particular observation. 

```python
df_train.iloc[10] # Grab observation #10

```

- Ideally we would like to take index  $x_i$ and produce a prediction $y_i$

The function will be as: 

$\mathcal{g}(x_i) = y_i$

And the formula to apply $\mathcal{g}(x_i)$ is defined as: 

$\mathcal{g}(x_i) = w_0 + w_1x_{i_1} + w_2x_{i_2} + w_3x_{i_3}$ 

Where : 
- $W_0$ is the prediction of the car without adding any feature. 
- $W_1$ weight corresponding to feature $x_i$

$y_i = w_i * x_i + b_i$