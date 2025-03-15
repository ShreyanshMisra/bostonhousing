# Multiplie Linear Regression on King County 

Multiple Linear Regression (MLR) is a statistical method used to estimate the relationship between a dependent variable and multiple independent variables, assuming a linear relationship between them.

The MLR equation is given by:

$$Y=\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \varepsilon$$

where $\beta_0$ is the intercept, and $\beta_1$ and $\beta_2$ are the coefficients determined by minimizing the sum of the squared differences between observed and predicted values.Once fitted, the model can be used to predict $Y$ for given $X_n$ values.

This study focuses on predicting home prices in King County, Washington using Multiple Linear Regression. The dataset, sourced from Kaggle (www.kaggle.com/datasets/shivachandel/kc-house-data), contains 21,613 records of homes sold between May 2014 and May 2015. We specifically analyzed nine predictors to estimate house prices. 

Our investigation leveraged several R packages for data preprocessing, visualization, and modeling. `ggplot2` was used for data visualization, while `tidyr` and `dplyr` helped with data transformation. The `caret` package was used machine learning workflows, and `car`, `lmtest`, and `nlme` were used for regression and linearity testings.

We find that the logarithmic function of 8 of these predictors are a very strong linear predictor of the logarithmic function of price.

# Conclusion

Our multiple linear regression model effectively predicts the outcome variable based on eight key predictors. The model passed linearity tests and demonstrated moderate explanatory power with an R-squared of 0.4769. The log transformation improved the linearity and distribution of residuals, ensuring a more accurate fit.

The model provided meaningful insights into how different factors influence the outcome variable. Key predictors such as `sqft_living`, `grade`, and `view` have significant positive impacts on price. However, the presence of heteroscedasticity suggests variability in residuals, which could impact predictive consistency. Furthermore, an R-squared of 0.4769 means over 50% of the variance remains unexplained.

Overall, while our model performs well within its scope, some future refinements can improve accuracy and generalizability.

# References
- https://www.kaggle.com/datasets/shivachandel/kc-house-data
- https://www.sthda.com/english/articles/40-regression-analysis/168-multiple-linear-regression-in-r/
