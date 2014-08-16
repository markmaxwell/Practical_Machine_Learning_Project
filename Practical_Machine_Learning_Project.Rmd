Practical Machine Learning Project 
========================================================

### Abstract
Using Human Activity Recognition (HAR) data that was collected from sensors on the human body while a person performs an exercise, a prediction model was constructed to determine what activity was conducted based off the sensor data provided. 

After pre-processing the training data and reducing the data to useable features, several models were fit to the training data and their in-sample accuracy was assessed. The final model that was selected was a Random Forest model (method = rf) that had a 99.9% in-sample accuracy rate and provided a cross-validation of 20 out of 20 correctly matched predictions against the testing dataset. The other model that was testing and produced the same cross-validation accuracy of 20 out of 20 prediction match rate was the Generalized Boosted Model (method = gbm) and had a 98.6% in sample accuracy rate. The analysis below documents the steps take to achieve the "best fit" Random Forest model.

More information on the data and other HAR studies can be found here: http://groupware.les.inf.puc-rio.br/har


```{r,echo=TRUE,eval=TRUE,message=FALSE}
library(caret)
library(randomForest)
library(ggplot2)
library(e1071)
library(rpart)
library(MASS)
library(markdown)
library(knitr)
```

Read in the training and testing data. Because the data was already separated into two separate training and testing files there was no need to partition the data.

```{r}
# Set the Working directory for the project
setwd("C:/Users/Mark Maxwell/Desktop/Coursera/Data Science Specialization Track/08. Practical Machine Learning/Course Project/Data")

# Read in the testing dataset
testing <- read.csv("pml-testing.csv",header = TRUE)

# Read in the training dataset
training <- read.csv("pml-training.csv",header = TRUE)
```

### Preprocess the training data. 
Processing the dataset included removing variables that had no predictive power such as the name of the person performing the exercise and the timestamp that the exercise was performed. Additionally, the dataset contained a number of variables that were mostly NAs or were persisted in the training dataset as blanks. These variables were removed from the dataset before any models were fit to remove noise and decrease the time needed to compute each model.


```{r}
# Remove variables form the training dataset that have no predictive power
training <- droplevels(training[,!names(training) %in% c("X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","user_name")]) 

# Reomve NAs form the training dataset
training <- droplevels(training[,colSums(is.na(training)) == 0])

# Reomve all the columns that are mostly empty cells
training <- droplevels(training[,colSums(training=="") == 0])
```

### Testing several models.
Several models were fit then their predictions were reviewed and checked for their in sample accuracy. Below is the code for the process used to test each model but only the final Random Forest model is evaluated below in this document.

```
# Linear Discriminate Analysis Model
LDAmodelFit <- train(classe ~ ., method = "lda", data = training)
predict(LDAmodelFit,testing)
LDAmodelFit

# Naive Bayes Model
NBmodelFit <- train(classe ~ ., method = "nb", data = training)
predict(NBmodelFit,testing)
NBmodelFit

# General Boosted Model
GBMmodelFit <- train(classe ~ ., method = "gbm", verbose = FALSE, data = training)
predict(GBMmodelFit,testing)
GBMmodelFit

# Classification Tree
RPARTmodelFit <- train(classe ~ ., method = "rpart", data = training)
predict(RPARTmodelFit,testing)
RPARTmodelFit

# Random Forest Model
RFmodelFit <- train(classe ~ ., method = "rf", data = training)
predict(RFmodelFit,testing)
RFmodelFit
```

### Final Model Selection.
The final model that was selected for use was the Random Forest model that had a 99.9% in sample accuracy rate and provided a cross-validated match rate of 20 out of 20 correctly matched predictions against the testing dataset. 

*The model results below are printed out in the markdown file so they would not have to be recomputed every time the document is knit to HTML.

```{r,echo=TRUE,cache=TRUE}
RFmodelFit <- train(classe ~ ., method="rf", data=training)
```

```{r,echo=TRUE}
predict(RFmodelFit,testing)
```

```{r,echo=TRUE}
RFmodelFit 
```

```{r,echo=TRUE}
RFmodelFit$finalModel
```