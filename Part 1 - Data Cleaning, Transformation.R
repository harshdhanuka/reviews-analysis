

# Team FINANCE 3

# Project - Deliverable 1  -  lean and Prepare Dataset for Analysis and Prediction

# DATA SET - https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

# The aim is to Predict the Rating of the clothes.


RNGversion(vstr = 3.6)

rm(list=ls())

library(ISLR); library(ggplot2); library(caret); library(caTools); library(tidyr); library(dplyr); library(lm.beta);
library(leaps); library(car); library(mice);
library(gbm); library(glmnet); library(car); library(cluster); library(mclust)
library(rapportools); library(data.table); library(ngram); library(stringr); library(corrplot);


# read the data set
getwd(); setwd("/Users/harshdhanuka/Desktop/R Project/")
data = read.csv('Womens Clothing Reviews.csv',  na.strings = c("NA", "N/A", ""))


# evaluate the structure and contents of the data set 
str(data)
summary(data)
names(data)
dim(data)


#################################################################################

# DATA EXPLORATION, CLEANING and TRANSFORMATION
 

# Check Astronomical Values

ggplot(data=data, aes(x=Age)) + 
  geom_histogram(fill="blue", binwidth = 10)
table(data$Age)

ggplot(data=data, aes(x=Rating)) + 
  geom_histogram(fill="blue", binwidth = 10)
table(data$Rating)

ggplot(data=data, aes(x=Recommended.IND)) + 
  geom_histogram(fill="blue", binwidth = 10)
table(data$Recommended.IND)

ggplot(data=data, aes(x=Positive.Feedback.Count)) + 
  geom_histogram(fill="blue", binwidth = 10)
table(data$Positive.Feedback.Count)

# No astronomical values detected, no changes made


################################################################################

# Check Outliers

# No outliers in the data set as per summary of the dataset, no changes made


###############################################################################

# Check high correlation amongst variables

# first we isolate the numeric variables 
numericVars = which(sapply(data, is.numeric)) #index vector numeric variables

# then construct a correlation data frame
data_numVar = data[, numericVars]
cor_numVar = cor(data_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

# sorting on decreasing correlations with Rating
cor_sorted = as.matrix(sort(cor_numVar[,'Rating'], decreasing = TRUE))
print(cor_sorted)

# selecting only high corelations, greater than 0.1
CorHigh = names(which(apply(cor_sorted, 1, function(x) abs(x)>0.1)))  # greater than 10% as correlation check factor
cor_numVar = cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")

# These results might impact our feature selction process moving forward

# No high correlation, no need to drop any variables later


#########################################################################

# Handling NA's and Null's

# check NA's
sum(is.na(data$Title))
sum(is.na(data$Review.Text))
sum(is.na(data$Department.Name))
sum(is.na(data$Class.Name))

# Removing all NA rows, as they are only 16% in Title, 4% in Review.Text, and 0.06% in Division.Name, Department.Name and Class.Name respectively
data = na.omit(data)

# no data imputation required


# Save Non-tokenized dataset for simple text analysis in Section 1 of the Project 1.2
write.csv(data, "Clean_Womens_Reviews_Simple.csv",row.names = F)

########################################################################

# DATA PREPARATION

# For CLUSTER Analysis on the Age Group, we do not need further preparation 

## ## ## ## ##

# Prepare for  SENTIMENT ANALYSIS and RATING PREDICTION - TOKENIZE

# for Review.Text

# 1 -- Create a corpus from the variable 'Review.Text'
# install.packages('tm')
library(tm)
corpus = Corpus(VectorSource(data$Review.Text))

# 2 -- Use tm_map to 
#(a) transform text to lower case, 
corpus = tm_map(corpus,FUN = content_transformer(tolower))

#(b)URL'S
corpus = tm_map(corpus, FUN = content_transformer(FUN = function(x)gsub(pattern = 'http[[:alnum:][:punct:]]*',
                                                                replacement = ' ',x = x)))

#(c) remove punctuation, 
corpus = tm_map(corpus,FUN = removePunctuation)

#(d) remove English stopwords using the following dictionary tm::stopwords('english) 
corpus = tm_map(corpus,FUN = removeWords,c(stopwords('english')))

#(e) remove whitespace
corpus = tm_map(corpus,FUN = stripWhitespace)

# 3 -- Create a dictionary
dict = findFreqTerms(DocumentTermMatrix(Corpus(VectorSource(data$Review.Text))), lowfreq = 0)
dict_corpus = Corpus(VectorSource(dict))

# 4 -- Use tm_map to stem words
corpus = tm_map(corpus,FUN = stemDocument)

# 5 -- Create a DocumentTermMatrix
dtm = DocumentTermMatrix(corpus)

inspect(dtm)
dim(dtm)

## ## ## ## ##

# for Title

# 1 -- Create a corpus from the variable 'Title'
corpus2 = Corpus(VectorSource(data$Title))

# 2 -- Use tm_map to 
#(a) transform text to lower case, 
corpus2 = tm_map(corpus2,FUN = content_transformer(tolower))

#(b) URL'S
corpus2 = tm_map(corpus2,FUN = content_transformer(FUN = function(x)gsub(pattern = 'http[[:alnum:][:punct:]]*',
                                                                replacement = ' ',x = x)))

#(c) remove punctuation, 
corpus2 = tm_map(corpus2,FUN = removePunctuation)

#(d) remove English stopwords using the following dictionary tm::stopwords('english) 
corpus2 = tm_map(corpus2,FUN = removeWords,c(stopwords('english')))

#(e) remove whitespace
corpus2 = tm_map(corpus2,FUN = stripWhitespace)

# 3 -- Create a dictionary
dict2 = findFreqTerms(DocumentTermMatrix(Corpus(VectorSource(data$Title))),lowfreq = 0)
dict_corpus2 = Corpus(VectorSource(dict2))

# 4 -- Use tm_map to stem words
corpus2 = tm_map(corpus2,FUN = stemDocument)

# 5 -- Create a DocumentTermMatrix
dtm2 = DocumentTermMatrix(corpus2)

inspect(dtm2)
dim(dtm2)


#########################################################################

# REMOVE SPARSE TERMS

# We will remove those words which appear in less than 5% of the reviews

xdtm = removeSparseTerms(dtm,sparse = 0.95)
xdtm

xdtm2 = removeSparseTerms(dtm2,sparse = 0.95)
xdtm2


##########################################################################

# COMPLETE THE STEMS AND SORT THE TOKENS

xdtm = as.data.frame(as.matrix(xdtm))
colnames(xdtm) = stemCompletion(x = colnames(xdtm),
                                dictionary = dict_corpus,
                                type='prevalent')
colnames(xdtm) = make.names(colnames(xdtm))

sort(colSums(xdtm),decreasing = T)  # sort to see most common terms

## ## ## ## ##

xdtm2 = as.data.frame(as.matrix(xdtm2))
colnames(xdtm2) = stemCompletion(x = colnames(xdtm2),
                                dictionary = dict_corpus,
                                type='prevalent')
colnames(xdtm2) = make.names(colnames(xdtm2))

sort(colSums(xdtm2),decreasing = T)  # sort to see most common terms


##########################################################################

# write.csv(data, "Clean_Womens_Reviews.csv",row.names = F)

# data = read.csv('Clean_Womens_Reviews.csv')


####################################   T H E   E N D   ####################################



