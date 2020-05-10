
# Team FINANCE 3

# Project Deliverable 2  -  Perform analysis on the dataset and build graphical representatons, predictions, etc.
# DATA SET - https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

# The OBJECTIVE is to perform exploratory analysis, predict the rating of the clothes, and do clustering analysis.
# Line 49   -  SECTION 1: Exploratory Analysis of different variables
# Line 120  -  SECTION 2: Exploratory Analysis of text column 'Review.Text' and numerical column 'Rating'
# Line 246  -  SECTION 3: Sentiment Analysis on text colum 'Review.Text', formation of Wordclouds
# Line 403  -  SECTION 4: Data Preparation for Predictive Modelling (TF, TF-IDF of text columns 'Review.Text' and 'Title'),
#                         and Exploratory Analysis from Corpus for 'Review.Text'
# Line 611  -  SECTION 5: Predictive Modelling (CART and Regression) using only text columns 'Review.Text' and 'Title'
# Line 734  -  SECTION 6: Clustering and Predictive Modelling using clustering techniques for non-text columns, dendogram for text columns
# Line 926  -  SECTION 7: Looking at Future, what else we could have done.

RNGversion(vstr = 3.6)
rm(list=ls())

# Load all necessary libraries
library(ggplot2); library(ggthemes); library(tidyr); library(dplyr)
library(cluster); library(mclust)
library(stringr); library(corrplot);
library(tidytext);library(janeaustenr); library(gridExtra)

# Read the cleaned data set from Project 1.1
getwd(); setwd("/Users/harshdhanuka/Desktop/R Project/")
data = read.csv('Clean_Womens_Reviews_Simple.csv', stringsAsFactors = F)

# Evaluate the structure and contents of the dataset 
str(data)
summary(data)

# Check column names
names(data)
# The first column 'X' is the original (given) serial number for the rows. We rename it to 'id' for simplicity
names(data)[1] = "id"

dim(data)
# Cleaned dataset with 19662 rows and 11 columns



###############################################################################################################
###############################################################################################################



## SECTION 1: Exploratory Analysis of different variables


# Part 1: Ratings  -  Number of Reviewers by Age (Age Group)


data$bins = cut(data$Age, breaks = c(0,20,40,60,80,100), labels = c("Centennials(0-20)","Young Adults(21-40)",
                                                                    "Adults(41-60)","Retired(61-80)","Traditionalists(81-100)"))
age_groups = data %>% select(bins,Age) %>% group_by(bins) %>% summarise(count = n())
ggplot(data=age_groups,aes(x=bins,y=count)) +  geom_bar(stat = "identity",fill="blue") +  
  labs(x = 'Age Groups', y = 'Number of Reviews')

##  -  Ages groups 21-40 are the used who use e-commerce the most, hence they have given the most reviews 
##  -  The lowest raters are the ones below 20 years, reasons maybe limited access to internet or devices
##  -  See visualization graph


######################################


# Part 2: Distribution of Departments where each Age Group tends to shop the most


age_groups_dept = data %>% select(bins,Class.Name, id)  %>% group_by(Class.Name, bins) %>% summarise(count = n())

ggplot(age_groups_dept, aes(x = bins, y = count,fill=Class.Name)) + geom_bar(stat='identity') +
  labs(x = 'Age Groups', y = 'Number of Reviews') +  theme(axis.text.x = element_text(angle = 90, hjust = 1))

##  -  'Dresses' are the most common, and are shopped by age groups 21 to 60
##  -  See visualization graph


######################################


# Part 3: Most Reviewed Products by 'Class.Name'


most_reviewed_products <- data %>% select(Class.Name) %>% group_by(Class.Name) %>% summarise(count = n()) %>% arrange(desc(count)) %>% head(10)
colnames(most_reviewed_products)[1] = "Class of Product"
colnames(most_reviewed_products)[2] = "Number of Reviews"
#install.packages('gridExtra')
library(gridExtra)
table1 = tableGrob(most_reviewed_products)
grid.arrange(table1,ncol=1)

##  -  We see that 'Dresses' top the list followed by 'Knits' and 'Blouses'
##  -  See visualization table 


######################################


# Part 4: Understanding the distribution of 'Rating' by 'Department.Name'


ggplot(data.frame(prop.table(table(data$Department.Name))), aes(x=Var1, y = Freq*100)) + geom_bar(stat = 'identity') + 
  xlab('Department Name') + ylab('Percentage of Reviews/Ratings (%)') + geom_text(aes(label=round(Freq*100,2)), vjust=-0.25) + 
  ggtitle('Percentage of Reviews By Department')

##  -  'Tops' have the highest percentage of reviews and ratings in this dataset, followed by 'dresses'. 
##  -  Items in the 'Jackets' and 'Trend' department received the lowest number of reviews.
##  -  See visualization graph



###############################################################################################################
###############################################################################################################



## SECTION 2: Exploratory Analysis of text column 'Review.Text' and numerical column 'Rating'


# Explore the numeric column 'Rating' and the text column 'Review.Text' and understand their statistical features and distribution

# Part 1: Ratings  -  mean and median


  # Mean and Median Ratings
data %>%
  summarize(Average_rating = mean(Rating), Median_rating = median(Rating))

  # Distribution of Ratings
ggplot(data = data, aes(x = Rating)) + geom_histogram(fill = 'black') + theme_grey() + coord_flip()

##  -  Average Rating = 4.18 and Median Rating = 5
##  -  Indicates most of the customers have rated all the different products positively, with higher ratings for most reviews
##  -  See visualization graph


######################################


# Part 2: Review.Text  -  Character, Words and Sentences counts for all Reviews


  # Characters
mean_characters = mean(nchar(data$Review.Text));
median_characters = median(nchar(data$Review.Text))

  # Words
mean_words = mean(str_count(string = data$Review.Text,pattern = '\\S+'));
median_words = median(str_count(string = data$Review.Text,pattern = '\\S+'))

  # Sentences
mean_sentences = mean(str_count(string = data$Review.Text,pattern = "[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"));
median_sentences = median(str_count(string = data$Review.Text,pattern = "[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"))

counts = data.frame(Variables = c("Characters", "Words", "Sentences"),
                    Mean = round(c(mean_characters, mean_words, mean_sentences),2),
                    Median = round(c(median_characters, median_words, median_sentences),2))
counts

##  -  The counts for each are more or less similar in their own mean and median
##  -  Implies that the counts distribution is highly symmetric and the skewless is low across the individual counts.


######################################


# Part 3: Review.Text length and Ratings  -  correlation


  # Characters
cor(nchar(data$Review.Text),data$Rating)
cor.test(nchar(data$Review.Text),data$Rating)

  # Words
cor(str_count(string = data$Review.Text,pattern = '\\S+'),data$Rating)
cor.test(str_count(string = data$Review.Text,pattern = '\\S+'),data$Rating)

  # Sentences
cor(str_count(string = data$Review.Text,pattern = "[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"),data$Rating)
cor.test(str_count(string = data$Review.Text,pattern = "[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"),data$Rating)

##  -  Cor for: Characters = -0.05478506, Words = -0.05622374, Sentences = 0.01813276
##  -  Low correlations for all three variables
##  -  Implies that the length of the 'Review.Text' do not really impact the 'Rating' given. 


######################################


# Part 4: 'Review.Text' text characteristics and Ratings  -  correlation


  # Screaming Reviews - Upper Case Letters
proportionUpper = str_count(data$Review.Text,pattern='[A-Z]')/nchar(data$Review.Text)
cor(proportionUpper,data$Rating)
cor.test(proportionUpper,data$Rating)

##  -  Low correlations for all parameters
##  -  Implies that the Upper Case letters in 'Review.Text' do not really impact the 'Ratings'

  # Exclamation Marks
summary(str_count(data$Review.Text,pattern='!')) 
proportionExclamation = str_count(data$Review.Text,pattern='!')/nchar(data$Review.Text)
cor(proportionExclamation,data$Rating)
cor.test(proportionExclamation,data$Rating)

##  -  Cor for: Upper Case = 0.05779606, Exclamation Marks = 0.1776584
##  -  Low correlations for both variables
##  -  Implies that the Exclamation Marks in 'Review.Text' do not greatly impact the 'Ratings'
##  -  But it has more impact than Upper case letter as its correlation is higher than Upper Case letters


######################################


# Part 5: 'Review.Text'  -  most common words


  # Most common words, out of all words
library(qdap) 
freq_terms(text.var = data$Review.Text,top = 10) 
plot(freq_terms(text.var = data$Review.Text,top = 10))

##  -  The most common used words are - the, i, and
##  -  But this is irrelevant. We need to remove stop words before computing this
##  -  See visualization for graph

  # Most common words, excluding stop words
freq_terms(text.var=data$Review.Text,top=10,stopwords = Top200Words)
plot(freq_terms(text.var=data$Review.Text,top=10,stopwords = Top200Words))

##  -  The top used words are - dress, size, love
##  -  See visualization for graph
##  -  (Check Section 3, Part 5 (Line 367) below for wordcloud of common words)
##  -  (Check Section 4, Part 5 (Line 595) below for wordcloud from corpus, which removes stop words, punctuations, sparse terms, etc)


###############################################################################################################
###############################################################################################################



## SECTION 3: Sentiment Analysis on text colum 'Review.Text', formation of Wordclouds


# Conduct Sentiment Analysis using the various Lexicons, and bag of words, and word clouds

# Part 1: Binary Sentiment (positive/negative) - Bing Lexicon


data %>% select(id,Review.Text)%>% group_by(id)%>% unnest_tokens(output=word,input=Review.Text)%>% ungroup()%>% inner_join(get_sentiments('bing'))%>%
  group_by(sentiment)%>% summarize(n = n())%>% mutate(proportion = n/sum(n))

data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(get_sentiments('bing'))%>% group_by(sentiment)%>%
  count()%>% ggplot(aes(x=sentiment,y=n,fill=sentiment))+geom_col()+theme_economist()+guides(fill=F)+ coord_flip()

##  -  Positive words = 90474 and Negative words - 22938
##  -  Approx 80% words are positive in the entire reviews set, which justifies the higher review 'Ratings' as seen before
##  -  See visualization graph

  # Correlation between Positive Words and Review helpfulness
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(get_sentiments('bing'))%>% group_by(id,Rating)%>%
  summarize(positivity = sum(sentiment=='positive')/n())%>% ungroup()%>% summarize(correlation = cor(positivity,Rating))

##  -  The correlation is around 36%, which indicates that a lot of positive words doesnt directly imply a good Rating, but does to a limited extent.


######################################


# Part 2: NRC Sentiment Polarity Table - Lexicon


library(lexicon)
data %>% select(id, Review.Text)%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(y = hash_sentiment_nrc,by = c('word'='x'))%>%
  ungroup()%>% group_by(y)%>% summarize(count = n())%>% ungroup()

##  -  Count of '-1' words = 31221 and '1' words = 63759
##  -  Approx 67% words are in the '1' category


######################################


# Part 3: Emotion Lexicon - NRC Emotion Lexicon


nrc = get_sentiments('nrc')
nrc = read.table(file = 'https://raw.githubusercontent.com/pseudorational/data/master/nrc_lexicon.txt',
                 header = F,
                 col.names = c('word','sentiment','num'),
                 sep = '\t',
                 stringsAsFactors = F)
nrc = nrc[nrc$num!=0,]
nrc$num = NULL

  # Counts of emotions
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(sentiment)%>%count()

  # Plot of emotions
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(sentiment)%>% count()%>%
  ggplot(aes(x=reorder(sentiment,X = n),y=n,fill=sentiment))+geom_col()+guides(fill=F)+coord_flip()+theme_wsj()

##  -  'positive' has the highest count, followed by trust
##  -  See visualization graph

  # Ratings of each Review based on Emotions Expressed
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(id,sentiment,Rating)%>% count()

  # Ratings of all Reviews based on Emotion Expressed
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(id,sentiment,Rating)%>% count()%>%
  group_by(sentiment, Rating)%>% summarize(n = mean(n))%>% data.frame()

data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(id,sentiment,Rating)%>% count()%>%
  group_by(sentiment, Rating)%>% summarize(n = mean(n))%>% ungroup()%>% ggplot(aes(x=Rating,y=n,fill=Rating))+ geom_col()+
  facet_wrap(~sentiment)+ guides(fill=F)+coord_flip()

##  -  See visualization graph, shows distribution of 'Rating' across different emotions 

  # Correlation between emotion expressed and review rating
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(id,sentiment,Rating)%>% count()%>%
  ungroup()%>% group_by(sentiment)%>% summarize(correlation = cor(n,Rating))

  # Scatterplot of relationship
data%>% group_by(id)%>% unnest_tokens(output = word, input = Review.Text)%>% inner_join(nrc)%>% group_by(id,sentiment,Rating)%>% count()%>%
  ungroup()%>% group_by(sentiment)%>% ggplot(aes(x=Rating,y=n))+geom_point()+facet_wrap(~sentiment)+geom_smooth(method='lm',se=F)

##  -  There is a rise in the number of 'joy' and 'positive' words as the 'Rating' goes up.
##  -  And a drop in the number of 'negative' and 'disgust' words as the 'Rating' goes up.
##  -  See visualization graph


######################################


# Part 4: Sentiment score Lexicons - afinn Lexicon


afinn = get_sentiments('afinn')
afinn = read.table('https://raw.githubusercontent.com/pseudorational/data/master/AFINN-111.txt',
                   header = F,
                   quote="",
                   sep = '\t',
                   col.names = c('word','value'), 
                   encoding='UTF-8',
                   stringsAsFactors = F)

data %>% select(id,Review.Text)%>% group_by(id)%>% unnest_tokens(output=word,input=Review.Text)%>% inner_join(afinn)%>%
  summarize(reviewSentiment = mean(value))%>% ungroup()%>%
  summarize(min=min(reviewSentiment),max=max(reviewSentiment),median=median(reviewSentiment),mean=mean(reviewSentiment))

data %>% select(id,Review.Text)%>% group_by(id)%>% unnest_tokens(output=word,input=Review.Text)%>% inner_join(afinn)%>% 
  summarize(reviewSentiment = mean(value))%>% ungroup()%>% ggplot(aes(x=reviewSentiment,fill=reviewSentiment>0))+ geom_histogram(binwidth = 0.1)+
  scale_x_continuous(breaks=seq(-5,5,1))+scale_fill_manual(values=c('tomato','seagreen'))+ guides(fill=F)+ theme_wsj()

##  -  The lowest sentiment score for any 'Review.Text' is -3 and the maximum is 5.
##  -  The mean sentiment score is 1.71 and the median is 1.85
##  -  See visualization graph, shows distribution of sentiment scores and their counts


######################################


# Part 5: Wordcloud of 150 words (except stop words)


library(wordcloud)
wordcloudData = data%>% group_by(id)%>% unnest_tokens(output=word,input=Review.Text)%>% anti_join(stop_words)%>% group_by(word)%>%
  summarize(freq = n())%>% arrange(desc(freq))%>% ungroup()%>% data.frame()

set.seed(123)
wordcloud(words = wordcloudData$word,wordcloudData$freq,scale=c(3,1),max.words = 150,colors=brewer.pal(11,"Spectral"))

##  -  See visualization wordcloud
##  -  (Check Line 592 for wordcloud from corpus, which removes stop words, punctuations, sparse terms, etc)


######################################


# Part 6: Wordcloud of 100 Positive vs Negative words (except stop words)


wordcloudData =  data%>% group_by(id)%>% unnest_tokens(output=word,input=Review.Text)%>% anti_join(stop_words)%>%
  inner_join(get_sentiments('bing'))%>% ungroup()%>% count(sentiment,word,sort=T)%>% spread(key=sentiment,value = n,fill=0)%>% data.frame()
rownames(wordcloudData) = wordcloudData[,'word']
wordcloudData = wordcloudData[,c('positive','negative')]

set.seed(123)
comparison.cloud(term.matrix = wordcloudData,scale = c(2.5,0.8),max.words = 100, rot.per=0)

##  -  See visualization wordcloud, Green = Positive words, Red = Negative words


###############################################################################################################
###############################################################################################################



## SECTION 4: Data Preparation for Predictive Modelling (TF, TF-IDF of text columns 'Review.Text' and 'Title'),
##            and Exploratory Analysis from Corpus for 'Review.Text' 


# Re-run the steps for data preparation - tokenizaton, as was outlined in the previous Project 1.1 file (Line 113 of Project 1.1).

# Part 1: Data Preparation - Tokenization, for both 'Review.Text' and 'Title'


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

##  -  19662 documents with a total of 13633 terms


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

##  -  19662 documents with a total of 3204 terms

####################

# Remove Sparse Terms  -  We will remove those words which appear in less than 3% of the reviews

  # for Review.Text
xdtm = removeSparseTerms(dtm,sparse = 0.97)
xdtm
xdtm_cluster = xdtm # to be used later for clustering

  # for Title
xdtm2 = removeSparseTerms(dtm2,sparse = 0.97)
xdtm2; xdtm2_cluster = xdtm2

####################

# Complete Stems and Sort Tokens

  # for Review.Text
xdtm = as.data.frame(as.matrix(xdtm))
colnames(xdtm) = stemCompletion(x = colnames(xdtm),
                                dictionary = dict_corpus,
                                type='prevalent')
colnames(xdtm) = make.names(colnames(xdtm))
sort(colSums(xdtm),decreasing = T)  

##  -  sort to see most common terms

  # for Title
xdtm2 = as.data.frame(as.matrix(xdtm2))
colnames(xdtm2) = stemCompletion(x = colnames(xdtm2),
                                 dictionary = dict_corpus,
                                 type='prevalent')
colnames(xdtm2) = make.names(colnames(xdtm2))
sort(colSums(xdtm2),decreasing = T)  

##  -  sort to see most common terms


######################################


# Part 2: Document Term Matrix using Inverse Document Frequency - tfidf


  # for Review.Text
dtm_tfidf = DocumentTermMatrix(x=corpus,
                               control = list(weighting=function(x) weightTfIdf(x,normalize=F)))
xdtm_tfidf = removeSparseTerms(dtm_tfidf,sparse = 0.97)
xdtm_tfidf = as.data.frame(as.matrix(xdtm_tfidf))
colnames(xdtm_tfidf) = stemCompletion(x = colnames(xdtm_tfidf),
                                      dictionary = dict_corpus,
                                      type='prevalent')
colnames(xdtm_tfidf) = make.names(colnames(xdtm_tfidf))
sort(colSums(xdtm_tfidf),decreasing = T)

##  -  sort to see most common terms

  # for Title
dtm_tfidf2 = DocumentTermMatrix(x=corpus2,
                               control = list(weighting=function(x) weightTfIdf(x,normalize=F)))
xdtm_tfidf2 = removeSparseTerms(dtm_tfidf2,sparse = 0.97)
xdtm_tfidf2 = as.data.frame(as.matrix(xdtm_tfidf2))
colnames(xdtm_tfidf2) = stemCompletion(x = colnames(xdtm_tfidf2),
                                      dictionary = dict_corpus2,
                                      type='prevalent')
colnames(xdtm_tfidf2) = make.names(colnames(xdtm_tfidf2))
sort(colSums(xdtm_tfidf2),decreasing = T)

##  -  sort to see most common terms


######################################


# Part 3: Compare both DTM methods' results using graph

  # for Review.Text
data.frame(term = colnames(xdtm),tf = colMeans(xdtm), tfidf = colMeans(xdtm_tfidf))%>%
  arrange(desc(tf))%>%
  top_n(9)%>%
  gather(key=weighting_method,value=weight,2:3)%>%
  ggplot(aes(x=term,y=weight,fill=weighting_method))+
  geom_col(position='dodge')+
  coord_flip()+
  theme_economist()

##  -  the term dress was assigned a much higher weight in the tf method, becasue it occured in most of the reviews
##  -  but was assigned a lower weight in the tditf method, becasue it has little diagnostic value, since it occurs on most reviews.
##  -  See visualization graph


# for Title
data.frame(term = colnames(xdtm2),tf = colMeans(xdtm2), tfidf = colMeans(xdtm_tfidf2))%>%
  arrange(desc(tf))%>%
  top_n(10)%>%
  gather(key=weighting_method,value=weight,2:3)%>%
  ggplot(aes(x=term,y=weight,fill=weighting_method))+
  geom_col(position='dodge')+
  coord_flip()+
  theme_economist()

##  -  the term love and great were assigned a much higher weight in the tf method, becasue they occured in most of the titles
##  -  but was assigned a lower weight in the tditf method, becasue they have little diadnostic value, since they occur on most titles
##  -  See visualization graph


######################################


# Part 4: Add Rating back to dataframe of features


  # for Review.Text
clothes_data = cbind(Rating = data$Rating, xdtm)
clothes_data_tfidf = cbind(Rating = data$Rating, xdtm_tfidf)

  # for Title
clothes_data2 = cbind(Rating = data$Rating,xdtm2)
clothes_data_tfidf2 = cbind(Rating = data$Rating,xdtm_tfidf2)


######################################


# Part 5: WordCloud from the prepared corpus set (removing all stop words, punctuations, etc)


  # for Review.Text
set.seed(123)
wordcloud(corpus, scale=c(6,0.5), max.words=170, random.order=FALSE, rot.per=0.35, use.r.layout=FALSE, colors=brewer.pal(8, 'Dark2'))

##  -  See visualization wordcloud



###############################################################################################################
###############################################################################################################



## SECTION 5: Predictive Modelling (CART and Regression) using only text columns 'Review.Text' and 'Title'


# Part 1: Predictive Models (using TF)


  # for Review.Text
set.seed(617)
split = sample(1:nrow(clothes_data), size = 0.75*nrow(clothes_data))
train = clothes_data[split,]
test = clothes_data[-split,]

  # CART Method
library(rpart); library(rpart.plot)
tree = rpart(Rating~.,train)
rpart.plot(tree)
pred_tree = predict(tree,newdata=test)
rmse_tree = round(sqrt(mean((pred_tree - test$Rating)^2)),5); rmse_tree

##  -  See visualization Tree
##  -  RMSE = 1.009915

  # Regression Method
reg = lm(Rating~.,train)
pred_reg = predict(reg, newdata=test)
rmse_reg = round(sqrt(mean((pred_reg-test$Rating)^2)),5); rmse_reg

##  -  RMSE = 0.9013822


  # for Title
set.seed(617)
split = sample(1:nrow(clothes_data2), size = 0.75*nrow(clothes_data2))
train2 = clothes_data2[split,]
test2 = clothes_data2[-split,]

  # CART Method
tree2 = rpart(Rating~.,train2)
rpart.plot(tree2)
pred_tree2 = predict(tree2,newdata=test2)
rmse_tree2 = round(sqrt(mean((pred_tree2 - test2$Rating)^2)),5); rmse_tree2

##  -  See visualization Tree
##  -  RMSE = 1.075686

  # Regression Method
reg2 = lm(Rating~.,train2)
pred_reg2 = predict(reg2, newdata=test2)
rmse_reg2 = round(sqrt(mean((pred_reg2-test2$Rating)^2)),5); rmse_reg2

##  -  RMSE = 1.06697

##  -  Title is also not a bad predictor as well, the rmse lies within close range of Review.Text. But Review.Text gives the lowest rmse.


######################################


# Part 2: Predictive Models (using TF-IDF)


  # for Review.Text
set.seed(617)
split = sample(1:nrow(clothes_data_tfidf), size = 0.75*nrow(clothes_data_tfidf))
train = clothes_data_tfidf[split,]
test = clothes_data_tfidf[-split,]

  # CART Method
tree = rpart(Rating~.,train)
rpart.plot(tree)
pred_tree = predict(tree,newdata=test)
rmse_tree_idf = round(sqrt(mean((pred_tree - test$Rating)^2)),5); rmse_tree_idf

##  -  RMSE = 1.009915
##  -  See visualization Tree

  # Regression Method
reg = lm(Rating~.,train)
pred_reg = predict(reg, newdata=test)
rmse_reg_idf = round(sqrt(mean((pred_reg-test$Rating)^2)),5); rmse_reg_idf
##  -  RMSE = 0.9013822


  # for Title
set.seed(617)
split = sample(1:nrow(clothes_data_tfidf2), size = 0.75*nrow(clothes_data_tfidf2))
train2 = clothes_data_tfidf2[split,]
test2 = clothes_data_tfidf2[-split,]

  # CART Method
tree2 = rpart(Rating~.,train2)
rpart.plot(tree2)
pred_tree2 = predict(tree2,newdata=test2)
rmse_tree2_idf = round(sqrt(mean((pred_tree2 - test2$Rating)^2)),5); rmse_tree2_idf

##  -  RMSE = 1.075686
##  -  See visualization Tree

  # Regression Method
reg2 = lm(Rating~.,train2)
pred_reg2 = predict(reg2, newdata=test2)
rmse_reg2_idf = round(sqrt(mean((pred_reg2-test2$Rating)^2)),5); rmse_reg2_idf

##  -  RMSE = 1.06697

rmse_review_text_df = data.frame(for_Review.Text = c("Method", "TF", "TF-IDF"),CART_RMSE  = c(" ", rmse_tree, rmse_tree_idf),
                                  Regression_RMSE = c(" ", rmse_reg, rmse_reg_idf))
rmse_title_df = data.frame(for_Title = c("Method", "TF", "TF-IDF"),CART_RMSE  = c(" ", rmse_tree2, rmse_tree2_idf),
                                  Regression_RMSE = c(" ", rmse_reg2, rmse_reg2_idf))
rmse_review_text_df
rmse_title_df

##  -  Both methods, i.e., TF and TF-IDf give the exact same RMSE for both 'Review.Text' and 'Title'.
##  - 'Review.Text' always gives lower rmse than any method used for 'Title'. So we shoud use 'Review.Text' going forward.
##  -  For best rmse, we need to use the regression method of predictive modelling, but wmight need to compare results from TF and TF-IDF methods.



###############################################################################################################
###############################################################################################################



# SECTION 6: Clustering and Predictive Modelling using clustering techniques, except all text columns, dendogram for text columns clustering


# Part 1: Prepare Data for Cluster Analysis


library(caret)
set.seed(617)
split = createDataPartition(y=data$Rating,p = 0.75,list = F,groups = 100)
train = data[split,]
test = data[-split,]

train = subset(train, select = -c(id, Clothing.ID, Title, Review.Text, Division.Name, Department.Name, Class.Name, bins))
test = subset(test, select = -c(id, Clothing.ID, Title, Review.Text, Division.Name, Department.Name, Class.Name, bins))

  # Simple Regression
linear = lm(Rating~.,train)
summary(linear)
sseLinear = sum(linear$residuals^2); sseLinear
predLinear = predict(linear,newdata=test)
sseLinear = sum((predLinear-test$Rating)^2); sseLinear


  # Cluster and Regression
trainMinusDV = subset(train,select=-c(Rating))
testMinusDV = subset(test,select=-c(Rating))
  
  # Prepare Data for Clustering - Cluster Analysis is sensitive to scale. Normalizing the data.
preproc = preProcess(trainMinusDV)
trainNorm = predict(preproc,trainMinusDV)
testNorm = predict(preproc,testMinusDV)

  
######################################


# Part 2: Hierarchical and k-means Cluster Analysis


  # Hierarchical
distances = dist(trainNorm,method = 'euclidean')
clusters = hclust(d = distances,method = 'ward.D2')
library(dendextend)
plot(color_branches(cut(as.dendrogram(clusters), h = 20)$upper), k = 3, groupLabels = F) # displaying clusters with tree above 20 
rect.hclust(tree=clusters,k = 3,border='red')

##  -  Based on the plot, a 3 cluster solution looks good.

clusterGroups = cutree(clusters,k=2)
# install.packages('psych')

  # visualize
library(psych)
temp = data.frame(cluster = factor(clusterGroups),
                  factor1 = fa(trainNorm,nfactors = 2,rotate = 'varimax')$scores[,1],
                  factor2 = fa(trainNorm,nfactors = 2,rotate = 'varimax')$scores[,2])
ggplot(temp,aes(x=factor1,y=factor2,col=cluster))+
  geom_point()

##  -  See visualization graph


  # k-means clustering
set.seed(617)
km = kmeans(x = trainNorm,centers = 2,iter.max=10000,nstart=100)
km$centers
mean(km$cluster==clusterGroups) # %match between results of hclust and kmeans

  # Total within sum of squares Plot
within_ss = sapply(1:10,FUN = function(x) kmeans(x = trainNorm,centers = x,iter.max = 1000,nstart = 25)$tot.withinss)
ggplot(data=data.frame(cluster = 1:10,within_ss),aes(x=cluster,y=within_ss))+ geom_line(col='steelblue',size=1.2)+ 
  geom_point()+ scale_x_continuous(breaks=seq(1,10,1))

  # Ratio Plot
ratio_ss = sapply(1:10,FUN = function(x) {km = kmeans(x = trainNorm,centers = x,iter.max = 1000,nstart = 25)
km$betweenss/km$totss} )
ggplot(data=data.frame(cluster = 1:10,ratio_ss),aes(x=cluster,y=ratio_ss))+ geom_line(col='steelblue',size=1.2)+
  geom_point()+ scale_x_continuous(breaks=seq(1,10,1))

  # Silhouette Plot
library(cluster)
silhoette_width = sapply(2:10,FUN = function(x) pam(x = trainNorm,k = x)$silinfo$avg.width)
#ggplot(data=data.frame(cluster = 2:10,silhoette_width),aes(x=cluster,y=silhoette_width))+     # takes too much time
#  geom_line(col='steelblue',size=1.2)+ geom_point()+ scale_x_continuous(breaks=seq(2,10,1))


######################################


# Part 3: Apply to test, and Compare Results


  # Set the centers as 3
set.seed(617)
km = kmeans(x = trainNorm,centers = 3,iter.max=10000,nstart=100)

# install.packages('flexclust')
library(flexclust)
km_kcca = as.kcca(km,trainNorm) # flexclust uses objects of the classes kcca
clusterTrain = predict(km_kcca)
clusterTest = predict(km_kcca,newdata=testNorm)

table(clusterTrain)
table(clusterTest)

  # Split train and test based on cluster membership
train1 = subset(train,clusterTrain==1)
train2 = subset(train,clusterTrain==2)
test1 = subset(test,clusterTest==1)
test2 = subset(test,clusterTest==2)

  # Predict for each Cluster then Combine
lm1 = lm(Rating~.,train1)
lm2 = lm(Rating~.,train2)
pred1 = predict(lm1,newdata=test1)
pred2 = predict(lm2,newdata=test2)
sse1 = sum((test1$Rating-pred1)^2); sse1
sse2 = sum((test2$Rating-pred2)^2); sse2

predOverall = c(pred1,pred2)
RatingOverall = c(test1$Rating,test2$Rating)
sseOverall = sum((predOverall - RatingOverall)^2); sseOverall

  # Compare Results
paste('SSE for model on entire data',sseLinear)
paste('SSE for model on clusters',sseOverall)

##  -  SSE on Entire data = 2262.3200502617, SSE on Clusters = 1643.99972478085
##  -  Prediction using clusters is more accurate, as the standard error is less.


######################################


# Part 4: Predict Using Tree, and Compare Results


  # Simple Tree
library(rpart); library(rpart.plot)
tree = rpart(Rating~.,train,minbucket=10)
predTree = predict(tree,newdata=test)
sseTree = sum((predTree - test$Rating)^2); sseTree


  # Cluster Then Predict Using Tree
tree1 = rpart(Rating~.,train1,minbucket=10)
tree2 = rpart(Rating~.,train2,minbucket=10)
pred1 = predict(tree1,newdata=test1)
pred2 = predict(tree2,newdata=test2)

sse1 = sum((test1$Rating-pred1)^2); sse1
sse2 = sum((test2$Rating-pred2)^2); sse2

predTreeCombine = c(pred1,pred2)
RatingOverall = c(test1$Rating,test2$Rating)
sseTreeCombine = sum((predTreeCombine - RatingOverall)^2); sseTreeCombine


  # Compare Results
paste('SSE for model on entire data',sseTree)
paste('SSE for model on clusters',sseTreeCombine)

##  -  SSE on Entire data = 2262.07769316003, SSE on Clusters = 1643.2592670892
##  -  Prediction using clusters is more accurate, as the standard error is less.
##  -  Lowest Error is when we CLuster with Tree and predict


######################################


# Part 5: Clustering, and Dendogram from cleaned corpus, of 'Review.Text' and 'Title'


# We had defined 'xdtm_cluster' as the cleaned corpus earlier in Line 478

  # 'Review.Text'
#hc = hclust(d = dist(xdtm_cluster, method = "euclidean"), method = "complete")  # this takes massive time to run
#plot(hc)

  # 'Title'
hc = hclust(d = dist(xdtm2_cluster, method = "euclidean"), method = "complete")
plot(hc)

##  -  See visualization graph



###############################################################################################################
###############################################################################################################



# SECTION 7: Looking at Future, what else we could have done.


#  1: In-dept Cluster Analysis of text columns, using detailed scatterplots


# For clustering and prediction Modelling using the text column 'Review.Text', the following code can be used.

# Source 1 - https://gist.github.com/luccitan/b74c53adfe3b6dad1764af1cdc1f08b7
# Source 2 - https://medium.com/@SAPCAI/text-clustering-with-r-an-introduction-for-data-scientists-c406e7454e76

# We had defined 'xdtm_cluster' as the cleaned corpus earlier in Line 478, which will be used here for converting to matrix, etc... as per the code given.


######################################


#  2: Further detailed exploratory analysis


# Source - https://www.kaggle.com/dubravkodolic/reviews-of-clothings-analyzed-by-sentiments
# Source - https://www.kaggle.com/cosinektheta/mining-the-women-s-clothing-reviews

######################################


#  3: More prediction models, to evaluate better rmse measures


# Source - https://www.kaggle.com/ankitppn/logistic-regression-and-random-forest-models/output



####################################   T H E   E N D   ####################################

