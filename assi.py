import pandas as pd
import numpy as ny 
#we use these libraries


#load dataset and check if it works or not 
data= pd.read_csv("Dataset.csv")
print(data.head()) #first five rows of dataset
print (data.tail()) #last five rows of dataset
print (data.describe()) #descibe the dataset(types)
print ("dim",data. ndim) #dimintion for the data
print   ("shape",data.shape) #[row,column]
print ("--------------------------------------------------------------------------------------------------------")

# #&&&&&&&&&&&&THE FIRST PART WAS DONE &&&&&&&&&&&&&

#check if we have null value or not 
print(data.isnull().sum().sum())  
#create a Nan value cause i dont have a null value on my dataset
data111 =data.replace(0.0 ,ny.nan)
print (data111.isnull().sum().sum())  

print  (data111.fillna(method='pad'))

print (data111.fillna(method='pad').isnull().sum().sum())
  
print ( data111.fillna(method='bfill'))
print (data111.fillna(method='bfill').isnull().sum().sum())

print  (data111.fillna(method='pad',axis=1))


print  (data111.fillna({'Happiness Rank':'abcd'}))

 #after created a null value on my dataset now i handle the missing value
 
print(data111.dropna())
 
 
print(data111.dropna(how='any'))
 
print (data111.replace(to_replace=ny.nan,value=5.0))

print (data111.replace(to_replace=1.0,value=5.0))
print ("--------------------------------------------------------------------------------------------------------")

#&&&&&&&&&&&&&THE SECOND PART IS DONE &&&&&&&&&&&&& 
#we want to remove the noise by matplotlib 
labels=['low','medium','sign']
import matplotlib.pyplot as plt
bins=[20,25,30,60]
data['bin_cut_manual']=pd.cut(data['Freedom'], bins=bins ,lables=labels ,include_lowest=True)

plt.hist(data['bins_cut_manual'] ,bins=s)
plt.show()





print ("--------------------------------------------------------------------------------------------------------")

#&&&&&&&&&&&&&THE THIRD PART IS DONE &&&&&&&&&&&&&


#create dataframe
df_marks = pd.DataFrame(data)
 

print('Original DataFrame\n------------------')
print(df_marks)

new_row = {'Country':'Togo', 'Region':'Sub-Saharan Africa', 'Happiness Rank':158, 'Happiness Score':2.839 ,'Standard Error':0.06727 ,
'Economy (GDP per Capita)': 0.20868 ,'Family': 0.13995 ,'Health (Life Expectancy)':0.28443 ,'Freedom': 0.36453   ,'Trust (Government Corruption)':0.10731
,'Generosity':0.16681 , 'Dystopia Residual': 1.56726 }

#append row to the dataframe
df_marks = df_marks.append(new_row, ignore_index=True)

print('\n\nNew row added to DataFrame\n--------------------------')
print(df_marks)




 
# dropping ALL duplicate rows
print(df_marks.duplicated('Country'))
 
print(df_marks.shape)
df_marks_dup=df_marks.drop_duplicates('Country')

print(df_marks_dup.shape)

print ("--------------------------------------------------------------------------------------------------------")
#&&&&&&&&&&&&&THE THIRD FOURTH IS DONE &&&&&&&&&&&&&
#we want to change the type of inconsistent column 
print(data.dtypes)
print ( data.Freedom.dtypes)
print(data.Freedom.astype(int))


#&&&&&&&&&&&&&THE FIFTH PART IS DONE &&&&&&&&&&&&&



#Creating the Correlation matrix and Selecting the Upper trigular matrix
cor_matrix = data.corr().abs()
print(cor_matrix)

# Note that Correlation matrix will be mirror image about the diagonal and all the diagonal elements will be 1 ,we are selecting the upper traingular
upper_tri = cor_matrix.where(ny.triu(ny.ones(cor_matrix.shape),k=1).astype(ny.bool))
print(upper_tri)


#Droping the column with high correlation

to_drop= [column for column in upper_tri.columns if any(upper_tri[column] >= - 0.8)]
print(); print(to_drop)

#Now we are droping the columns which are in the list 'to_drop' from the dataframe
df1 = data.drop( to_drop ,axis=1)
print(); print(df1.head())


 #Droping the column with high correlation

to_drop= [column for column in upper_tri.columns if any(upper_tri[column] >= 0.8)]
print(); print(to_drop)

#Now we are droping the columns which are in the list 'to_drop' from the dataframe
df1 = data.drop( to_drop ,axis=1)
print(); print(df1.head())

#&&&&&&&&&&&&&THE SIXTH PART IS DONE &&&&&&&&&&&&&

import seaborn as sns

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")

sns.set_style('whitegrid')

df = data.groupby(['Freedom', 'Happiness Score'])['Happiness Rank'].sum().reset_index()

print (df['Happiness Rank'].plot(kind='hist'))

print (df['Happiness Rank'].describe()) 

pd.qcut([df['Happiness Rank']], q=4)

df['Generosity'] = pd.qcut(df['Happiness Rank'], q=4)
df['Freedom'] = pd.qcut(df['Happiness Rank'], q=10, precision=0)

print (df.head())

print(df['Generosity'].value_counts())

print(df['Freedom'].value_counts())

bin_labels_5 = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
df['quantile_ex_3'] = pd.qcut(df['ext price'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=bin_labels_5)
print (df.head())

df['quantile_ex_3'].value_counts()

results, bin_edges = pd.qcut(df['Happiness Rank'],
                            q=[0, .2, .4, .6, .8, 1],
                            labels=bin_labels_5,
                            retbins=True)

results_table = pd.DataFrame(zip(bin_edges, bin_labels_5),
                            columns=['Threshold', 'Tier'])

print(df.describe(include='category'))

print(df.describe(percentiles=[0, 1/3, 2/3, 1]))


df['quantile_ex_4'] = pd.qcut(df['Happiness Rank'],
                            q=[0, .2, .4, .6, .8, 1],
                            labels=False,
                            precision=0)
print (df.head())

print ( df.drop(columns = ['Generosity','Freedom', 'quantile_ex_3', 'quantile_ex_4']))

print (pd.cut(df['Happiness Rank'], bins=4))

print(pd.cut(df['ext price'], bins=4).value_counts())

cut_labels_4 = ['silver', 'gold', 'platinum', 'diamond']
cut_bins = [0, 70000, 100000, 130000, 200000]
df['cut_ex1'] = pd.cut(df['Happiness Rank'], bins=cut_bins, labels=cut_labels_4)

print (pd.cut(df['Happiness Rank'], bins=ny.linspace(0, 200000, 9)))
print(pd.interval_range(start=0, freq=10000, end=200000, closed='left'))

interval_range = pd.interval_range(start=0, freq=10000, end=200000)
df['cut_ex2'] = pd.cut(df['Happiness Rank'], bins=interval_range, labels=[1,2,3])
print (df.head())
print (df['Happiness Rank'].value_counts(bins=4, sort=False))

#&&&&&&&&&&&&&THE SEVENTH PART IS DONE &&&&&&&&&&&&&
import sklearn
import matplotlib.pyplot as plt

plt.scatter(data['Happiness Score'],data['Happiness Rank'])

plt.scatter(data['Family'],data['Happiness Rank'])
#Looking at above two scatter plots, using linear regression model makes sense as we can clearly see a linear relationship 
#The approach we are going to use here is to split available data in two sets

#The reason we don't use same training set for testing is because our model has seen those samples before, using same samples for making predictions might give us wrong impression about accuracy of our model. It is like you ask same questions in exam paper as you tought the students in the class.

X = df[['Mileage','Age(yrs)']]
X = data[['Happiness Score','Happiness Rank']]

y = data['Happiness Rank']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3) 
print(X_train)


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)


print(X_test)
clf.predict(X_test)
print (y_test)
clf.score(X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
print (X_test)