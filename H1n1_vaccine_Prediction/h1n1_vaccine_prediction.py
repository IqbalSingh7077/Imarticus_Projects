
import os
os.chdir("E:\\Imarticus\\Python\\project2-Logistic Regression")

# importing important libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing the dataset

df = pd.read_csv("h1n1_vaccine_prediction.csv") 


#descriptive stats
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.info()
df.dtypes
df.describe(include='all')
df.isnull().sum()# there are many missing values in this dataset


#*******EDA*******#

#1. h1n1_worry vs h1n1_vaccine

df.h1n1_worry.value_counts()
sum(df.h1n1_worry.isnull())#92 missing values
sns.countplot(df.h1n1_worry)
sns.countplot(df.h1n1_worry, hue=df.h1n1_vaccine)

'''
there are 92 missing values, which we will replace by the mean

we can club the categories into to categories, one being of least worried and another
being of most worried
'''

df['h1n1_worry'].fillna(df['h1n1_worry'].mode()[0], inplace = True)
sum(df.h1n1_worry.isnull())# no missing value


df['h1n1_worry']  = df.get('h1n1_worry').replace(0,"least_worried")
df['h1n1_worry']  = df.get('h1n1_worry').replace(1,"least_worried")
df['h1n1_worry']  = df.get('h1n1_worry').replace(2,"most_worried")
df['h1n1_worry']  = df.get('h1n1_worry').replace(3,"most_worried")

df.h1n1_worry.value_counts()
sum(df.h1n1_worry.isnull())#No missing values
sns.countplot(df.h1n1_worry)
sns.countplot(df.h1n1_worry, hue=df.h1n1_vaccine)

#performing chi-square test
from scipy.stats import chi2_contingency 
contigency= pd.crosstab(df['h1n1_worry'], df['h1n1_vaccine'])
contigency

c, p_value, dof, expected = chi2_contingency(contigency)
p_value
'''
p-value = 1.1959246410047537e-70, important variable
'''

#EX_1= df[df.h1n1_vaccine== 1]
#EX_0= df[df.h1n1_vaccine== 0]

#len(EX_1)
#len(EX_0)


#statistics, p=scipy.stats.ttest_ind(EX_1.h1n1_worry, EX_0.h1n1_worry)
#p






## 2. h1n1_awareness vs h1n1_vaccine

df.h1n1_awareness.value_counts()
sum(df.h1n1_awareness.isnull())# 116 missing values
sns.countplot(df.h1n1_awareness)
sns.countplot(df.h1n1_awareness, hue=df.h1n1_vaccine)
'''
few observations
1 - proportion of people taken vaccine, with little knowledge and good knowledge about
the flu is same
2 - despite having knowledge about the flu, people are resisting to take the vaccine

I will replace the missing values by median
I have decided not to club the categories
''' 

df['h1n1_awareness'].fillna(df['h1n1_awareness'].median(), inplace=True)
sum(df.h1n1_awareness.isnull())#no missing values
df.h1n1_awareness.value_counts()
'''
1.0    14714
2.0     9487
0.0     2506
Name: h1n1_awareness, dtype: int64
'''
# t-test
A_1= df[df.h1n1_vaccine== 1]
A_0= df[df.h1n1_vaccine== 0]
import scipy
scipy.stats.ttest_ind(A_1.h1n1_awareness, A_0.h1n1_awareness)
'''
p-value = 4.188858352900413e-83, could be an important predictor
'''



## 3. antiviral_medication vs h1n1_vaccine

df.antiviral_medication.value_counts()
sum(df.antiviral_medication.isnull())# 71 missing values
sns.countplot(df.antiviral_medication)
sns.countplot(df.antiviral_medication, hue = df.h1n1_vaccine)

'''
0.0    25335
1.0     1301
Name: antiviral_medication, dtype: int64

the data is more biased towards 'no', hence it would not be a great idea to include
this variable in model making
'''



## 4. contact_avoidance vs h1n1_vaccine

df.contact_avoidance.value_counts()
sum(df.contact_avoidance.isnull())# 208 missing value
sns.countplot(df.contact_avoidance)
sns.countplot(df.contact_avoidance, hue = df.h1n1_vaccine)

'''
i will fill the missing values by mode of this column

observations
1- people those who avoided contact with people having h1n1 flu like symptoms
took vaccine more, from this we can conclude that people those who take maximum
precautions tend to opt for vaccine more

'''

df['contact_avoidance'].fillna(df['contact_avoidance'].mode()[0], inplace = True)
sum(df.contact_avoidance.isnull())# no missing values
df.contact_avoidance.value_counts()
'''
1.0    19436
0.0     7271
Name: contact_avoidance, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['contact_avoidance'], df['h1n1_vaccine'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p
'''
2.2161129058289884e-14, could be an important predictor
'''


## 5. bought_face_mask vs h1n1_vaccine

df.bought_face_mask.value_counts()
sum(df.bought_face_mask.isnull())# 19 missing values
sns.countplot(df.bought_face_mask)
sns.countplot(df.bought_face_mask, hue = df.h1n1_vaccine)

'''
0.0    24847
1.0     1841
Name: bought_face_mask, dtype: int64

the data is more biased towards 'no' with 92%, hence it would not be a g
great idea to include this variable in model making
'''



## 6. wash_hands_frequently vs h1n1_vaccine

df.wash_hands_frequently.value_counts()
sum(df.wash_hands_frequently.isnull())# 42 missing values
sns.countplot(df.wash_hands_frequently)
sns.countplot(df.wash_hands_frequently, hue = df.h1n1_vaccine)
'''
i will replace missing values with the mode
the data is more biased towards 1.0 so not using this variable
'''



## 7. avoid_large_gatherings vs h1n1_vaccine

df.avoid_large_gatherings.value_counts()
sum(df.avoid_large_gatherings.isnull())# 87 missing values
sns.countplot(df.avoid_large_gatherings)
sns.countplot(df.avoid_large_gatherings, hue = df.h1n1_vaccine)

'''
i will replace the missing values by mode of this column

observation
1- people those who did not avoid the large gatherings took more vaccine shots, from
this we can cocnlude that people wanted to return to thier normal lives as soon as 
possible.
'''
df['avoid_large_gatherings'].fillna(df['avoid_large_gatherings'].mode()[0],inplace = True)
sum(df.avoid_large_gatherings.isnull())# no missing values
df.avoid_large_gatherings.value_counts()
'''
0.0    17160
1.0     9547
Name: avoid_large_gatherings, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['avoid_large_gatherings'], df['h1n1_vaccine'])
contigency

c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value = 0.003276654049065037, can be an important variable
'''



# 8. reduced_outside_home_cont vs h1n1_vaccine

df.reduced_outside_home_cont.value_counts()
sum(df.reduced_outside_home_cont.isnull())# 82 missing values
sns.countplot(df.reduced_outside_home_cont)
sns.countplot(df.reduced_outside_home_cont, hue = df.h1n1_vaccine)
'''
i will replace the missing values with the mode category

observations-
1- most of the people did not reduce the contact with people
outside their own house.
2 - people those who did not reduce the contact took more vaccine shots
'''

df['reduced_outside_home_cont'].fillna(df['reduced_outside_home_cont'].mode()[0], inplace = True)
sum(df.reduced_outside_home_cont.isnull())#no missing values
df.reduced_outside_home_cont.value_counts()
'''
0.0    17726
1.0     8981
Name: reduced_outside_home_cont, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['reduced_outside_home_cont'], df['h1n1_vaccine'])
contigency

c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value = 0.00032755872863934096, can be a good predictor
'''


# 9. avoid_touch_face vs h1n1_vaccine

df.avoid_touch_face.value_counts()
sum(df.avoid_touch_face.isnull())#128 missing values
sns.countplot(df.avoid_touch_face)
sns.countplot(df.avoid_touch_face, hue = df.h1n1_vaccine)

'''
i will replace the missing values with mode

observations - 
1- more people took precaution of not touching thier face
2- those who took precaution, also took more vaccine doses
'''

df['avoid_touch_face'].fillna(df['avoid_touch_face'].mode()[0],inplace=True)
sum(df.avoid_touch_face.isnull())#no missing values
df.avoid_touch_face.value_counts()
'''
1.0    18129
0.0     8578
Name: avoid_touch_face, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['avoid_touch_face'], df['h1n1_vaccine'])
contigency

c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value = 6.3218424278463514e-31, can be a good predictor
'''



## 10. dr_recc_h1n1_vacc vs h1n1_vaccine

df.dr_recc_h1n1_vacc.value_counts()
sum(df.dr_recc_h1n1_vacc.isnull())# 2160 missing values
sns.countplot(df.dr_recc_h1n1_vacc)
sns.countplot(df.dr_recc_h1n1_vacc, hue = df.h1n1_vaccine)

'''
I will fill the missig values with the mode of this category

it can be seen that doctors recommendation did not affect much, as the proportion
of people taking vaccine with or without recommendation is approximately the same

we can drop this variable as the data is more dominant towards 'NO' with around 80%
after filling in the N.A values
'''


## 11. dr_recc_seasonal_vacc vs h1n1_vaccine

df.dr_recc_seasonal_vacc.value_counts()
sum(df.dr_recc_seasonal_vacc.isnull())## 2160 missing values 
sns.countplot(df.dr_recc_seasonal_vacc)
sns.countplot(df.dr_recc_seasonal_vacc, hue = df.h1n1_vaccine)
'''
I will treat the missing values with mode of the category

'''

df['dr_recc_seasonal_vacc'].fillna(df['dr_recc_seasonal_vacc'].mode()[0], inplace = True)
sum(df.dr_recc_seasonal_vacc.isnull())## no missing values
df.dr_recc_seasonal_vacc.value_counts()
'''
0.0    18613
1.0     8094
Name: dr_recc_seasonal_vacc, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['dr_recc_seasonal_vacc'], df['h1n1_vaccine'])
contigency

c,p,dof,expected = chi2_contingency(contigency)
p

'''
p-value = 3.3126165190104603e-280, can be an important predictor
'''


## 12. chronic_medic_condition vs h1n1_vaccine

df.chronic_medic_condition.value_counts()
sum(df.chronic_medic_condition.isnull())#971 missing values
sns.countplot(df.chronic_medic_condition)
sns.countplot(df.chronic_medic_condition, hue = df.h1n1_vaccine)

'''
I will fill the missing values with mode category

observation -
* people with no chronic condition took vaccine more than people with chronic condition
'''

df['chronic_medic_condition'].fillna(df['chronic_medic_condition'].mode()[0], inplace = True)
sum(df.chronic_medic_condition.isnull())# no missing values
df.chronic_medic_condition.value_counts()
'''
0.0    19417
1.0     7290
Name: chronic_medic_condition, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['chronic_medic_condition'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine                 0     1
chronic_medic_condition             
0.0                      15751  3666
1.0                       5282  2008
'''

c,p,dof,expected= chi2_contingency(contigency)
p
'''
1.5428233060113362e-53, can be an important variable
'''

## 13. cont_child_undr_6_mnths vs h1n1_vaccine

df.cont_child_undr_6_mnths.value_counts()
sum(df.cont_child_undr_6_mnths.isnull())# 820 missing values
sns.countplot(df.cont_child_undr_6_mnths)
sns.countplot(df.cont_child_undr_6_mnths, hue = df.h1n1_vaccine)

'''
I will drop this variable as the data is more biased towards "yes" category
by 92% before treating missing values

'''


## 14. is_health_worker vs h1n1_vaccine

df.is_health_worker.value_counts()
df.is_health_worker.isnull().sum()## 804 missing values
sns.countplot(df.is_health_worker)
sns.countplot(df.is_health_worker, hue=df.h1n1_vaccine)

'''
I will drop this variable as the data is more baised to 'yes' category by 
86% without treating missing values
'''



## 15. has_health_insur vs h1n1_vaccine

df.has_health_insur.value_counts()
sum(df.has_health_insur.isnull())## 12274 missing values
sns.countplot(df.has_health_insur)


'''
dropping the varibable, as it has 45% missing values, and replacing it with 
the mode category wil make it more biased towards category 1.0 i.e. 'yes'
'''



## 16. is_h1n1_vacc_effective vs h1n1_vaccine

df.is_h1n1_vacc_effective.value_counts()
sum(df.is_h1n1_vacc_effective.isnull())## 391 missing values
sns.countplot(df.is_h1n1_vacc_effective)
sns.countplot(df.is_h1n1_vacc_effective, hue = df.h1n1_vaccine)

'''
I will replace all the missing values with the median as the data is skewed.

observations:
       
1- more people took vaccine, who thought vaccine is more effective.
2- nearly 18% of the population doesnt have the knowledge about effectiveness of
the vaccine


I will club the categories as follows-

thinks it is not effective - 1 & 2
dont know - 3
thinks it is effective - 4 & 5
'''

df['is_h1n1_vacc_effective'].fillna(df['is_h1n1_vacc_effective'].median(), inplace = True)
sum(df.is_h1n1_vacc_effective.isnull())# no missing values

df.is_h1n1_vacc_effective.value_counts()
sns.countplot(df.is_h1n1_vacc_effective, hue=df.h1n1_vaccine)

'''

4.0    12074
5.0     7166
3.0     4723
2.0     1858
1.0      886
Name: is_h1n1_vacc_effective, dtype: int64
'''

# t-test

A_1= df[df.h1n1_vaccine== 1]
A_0= df[df.h1n1_vaccine== 0]
import scipy
scipy.stats.ttest_ind(A_1.is_h1n1_vacc_effective, A_0.is_h1n1_vacc_effective)
'''
pvalue=0.0, not an important varibale as p-value is > 0.05.
'''
## 17. is_h1n1_risky vs h1n1_vaccine

df.is_h1n1_risky.value_counts()
sum(df.is_h1n1_risky.isnull())# 388 missing values
sns.countplot(df.is_h1n1_risky)
sns.countplot(df.is_h1n1_risky, hue = df.h1n1_vaccine)

'''
I will replace the missing values with median category

observations:

1- People who considered absence of the vaccine will be risky took more vaccine shots

I will club categories as following
'''
df['is_h1n1_risky'].fillna(df['is_h1n1_risky'].median(), inplace = True)
sum(df.is_h1n1_risky.isnull())# no missing values
df.is_h1n1_risky.value_counts()
sns.countplot(df.is_h1n1_risky, hue = df.h1n1_vaccine)
'''
2.0    10307
1.0     8139
4.0     5394
5.0     1750
3.0     1117
Name: is_h1n1_risky, dtype: int64
'''
# t-test
A_1= df[df.h1n1_vaccine== 1]
A_0= df[df.h1n1_vaccine== 0]
import scipy
scipy.stats.ttest_ind(A_1.is_h1n1_risky, A_0.is_h1n1_risky)
'''
pvalue=0.0, not an important varibale as p-value is > 0.05.
'''


## 18. sick_from_h1n1_vacc vs h1n1_vaccine

df.sick_from_h1n1_vacc.value_counts()
sum(df.sick_from_h1n1_vacc.isnull())# 395 missing values
sns.countplot(df.sick_from_h1n1_vacc)
sns.countplot(df.sick_from_h1n1_vacc, hue=df.h1n1_vaccine)
cond = df[df['sick_from_h1n1_vacc']==3.0]
cond.h1n1_vaccine.value_counts()
'''
I will replace the missing values with mode category

observations-

1- most people are not worried of getting sick after taking a h1n1 vaccine
2- less number of people were worried of getting sick, as a result we can see that
   most few people did not take the vaccine as we can observe that in the plot
3- based on cond.h1n1_vaccine.value_counts(), we observed that 148 people who doesn't know
   about the h1n1_vaccine affects did not take the vaccine
   

Based on the observations I will club the categories into 2 as follows
worried > 4 & 5
not worried > 1, 2 & 3 (including 3 beacause the data is less in this category, 
                        and because of the observation in point 3)
'''


df['sick_from_h1n1_vacc'].fillna(df['sick_from_h1n1_vacc'].mode()[0], inplace = True)
sum(df.sick_from_h1n1_vacc.isnull())# no missing value

df['sick_from_h1n1_vacc'] = df.get('sick_from_h1n1_vacc').replace(1.0,'not_worried')
df['sick_from_h1n1_vacc'] = df.get('sick_from_h1n1_vacc').replace(2.0,'not_worried')
df['sick_from_h1n1_vacc'] = df.get('sick_from_h1n1_vacc').replace(3.0,'not_worried')
df['sick_from_h1n1_vacc'] = df.get('sick_from_h1n1_vacc').replace(4.0,'worried')
df['sick_from_h1n1_vacc'] = df.get('sick_from_h1n1_vacc').replace(5.0,'worried')

df.sick_from_h1n1_vacc.value_counts()
sns.countplot(df.sick_from_h1n1_vacc, hue=df.h1n1_vaccine)
'''
not_worried    18670
worried         8037
Name: sick_from_h1n1_vacc, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['sick_from_h1n1_vacc'], df['h1n1_vaccine'])
contigency

c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value-5.5700849919347626e-49, can be a good predictor
'''



## 19. is_seas_vacc_effective vs h1n1_vaccine

df.is_seas_vacc_effective.value_counts()
sum(df.is_seas_vacc_effective.isnull())# 462 missing values
sns.countplot(df.is_seas_vacc_effective)
'''
I will fill the missing values with the median category.
will perform t-test as the variable is ordinal
'''
df['is_seas_vacc_effective'].fillna(df['is_seas_vacc_effective'].median(), inplace = True)
sum(df.is_seas_vacc_effective.isnull())# 0 missing values

# t-test
A_1= df[df.h1n1_vaccine== 1]
A_0= df[df.h1n1_vaccine== 0]
import scipy
scipy.stats.ttest_ind(A_1.is_seas_vacc_effective, A_0.is_seas_vacc_effective)
'''
pvalue=1.4482499412706185e-188, can be an important variable.
'''


## 20. is_seas_risky vs h1n1_vaccine

df.is_seas_risky.value_counts()
sum(df.is_seas_risky.isnull())# 514 missing values
sns.countplot(df.is_seas_risky)
sns.countplot(df.is_seas_risky, hue = df.h1n1_vaccine)
'''
I will fill the missing values with the median category

we will club these categories into 2 as follows:
1 - low_risk = 1,2 & 3 (including 3 because the amount of data is very low in this
                        category which is 0.02 % of the whole dataset & most people\
                        in this category did not take the vaccine, so it is more logical
                        to club it into low_risk)
2 - high_risk = 4 & 5
'''

df['is_seas_risky'].fillna(df['is_seas_risky'].median(), inplace =True)
sum(df['is_seas_risky'].isnull())# no missing values

df['is_seas_risky']= df.get('is_seas_risky').replace(1.0,'low_risk')
df['is_seas_risky']= df.get('is_seas_risky').replace(2.0,'low_risk')
df['is_seas_risky']= df.get('is_seas_risky').replace(3.0,'low_risk')
df['is_seas_risky']= df.get('is_seas_risky').replace(4.0,'high_risk')
df['is_seas_risky']= df.get('is_seas_risky').replace(5.0,'high_risk')

df.is_seas_risky.value_counts()
sns.countplot(df.is_seas_risky, hue = df.h1n1_vaccine)
'''
low_risk     16119
high_risk    10588
Name: is_seas_risky, dtype: int64
'''
# chi-square test

contigency = pd.crosstab(df['is_seas_risky'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine       0     1
is_seas_risky             
high_risk       7071  3517
low_risk       13962  2157
'''

c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value = 0.0, rejecting the variable as p-value is > 0.05
'''


## 21. sick_from_seas_vacc vs h1n1_vaccine

df.sick_from_seas_vacc.value_counts()
sum(df.sick_from_seas_vacc.isnull())#537 missing values
sns.countplot(df.sick_from_seas_vacc)
sns.countplot(df.sick_from_seas_vacc, hue = df.h1n1_vaccine)

'''
I will replace the missing values with the median category as the data is ordinal.
 
I'll club the categories as follows-
1 - not_worried = 1 & 2 
2 - worried = 4,5 & 3(including 3 because the amount of data is very low in this
                        category which is 0.003 % of the whole dataset & most people\
                        in this category did not take the vaccine, so it is more logical
                        to club it into worried)
'''

df['sick_from_seas_vacc'].fillna(df['sick_from_seas_vacc'].median(), inplace = True)
sum(df.sick_from_seas_vacc.isnull())# no missing values

df['sick_from_seas_vacc'] = df.get('sick_from_seas_vacc').replace(1.0,'not_worried')
df['sick_from_seas_vacc'] = df.get('sick_from_seas_vacc').replace(2.0,'not_worried')
df['sick_from_seas_vacc'] = df.get('sick_from_seas_vacc').replace(3.0,'worried')
df['sick_from_seas_vacc'] = df.get('sick_from_seas_vacc').replace(4.0,'worried')
df['sick_from_seas_vacc'] = df.get('sick_from_seas_vacc').replace(5.0,'worried')

df.sick_from_seas_vacc.value_counts()
sns.countplot(df.sick_from_seas_vacc)
'''
not_worried    20040
worried         6667
Name: sick_from_seas_vacc, dtype: int64
'''
'''
rejecting the variable as the data is more dominant towards not worried by atleast 76%
'''



## 22. age_bracket vs h1n1_vaccine

df.age_bracket.value_counts()
sum(df.age_bracket.isnull())# no missing values
sns.histplot(df.age_bracket)# histogram
plt.xticks(rotation = 45)
sns.countplot(df.age_bracket, hue=df.h1n1_vaccine)# histogram
plt.xticks(rotation = 45)
'''
age_bracket is a continues variable which is recorded in categories, hence we will
treat it as a categorical variable.

I will club these categories as follows
1- young = 18-34 years
2- middle_aged = 34-44 years & 45-54 years
3- senior citizens = 55-64 years & 65+ years
'''

df['age_bracket']=df.get('age_bracket').replace('18 - 34 Years', 'young')
df['age_bracket']=df.get('age_bracket').replace('35 - 44 Years', 'middle_aged')
df['age_bracket']=df.get('age_bracket').replace('45 - 54 Years', 'middle_aged')
df['age_bracket']=df.get('age_bracket').replace('55 - 64 Years', 'senior_citizens')
df['age_bracket']=df.get('age_bracket').replace('65+ Years', 'senior_citizens')

df.age_bracket.value_counts()
sns.countplot(df.age_bracket, hue=df.h1n1_vaccine)
'''
senior_citizens    12406
middle_aged         9086
young               5215
Name: age_bracket, dtype: int64

from this we can observe that most vaccine shots are given to senior citizens
followed by middle_aged and young people
'''

# chi-square test

contigency = pd.crosstab(df['age_bracket'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine        0     1
age_bracket                
middle_aged      7305  1781
senior_citizens  9504  2902
young            4224   991
'''

c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value = 9.80895497884605e-15, can be a good predictor
'''



## 23. qualification vs h1n1_vaccine

df.qualification.value_counts()
sum(df.qualification.isnull())#1407 missing values
sns.countplot(df.qualification)
plt.xticks(rotation = 45)
sns.countplot(df.qualification, hue = df.h1n1_vaccine)
plt.xticks(rotation = 45)
cond = df[df['qualification'].isnull()]
cond.h1n1_vaccine.value_counts()

'''
I will replace the missing values with mode category

I will club the categories as follows-
1- non_college = < 12 Years & 12 Years
2- college = college Graduate & Some College
'''

df['qualification'].fillna(df['qualification'].mode()[0], inplace = True)
sum(df.qualification.isnull())# no missing values

df['qualification']=df.get('qualification').replace('< 12 Years','non_college')
df['qualification']=df.get('qualification').replace('12 Years','non_college')
df['qualification']=df.get('qualification').replace('College Graduate','college')
df['qualification']=df.get('qualification').replace('Some College','college')

df.qualification.value_counts()
sns.countplot(df.qualification, hue = df.h1n1_vaccine)
'''
college        18547
non_college     8160
Name: qualification, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['qualification'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine       0     1
qualification             
college        14339  4208
non_college     6694  1466
'''
c,p,dof,expected = chi2_contingency(contigency)
p
'''
p-value = 4.135160581862431e-18, can be a good predictor
'''


## 24. race vs h1n1_vaccine

df.race.value_counts()
sum(df.race.isnull())# no missing values
sns.countplot(df.race, hue=df.h1n1_vaccine)
'''
rejecting the variable as data is more dominant towards whire category by 80%, even
if we club the categories data will still be baised
'''


## 25. sex vs h1n1_vaccine
df.sex.value_counts()
sum(df.sex.isnull())# no missing values
sns.countplot(df.sex, hue= df.h1n1_vaccine)
'''
Female    15858
Male      10849
Name: sex, dtype: int64

More females took vaccine shots.
'''

# chi-square Test
contigency = pd.crosstab(df['sex'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine      0     1
sex                      
Female        12378  3480
Male           8655  2194

'''

c,p,dof,exp = chi2_contingency(contigency)
p
'''
p-value = 0.0007709155489949327, can be a good predictor
'''


## 25. income_level vs h1n1_vaccine
df.income_level.value_counts()
sum(df.income_level.isnull())# 4423 missing values
sns.countplot(df.income_level, hue = df.h1n1_vaccine)
plt.xticks(rotation = 45)
pd.crosstab(df['income_level'], df['qualification'])
'''
I will fill the missing values with mode category and replace the name for
2nd category as middle class.
'''

df['income_level'].fillna(df['income_level'].mode()[0], inplace=True)
sum(df.income_level.isnull())# no missing values

df['income_level']=df.get('income_level').replace('<= $75,000, Above Poverty','middle_class')
df['income_level']=df.get('income_level').replace('> $75,000','above_poverty')

df.income_level.value_counts()
'''
middle_class     17200
above_poverty     6810
Below Poverty     2697
Name: income_level, dtype: int64
'''

# chi_squaretest

contigency= pd.crosstab(df['income_level'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine       0     1
income_level              
Below Poverty   2181   516
above_poverty   5087  1723
middle_class   13765  3435
'''
c,p,dof,exp = chi2_contingency(contigency)
p
'''
p-value = 1.880722047715135e-20, important varibale
'''


## 26. marital_status vs h1n1_vaccine

df.marital_status.value_counts()
sum(df.marital_status.isnull())#1408 missing values
sns.countplot(df.marital_status, hue = df.h1n1_vaccine)
'''
I will replace the missing values with mode category

more married people took vaccine shots than unmarried
'''

df['marital_status'].fillna(df['marital_status'].mode()[0], inplace= True)
sum(df.marital_status.isnull())# no missing values
df.marital_status.value_counts()
'''
Married        14963
Not Married    11744
Name: marital_status, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['marital_status'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine        0     1
marital_status             
Married         11539  3424
Not Married      9494  2250
'''

c,p,dof, exp = chi2_contingency(contigency)
p
'''
p-value = 1.6985751321912323e-13, can be a good predictor
'''


## 27. housing_status vs h1n1_vaccine

df.housing_status.value_counts()
sum(df.housing_status.isnull())#2042 missing values
sns.countplot(df.housing_status, hue = df.h1n1_vaccine)

'''
I will fill missing values with mode category

those who own their own house are taking more vaccine shots.
'''
df['housing_status'].fillna(df['housing_status'].mode()[0], inplace = True)
sum(df['housing_status'].isnull())# no missing values
df.housing_status.value_counts()
'''
Own     20778
Rent     5929
Name: housing_status, dtype: int64
'''

# chi-square test 
contigency = pd.crosstab(df['housing_status'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine        0     1
housing_status             
Own             16223  4555
Rent             4810  1119
'''
c,p,dof,exp = chi2_contingency(contigency)
p
'''
p-value = 4.5507157863887266e-07, can be a good predictor
'''


## 28. employment vs h1n1_vaccine
df.employment.value_counts()
sum(df.employment.isnull())#1463 missing values
sns.countplot(df.employment, hue = df.h1n1_vaccine)
'''
I will treat the missing values with the mode category

most of the people those who took vaccine shots are employed
'''
df['employment'].fillna(df['employment'].mode()[0], inplace = True)
sum(df['employment'].isnull())# no missing values
df.employment.value_counts()
'''
Employed              15023
Not in Labor Force    10231
Unemployed             1453
Name: employment, dtype: int64
'''

# chi-square test

contigency = pd.crosstab(df['employment'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine            0     1
employment                     
Employed            11829  3194
Not in Labor Force   7988  2243
Unemployed           1216   237
'''
c,p,dof,exp = chi2_contingency(contigency)
p
'''
p-value = 6.274688521364308e-06, can be a good predictor
'''


## 29. census_msa vs h1n1_vaccine
df.census_msa.value_counts()
sum(df.census_msa.isnull())# no missing values
sns.countplot(df.census_msa, hue = df.h1n1_vaccine)
plt.xticks(rotation = 45)
'''
More people from MSA city (be it principle or not) are taking vaccine shots

I will club categories as

1 - MSA = MSA, Not Principle  City & MSA, Principle City 
2- Non-MSA
'''

df['census_msa']= df.get('census_msa').replace('MSA, Not Principle  City','MSA')
df['census_msa']= df.get('census_msa').replace('MSA, Principle City','MSA')

df.census_msa.value_counts()
sns.countplot(df.census_msa, hue=df.h1n1_vaccine)

# chi-square test
contigency = pd.crosstab(df['census_msa'], df['h1n1_vaccine'])
contigency
'''
h1n1_vaccine      0     1
census_msa               
MSA           15361  4148
Non-MSA        5672  1526
'''
c,p,dof,exp = chi2_contingency(contigency)
p
'''
p-value = 0.9263581190403177, not a good predictor
'''


## 30. no_of_adults vs h1n1_vaccine
df.no_of_adults.value_counts()
sum(df.no_of_adults.isnull())#249 missing values
sns.countplot(df.no_of_adults, hue = df.h1n1_vaccine)
'''
I will treat missing values with median.
'''

df['no_of_adults'].fillna(df['no_of_adults'].median(), inplace = True)
sum(df.no_of_adults.isnull())#0 missing values

# t-test

A_1= df[df.h1n1_vaccine== 1]
A_0= df[df.h1n1_vaccine== 0]
import scipy
scipy.stats.ttest_ind(A_1.no_of_adults, A_0.no_of_adults)
'''
pvalue=0.231407296881803, not an important variable
'''


# 31. no_of_children vs h1n1_vaccine
df.no_of_children.value_counts()
sum(df.no_of_children.isnull())#249 missing values
sns.countplot(df.no_of_children, hue = df.h1n1_vaccine)

'''
I will treat missing values with median, as the data is skewed
'''

df['no_of_children'].fillna(df['no_of_children'].median(), inplace = True)
sum(df.no_of_children.isnull())#0 missing values

# t-test

A_1= df[df.h1n1_vaccine== 1]
A_0= df[df.h1n1_vaccine== 0]
import scipy
scipy.stats.ttest_ind(A_1.no_of_children, A_0.no_of_children)
'''
pvalue=0.6748820649081326, rejecting the variable
'''


### model building

'''
copying the data base to a new varibale excluding rejected variables
'''
df1 = df.iloc[:,[1,2,4,7,8,9,11,12,18,19,22,23,25,26,27,28,29,33]]
df1.columns
import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.glm(formula='''h1n1_vaccine~h1n1_worry+h1n1_awareness+contact_avoidance+
                avoid_large_gatherings+reduced_outside_home_cont+avoid_touch_face+
                dr_recc_seasonal_vacc+chronic_medic_condition+sick_from_h1n1_vacc+
                is_seas_vacc_effective+age_bracket+qualification+sex+income_level+marital_status
                +housing_status+employment''',
                data = df1, family = sm.families.Binomial())

result = model.fit()
print(result.summary())

predictions = result.predict()
predictions_nominal = [0 if x<0.5 else 1 for x in predictions]


#confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df1['h1n1_vaccine'], predictions_nominal))

# accuracy
((206584+457)/(20658+357+5217+457))
## 0.7906

# ROC curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = roc_curve(df1['h1n1_vaccine'], predictions)
roc_auc = auc(fpr,tpr)
roc_auc# 0.72

plt.title("ROC curve for h1n1_vaccine classifier")
plt.xlabel("False positive rate (1-specificity)")
plt.xlabel("False positive rate (1-specificity)")
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(fpr, tpr, label = "AUC=" +str(roc_auc))
plt.legend(loc = 4)
plt.show

#classification report
print(classification_report(df['h1n1_vaccine'], predictions_nominal, digits = 3))

df1.isnull().sum()



## Model 2 after removing insignificant variables
'''
isignificant variables
1- housing_status
2- income_level
3- contact_avoidance
4- reduced_outside_home_cont
'''
model2 = smf.glm(formula='''h1n1_vaccine~h1n1_worry+h1n1_awareness+
                avoid_large_gatherings+avoid_touch_face+
                dr_recc_seasonal_vacc+chronic_medic_condition+sick_from_h1n1_vacc+
                is_seas_vacc_effective+age_bracket+qualification+sex+marital_status
                +employment''',
                data = df1, family = sm.families.Binomial())


result = model2.fit()
print(result.summary())

predictions = result.predict()
predictions_nominal = [0 if x<0.5 else 1 for x in predictions]

#confusion matrix
print(confusion_matrix(df['h1n1_vaccine'], predictions_nominal))

# accuracy
((20673+440)/(20673+440+5234+360))
## 0.7906

# ROC curve
fpr, tpr, thresholds = roc_curve(df1['h1n1_vaccine'], predictions)
roc_auc = auc(fpr,tpr)
roc_auc# 0.70

plt.title("ROC curve for h1n1_vaccine classifier")
plt.xlabel("False positive rate (1-specificity)")
plt.xlabel("False positive rate (1-specificity)")
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(fpr, tpr, label = "AUC=" +str(roc_auc))
plt.legend(loc = 4)
plt.show

#classification report
print(classification_report(df['h1n1_vaccine'], predictions_nominal, digits = 3))




'''
Now we will build a logistic model with these significant variables
'''
# Dropping insignificant vars

df1 = df1.drop(['housing_status','income_level','contact_avoidance','reduced_outside_home_cont'], axis = 1)
df1.shape
df2 = df1.copy()
df2.columns

# scaling the variables

from sklearn.preprocessing import StandardScaler
std_scale= StandardScaler()
df2['h1n1_awareness'] = std_scale.fit_transform(df2[['h1n1_awareness']])
df2['avoid_large_gatherings'] = std_scale.fit_transform(df2[['avoid_large_gatherings']])
df2['avoid_touch_face'] = std_scale.fit_transform(df2[['avoid_touch_face']])
df2['dr_recc_seasonal_vacc'] = std_scale.fit_transform(df2[['dr_recc_seasonal_vacc']])
df2['chronic_medic_condition'] = std_scale.fit_transform(df2[['chronic_medic_condition']])
df2['is_seas_vacc_effective'] = std_scale.fit_transform(df2[['is_seas_vacc_effective']])

df2[['h1n1_awareness','avoid_large_gatherings','avoid_touch_face','dr_recc_seasonal_vacc','chronic_medic_condition','is_seas_vacc_effective']].head(3)

## encoding categorical variables


df_dum = pd.get_dummies(df2)
df_dum.columns

# balancing the dataset
#our dataset is imbalnces as it can be observed by the plot below

sns.countplot(df_dum.h1n1_vaccine)

X = df_dum.drop(['h1n1_vaccine'], axis = 1)
Y = df_dum['h1n1_vaccine']
from imblearn.over_sampling import SMOTE
# OverSampling using SMOTE
smote = SMOTE(random_state = 402)
X_smote, Y_smote = smote.fit_resample(X,Y)

print(X_smote.shape, Y_smote.shape)

from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(Y_smote)))

#After re-sampling
df = pd.concat([X_smote, Y_smote], axis=1)
sns.countplot(df.h1n1_vaccine)
df.h1n1_vaccine.value_counts()
df.h1n1_vaccine.shape

# spliting the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_smote, Y_smote, test_size = 0.2 ,random_state = 42)





##***** Logistic Regression ******##

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
from sklearn.metrics import confusion_matrix

# A parameter grid for Logistic Regression
params = {
        'C': [0.001,0.01,0.1,1,10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
         }

from sklearn.model_selection import RandomizedSearchCV

random_cv=RandomizedSearchCV(estimator=LR,param_distributions=params,
                             cv=5,n_iter=5,scoring='roc_auc',n_jobs=1,verbose=3,return_train_score=True,random_state=121)
random_cv.fit(X_train,y_train)

#best parameter 
random_cv.best_params_

#Model Building
LR = LogisticRegression(C = 1, solver = 'lbfgs').fit(X_train,y_train)
LR

#Prediction & Evaluation

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
yhat[0:5]
yhat_prob[0:5]

from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat)

#Confusion matrix plotting
from sklearn.metrics import confusion_matrix
labels = ['will take', 'wont take']
cm=confusion_matrix(y_test, yhat)
cm
axes=sns.heatmap(cm, square=True, annot=True,fmt='d',cbar=True,cmap=plt.cm.Blues)
ticks=np.arange(len(labels))+0.5
plt.title('Confusion matrix of the classifier')
plt.xlabel('True')
plt.ylabel('Predicted')
axes.set_xticks(ticks)
axes.set_xticklabels(labels,rotation=0)
axes.set_yticks(ticks)
axes.set_yticklabels(labels,rotation=0)


from sklearn.metrics import classification_report
print (classification_report(y_test, yhat))



#different accuracy scores
from sklearn.metrics import log_loss
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
print("Logistic Regression's Accuracy: ", metrics.accuracy_score(y_test, yhat))
print("Logistic Regression's LogLoss : ", log_loss(y_test, yhat_prob))
print("Logistic Regression's F1-Score: ", f1_score(y_test, yhat, average='weighted'))
-np.mean(cross_val_score(LR,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 4))


#ROC curve
#!pip install scikit-plot
import scikitplot as skplt
y_true = y_test
y_probas = yhat_prob
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()


'''
Clasification report

              precision    recall  f1-score   support

           0       0.73      0.83      0.78      4287
           1       0.79      0.69      0.74      4127

    accuracy                           0.76      8414
   macro avg       0.76      0.76      0.76      8414
weighted avg       0.76      0.76      0.76      8414


Confusion Matrix

array([[3538,  749],
       [1276, 2851]], dtype=int64)


ROC/AUC = 0.84

'''














