# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:18:04 2021

@author: ASUS
"""


import pandas as pd
import numpy as np
from numpy.random import randint as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib


path = 'D:/University related/پایان نامه/مقاله دوم پایان نامه/figures/wa_trip_prediction_data.csv'

data = pd.read_csv(path, sep=',')

color_palette = sns.color_palette('Set3')
# plt.rcParams.update({'font.size': 12})
# set font to "Times New Roman" globaly
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams.update({'font.weight': 'ultralight'})



#Arial , Times New Roman , Symbol, Courier
font = {'family' : 'Roboto',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


data.M_pattern.replace(
    {'1_edu':'1 education', '1_sch':'1 school',
     '1_sch_1_edu':'1 school 1 education',
     '1_work':'1 work', 
     '1_work_1_edu': '1 work 1 education',
     '2_work':'2 works', '3_work': '3 works',
     'none': 'None'
     }, inplace=True
    )


# draw distplot for taz related features

# binsize = 20

# plt.figure(figsize=(15, 20))

# plt.subplot(3, 2, 1)
# sns.distplot(data['MTLRTSHR'], bins=binsize, color='gray')

# plt.subplot(3, 2, 2)
# sns.distplot(data['MTLRTLNG'], bins=binsize, color='gray')

# plt.subplot(3, 2, 3)
# sns.distplot(data['ALLPKSHR'], bins=binsize, color='gray')

# plt.subplot(3, 2, 4)
# sns.distplot(data['ALLPKLNG'], bins=binsize, color='gray')

# plt.subplot(3, 2, 5)
# sns.distplot(data['ALLOPSHR'], bins=binsize, color='gray')

# plt.subplot(3, 2, 6)
# sns.distplot(data['ALLOPLNG'], bins=binsize, color='gray')
# plt.savefig('./distplot20.tif', bbox_inches='tight')
# plt.show()

# draw histogram for taz related features

# plt.figure(figsize=(20, 10))

# plt.subplot(2, 3, 1)
# sns.histplot(data['MTLRTSHR'], bins=binsize, color='gray')

# plt.subplot(2, 3, 2)
# sns.histplot(data['MTLRTLNG'], bins=binsize, color='gray')

# plt.subplot(2, 3, 3)
# sns.histplot(data['ALLPKSHR'], bins=binsize, color='gray')

# plt.subplot(2, 3, 4)
# sns.histplot(data['ALLPKLNG'], bins=binsize, color='gray')

# plt.subplot(2, 3, 5)
# sns.histplot(data['ALLOPSHR'], bins=binsize, color='gray')

# plt.subplot(2, 3, 6)
# sns.histplot(data['ALLOPLNG'], bins=binsize, color='gray')
# plt.savefig('D:/figures/histplot20.svg', bbox_inches='tight')
# plt.show()

### draw barplot for mandatory trip and non mandatory trips

#to avoid random order of bar plots
# data.sort_values(
#     'M_pattern', inplace=True,
#     ascending=True
#     )

# mandatory trips    
## for gender

# plt.figure(figsize=(8,5))

# ax = sns.countplot(
#     data=data, x='M_pattern',
#     hue='R_SEX', palette=['#FF0B00',"#667EF4"]
# )    # amazing!!!! you can pass colors as palette of HEX color codes

# ax = sns.countplot(
#     data=data, x='M_pattern',
#     hue='R_SEX', palette=color_palette, fill=False
#     , lw=0.5, ec='black', order=[
#         '1 work', '2 works', '3 works',
#         '1 work 1 education', '1 education',
#         '1 school', '1 school 1 education',
#         'None'
#         ]
#     )
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Mandatory trip pattern')
# plt.ylabel('Count')

# hatches= ['//', "x",'..', '||', '**', '++',
#  'xx', 'OO', '..', '**']

# for hatch_pattern, these_bars in zip(hatches, ax.containers):
#     for this_bar in these_bars:
#         this_bar.set_hatch(3 * hatch_pattern)

# legend_labels, _ = ax.get_legend_handles_labels()

# ax.legend(
#     legend_labels, ['Male', 'Female'],
#     title='Gender', bbox_to_anchor=(1, 1)
#     )

# plt.savefig('./gender_count.svg', bbox_inches='tight')
# plt.show()
#or
# (data.groupby('R_SEX')['M_pattern'].value_counts()
#  .unstack('R_SEX').plot.bar())
###################################################################
# non-mandatory trips

# define age ranges
# data['R_AGE_bins'] = pd.cut(
#     data['R_AGE'], bins=[0,18,30,45,60,106], right=False
#     )

# plt.figure(figsize=(12, 5))

# ax = sns.countplot(
#     data=data, x='M_pattern', hue='R_AGE_bins',
#     palette=color_palette, fill=False, lw=0.5,
#     ec='black', order=[
#         '1 work', '2 works', '3 works',
#         '1 work 1 education', '1 education',
#         '1 school', '1 school 1 education',
#         'None'
#         ]
#     )

# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Mandatory trip pattern')
# plt.ylabel('Count')

# for hatch_pattern, these_bars in zip(hatches, ax.containers):
#     for this_bar in these_bars:
#         this_bar.set_hatch(3 * hatch_pattern)
        
# legend_labels, _ = ax.get_legend_handles_labels()
# plt.legend(
#     legend_labels, [
#         '0-18', '18-30', '30-45',
#         '45-60', '60 and more'
#         ]
#     , title='Age range',
#     bbox_to_anchor=(1, 1)
#     )

# plt.savefig('./age_count.svg', bbox_inches='tight')
# plt.show()

###################################################################
### Count plot for income range

data.sort_values(
    'M_pattern', inplace=True,
    ascending=True
    )

plt.figure(figsize=(9, 5))
ax = sns.countplot(
    data=data, x='M_pattern', hue='HHFAMINC',
      palette=color_palette, fill=False, lw=0.5,
    ec='black' , order=[
        '1 work', '2 works', '3 works',
        '1 work 1 education', '1 education',
        '1 school', '1 school 1 education',
        'None'
        ]
    )

plt.xticks(rotation=45, ha='right')
plt.xlabel('Mandatory trip pattern')
plt.ylabel('Count')

hatches= ['/', "x", '|', '*', '++',
  'xx', 'OO', '..', '**']

for hatch_pattern, these_bars in zip(hatches, ax.containers):
    for this_bar in these_bars:
        this_bar.set_hatch(3 * hatch_pattern)
       
legend_labels, _ = ax.get_legend_handles_labels()
  
plt.legend(
    legend_labels, [
        'Less than 10,000 to 50,000 USD (Low)',
        '50,000 to 100,000 USD (Mid)',
        '100,000 to 150,000 USD (Up-mid)',
        'More than 150,000 USD (High)'
        ]
    , title='Income categories'
    
    )

plt.savefig('./income_count.svg', bbox_inches='tight')
plt.show()

### a pie chart to show mode choice
# capitaler = lambda x: x.capitalize()

# data.pmode = data.pmode.apply(capitaler)

# mode_count = data.pmode.value_counts()

# #

# plt.figure()
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
# #explsion
# explode = (0.05,0.05,0.05)

# plt.pie(
#     mode_count.values, colors = colors,
#     labels=[
#             'Private car users', 'Public transport users',
#             'Neutral'
#         ]
#     , autopct='%1.1f%%',
#     startangle=100, pctdistance=0.85,
#     explode = explode
#     )
#draw circle
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# plt.gcf()
# plt.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
# plt.tight_layout()
# plt.show()

# plot
# data_o = data[data.other <= 4]
# plt.figure(figsize=(10,8))
# ax = sns.lineplot(
#     data=data_o, x='R_AGE',
#     y='other', hue='R_SEX',
#     palette=[(0,0,0), ((0,0,0))],
#     estimator=None
#     )
# ax.lines[0].set_linestyle('--')
# ax.lines[1].set_linestyle('dotted')
# plt.xlabel('Age')
# plt.ylabel('Non-madatory trip count')
# plt.legend(title='Gender', labels=['Female', 'Male'])
# plt.savefig('D:/figures/non_mandatory.svg', bbox_inches='tight')
# plt.show()


### Confusion matrix for mandatory trips
data_c_m = dict()

# c stands for confusion 
data_c_m['None'] =  [1909, 317, 37, 5, 37, 5, 6, 249]
data_c_m['1 work'] =  [296, 2112, 152, 14, 13, 10, 0, 15]
data_c_m['2 works'] =  [78, 388, 212, 12, 5, 1, 0, 0]
data_c_m['3 works'] =  [29, 196, 63, 12, 0, 0, 0, 0]
data_c_m['1 education'] =  [30, 3, 1, 0, 63, 4, 0, 0]
data_c_m['1 work 1 education'] =  [8, 15, 6, 0, 5, 7, 0, 0]
data_c_m['1 school 1 edu'] =  [7, 0, 0, 0, 0, 0, 2, 32]
data_c_m['1 school'] =  [47, 0, 0, 0, 0, 0, 3, 766]

# create a dataframe of confusion matrix
c_df_m = pd.DataFrame(data=data_c_m, index=data_c_m.keys())
# add a column to calculate percentage
c_df_m['sum'] = c_df_m.sum(axis=1)
# get the percentage
c_df_m = c_df_m.apply(lambda x: x/x.max(), axis=1)

### Confusion matrix for mandatory trips
data_c_n = dict()
data_c_n['1 trip'] = [1322, 134, 145, 303]
data_c_n['2 trips'] = [475, 396, 118, 562]
data_c_n['3 trips'] = [365, 155, 226, 489]
data_c_n['4 trips and more'] = [449, 252, 137, 1644]

c_df_n = pd.DataFrame(data=data_c_n, index=data_c_n.keys())
# add a column to calculate percentage
c_df_n['sum'] = c_df_n.sum(axis=1)
# get the percentage
c_df_n = c_df_n.apply(lambda x: x/x.max(), axis=1)

# drop the artificially made sum col
c_df_n.drop('sum', axis=1, inplace=True)
c_df_m.drop('sum', axis=1, inplace=True)

# draw the non mandatory trip heatmap
plt.figure(figsize=(5,3))

ax = sns.heatmap(
    c_df_n.round(2), cmap='Oranges', # CMRmap
    annot=True, linewidth=0.3,
    linecolor='black', square=True,
    )
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')
plt.xlabel('Predictions')
plt.ylabel('Actual labels')
plt.title('Non-nandatory Trips')
plt.savefig('./nonmandatory_confusion.svg', bbox_inches='tight')
plt.show()

###################################################################
# mandatory
plt.figure(figsize=(7,5))
ax = sns.heatmap(
    c_df_m.round(2), cmap='Purples',
    annot=True, linewidth=0.3,
    linecolor='black', square=True
    )

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')
plt.xlabel('Predictions')
plt.ylabel('Actual labels')
plt.title('Mandatory Trips')
plt.savefig('./mandatory_confusion.svg', bbox_inches='tight')
plt.show()

### draw a plot for non-mandatory trips


# set all trips more tha 4 to 4
data.loc[data['other'] >= 4, 'other'] = 4

# change type to string so that it can be grouped
data.OTHER = data.OTHER.astype('str')

# define a new column to sum values over
data['R_AGE_bins'] = pd.cut(
    data['R_AGE'], bins=[0,18,30,45,60,106], right=False
    )

data['sum'] = 1
non_mand_df = (data
             .groupby(['R_AGE_bins', 'OTHER'])['sum']
             .sum().reset_index())

# set category type to str and change the names as desired
non_mand_df.R_AGE_bins = (non_mand_df.R_AGE_bins
                         .astype('str')
                         .replace({
                             '[0, 18)':'18 & younger','[18, 30)':'18 to 30',
                             '[30, 45)':'30 to 45', '[45, 60)':'45 to 60',
                             '[60, 106)': '60 & older'})
                         )
# make a function to plot the desired liplot
def make_lineplot(df, name):
    fig = plt.figure()
    plt.xlabel('Age range')
    plt.ylabel('Count')
    #plt.title(name)
    #plt.xticks()
    plt.grid('True')
    
    markers = ["x", "*", "s", "P", "o"]
    linestyle = ["dashed", "solid", "solid", "dashdot", "dotted"]
    color = ['green','red','blue','hotpink','darkviolet']
    n=0
    for age_range in df.OTHER.unique():
        # for each unique age range
        sub_df = df[df.OTHER==age_range]
        plt.plot(
            sub_df['R_AGE_bins'], sub_df['sum'],
            marker=markers[n], label=age_range,
            color=color[n], linestyle=linestyle[n])
        n += 1
    plt.legend(title='No. of trip', bbox_to_anchor=(1, 1))
    plt.savefig(
        './'+name+'.svg', bbox_inches='tight')
    plt.show()
    plt.close()

make_lineplot(non_mand_df, 'Nonemandatory_age_lineplot')















