import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from matplotlib.widgets import Cursor

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')
plt.rcParams['lines.linewidth'] = 0.75

#----------------------------------------------- FUNCTIONS

def valueScaler(dataframe,low,high):
    temp_df = pd.DataFrame(dataframe.copy())
    df_index = dataframe.index
    try:
        df_columns = dataframe.columns
    except:
        pass
    temp_dates = pd.RangeIndex(0,int(len(temp_df)),1)
    temp_df.index = temp_dates
    x = temp_df.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(low, high))
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_df = pd.DataFrame(x_scaled)
    scaled_df.index = df_index
    try:
        scaled_df.columns = df_columns
    except:
        pass
    return scaled_df

def correlationValue(dataframe1,dataframe2):
    temp_df1 = pd.DataFrame(dataframe1)
    temp_df1 = temp_df1.rename(columns={temp_df1.columns[0]:'1'})
    temp_df1.index = np.arange(0,len(dataframe1),1)
    temp_df2 = pd.DataFrame(dataframe2)
    temp_df2 = temp_df2.rename(columns={temp_df2.columns[0]:'2'})
    temp_df2.index = np.arange(0,len(dataframe2),1)
    corr_df = pd.concat([temp_df1,temp_df2],axis=1)
    corrValue = (corr_df.corr()).iloc[0][1]
    return corrValue

def plotResults(df1,df2,df3,low,high,daterange,plotTitle,render,show,filename):

    plotMargin = daterange[1] - daterange[0]

    df2DateRange = [(df2.index[0] + plotMargin).strftime('%d/%m/%Y'),(df2.index[len(df2)-1] - plotMargin).strftime('%d/%m/%Y')]
    df2 = reindexHistDataframe(df2,[daterange[0],daterange[3]])
    df2 = valueScaler(df2,df1[daterange[0]:daterange[3]].min(),df1[daterange[0]:daterange[3]].max()) 

    df3DateRange = [(df3.index[0] + plotMargin).strftime('%d/%m/%Y'),(df3.index[len(df3)-1] - plotMargin).strftime('%d/%m/%Y')]
    df3 = reindexHistDataframe(df3,[daterange[0],daterange[3]])
    df3 = valueScaler(df3,df1[daterange[0]:daterange[3]].min(),df1[daterange[0]:daterange[3]].max())
    
    fig, ax = plt.subplots(figsize=(12,4))
    
    plt.plot(df1,label='Target')
    df2_label =  df2DateRange[0] + ' - ' + df2DateRange[1] + ' | ' + str(round(low, 3))
    plt.plot(df2, label=df2_label)
    df3_label = df3DateRange[0] + ' - ' + df3DateRange[1] + ' | ' + str(round(high, 3))
    plt.plot(df3, label=df3_label)

    plt.axvspan(daterange[0], daterange[1], alpha=0.05, color='black')
    plt.axvspan(daterange[2], daterange[3], alpha=0.05, color='black')
    
    ax.set_xlim([daterange[0],daterange[3]])

    hCursor = Cursor(ax,useblit=True, linewidth=0.5, color='C0', linestyle='dashed', dashes=[10, 10])
    
    plt.title(plotTitle)
    plt.legend(loc=1)

    if render is True:
        fig.savefig(filename, bbox_inches='tight')
    if show is False:
        plt.close()
    if show is True:
        plt.show()

def plotDataframes(target_df,check_df,daterange,plotTitle,render,show,filename):
    
    fig, ax = plt.subplots(figsize=(12,4))

    plt.plot(target_df,label='Target')
    plt.plot(check_df,label='Historical')

    plt.axvspan(daterange[0], daterange[1], alpha=0.05, color='black')
    plt.axvspan(daterange[2], daterange[3], alpha=0.05, color='black')
    
    ax.set_xlim([daterange[0],daterange[3]])

    hCursor = Cursor(ax,useblit=True, linewidth=0.5, color='C0', linestyle='dashed', dashes=[10, 10])
    
    plt.title(plotTitle)
    plt.legend(loc=1)

    if render is True:
        fig.savefig(filename, bbox_inches='tight')
    if show is False:
        plt.close()
    if show is True:
        plt.show()

def reindexHistDataframe(dataframe,daterange):
    newIndex = pd.date_range(daterange[0],daterange[1])
    temp_df = pd.DataFrame(dataframe.iloc[:].to_numpy(),index=pd.RangeIndex(0,len(dataframe)))   
    reindexed_df = pd.DataFrame(np.arange(0,len(newIndex),1),index=pd.RangeIndex(0,len(newIndex)))
    reindexed_df = reindexed_df.rename(columns={0:'TEMP'})
    reindexed_df = pd.concat([reindexed_df,temp_df],axis=1)
    reindexed_df = reindexed_df.drop(columns=['TEMP'])
    reindexed_df.index = newIndex
    return reindexed_df

def findHistoricalCorr(dataframe,comparison_df,period,step,visibleMargin,render,show,renderResults):

    target_df = dataframe[-period:]

    plotDateRange = [target_df.index[0]-timedelta(days=visibleMargin),
                     target_df.index[0],
                     target_df.index[len(target_df)-1],
                     target_df.index[len(target_df)-1]+timedelta(days=visibleMargin)]

    highestCorr = [0,0,0]
    lowestCorr = [0,0,0]

    timestamp = time.strftime("%d-%m-%Y-%H-%M")

    loopLength = len(dataframe) - period + step  
    for a in range(step,loopLength,step):

        startDate = int(len(dataframe) - period - a - (visibleMargin))
        endDate = int(len(dataframe) - a + (visibleMargin))
        
        if startDate >= (0):
            
            hist_df = comparison_df[startDate:endDate]
            hist_df = valueScaler(hist_df,target_df.min(),target_df.max())
            sliced_hist_df = hist_df[visibleMargin:-visibleMargin]
            tempCorrValue= correlationValue(target_df,sliced_hist_df)
            print(startDate,endDate,str(round(tempCorrValue, 3)))

            if tempCorrValue > highestCorr[0]:
                highestCorr[0] = tempCorrValue
                highestCorr[1] = startDate
                highestCorr[2] = endDate

            if tempCorrValue < lowestCorr[0]:
                lowestCorr[0] = tempCorrValue
                lowestCorr[1] = startDate
                lowestCorr[2] = endDate

            plotTitle = 'Correlation: ' + str(round(tempCorrValue, 3)) + ' | Target: '
            plotTitle += plotDateRange[1].strftime('%d/%m/%Y') + ' - ' + plotDateRange[2].strftime('%d/%m/%Y')
            plotTitle += ' | Historical: '+ sliced_hist_df.index[0].strftime('%d/%m/%Y') + ' - ' +  sliced_hist_df.index[len(sliced_hist_df)-1].strftime('%d/%m/%Y')
            
            hist_df = reindexHistDataframe(hist_df,[plotDateRange[0],plotDateRange[3]])

            filename = "output/hist_autocorr_" + timestamp + '_' + str(int(a/step)) + ".jpg"
            plotDataframes(dataframe,hist_df,plotDateRange,plotTitle,render,show,filename)

        else:
            pass

    print(lowestCorr)
    print(highestCorr)

    lowCorr_df = dataframe[lowestCorr[1]:lowestCorr[2]]   
    highCorr_df = dataframe[highestCorr[1]:highestCorr[2]]
    
    resultsTitle = 'Date: ' + plotDateRange[1].strftime('%d/%m/%Y')+ ' - ' + plotDateRange[2].strftime('%d/%m/%Y')
    resultsTitle += ' | Length: ' + str(period)
    resultsTitle += ' | Offset: ' + str(step)
    resultsTitle += ' | Plot Margin: '+ str(visibleMargin)
    
    filename = "output/hist_autocorr_" + timestamp + "_results.jpg"
    plotResults(dataframe[plotDateRange[0]:plotDateRange[3]],lowCorr_df,highCorr_df,lowestCorr[0],highestCorr[0],plotDateRange,resultsTitle,renderResults,True,filename)

#----------------------------------------------- DATA

df = pd.read_csv('data/data.csv',index_col=0)
df.index = pd.to_datetime(df.index)
df = df['Close']

#----------------------------------------------- PLOT

#main dataframe,comparison dataframe,period,step,visibleMargin,render,show,render results
findHistoricalCorr(df,df,100,10,50,True,False,True)

