
# code to take measurements from cyclope and compare them to meteorolgical data, and also to seeing mwasured in a telescope (LAST)
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt  
import os
import pandas as pd
import datetime as dt

########################## input- fill in ##################################

path = r"D:\Master\correlation"
os.chdir(path)
datafile = 'Seeing_Data.txt'
site = 'Neot_Smadar'
start = pd.to_datetime('2020-11-03 18:00:00')
end = pd.to_datetime('2020-11-04 05:30:00')
datestr = str(start.day)+'.'+str(start.month)+'.'+str(start.year)
datestr2 = str(end.day)+'.'+str(end.month)+'.'+str(end.year)

############################## functions ####################################

def data(datafile):
    file = open(datafile, "r")
    data = []
    for line in file:               # converts the data .txt file to a list of lists 
        file = open(datafile, "r")
        stripped_line = line. strip()
        line_list = stripped_line. split()
        data.append(line_list)
        file.close() 
    
    Date = []
    LST = []
    Seeing = []
    r0 = []
    row = 0
    while row < len(data):             # extract values of Date, Hour (LST) and seeing from data - to new lists each.
        Date.append(data[row][0])
        LST.append(data[row][4])
        Seeing.append(float(data[row][10]))
        r0.append(float(data[row][12]))      # r0 in mm 
        row += 1

    d = {'Date': Date, 'LST': LST, 'Seeing': Seeing, 'r0': r0}
    df = pd.DataFrame(data=d)
    
    time = pd.to_datetime(df['Date'] + ' ' + df['LST'],format = '%d/%m/%Y %H:%M:%S')
    df = pd.DataFrame({'time': time, 'seeing': df.Seeing, 'r0': df.r0 })
    return df 

def splicing (df,start,end):
    """ input: datafile = .txt file of seeing data ("Seeing_Data.txt"), 
    start, end = date and time of first and last measuremnts wanted, in form dd-mm-yy HH:MM:SS'
    site = observation site. no spaces
        output: spliced table of cyclope data, of the night between start_time and end_time"""
    
    for i in list(range(0,len(df))):
        if df.time.iloc[i].hour == 0:            # fix date for hour 00:-- and 01:--
            df.time.iloc[i] += dt.timedelta(days=1)
        elif df.time.iloc[i].hour == 1:
            df.time.iloc[i] += dt.timedelta(days=1)

    mask1 = (df['time'] >= start) & (df['time'] <= end) 
    df = df.loc[mask1]
    return df

def splicemet (file,start,end,rel) :
    """ 
    input: file = meteorological data csv, 
    date1 = first night, date2 = second night, 
    rel = spliced table of seeing data
    output: dataframe with relevant met parameters for specific night"""
    
    meteo = pd.read_csv(file, names = ["Date","Hour_LST", "Temp(C)", "Max_temp", "Min_temp", "Nan", "RH(%)", "Nan1","Nan2","Nan3","Nan4", "Rain(mm)", "Wind_speed", 
                                                 "Wind_dir","std_wind_dir", "Wind_gust", "Wind_gust_dir", "Max_min", "Max_10min", "end_10"],index_col = False) 
    met = meteo.drop(['Nan', 'Nan1', 'Nan2', 'Nan3', 'Nan4'], 1)
    time = pd.to_datetime(met['Date'] + ' ' + met['Hour_LST'], format = '%d/%m/%Y %H:%M')
    met['time'] = time
    mask = (met['time'] >= start) & (met['time'] <= end) 
    met = met.loc[mask]
    met = met.drop(['Date','Hour_LST'],1)
    met = met.set_index(met.time)
    met['Temp(C)'] = pd.to_numeric(met['Temp(C)'])
    met['RH(%)'] = pd.to_numeric(met['RH(%)'])
    
    i = 0
    while i < len(met.time)-1:
        if met.time[0] != av_10.index[0]:
            print('\x1b[1;30;43m' + 
                  '\n Error! - first hour does not match.' 
                  'please make sure the starting time is the same for met and av_10' 
                  + '\x1b[0m') 
            break
        elif met.time[i].hour != av_10.index[i].hour:
            met = met.drop(met.index[i],0)
        elif met.time[i].minute != av_10.index[i].minute:
            met = met.drop(met.index[i],0)
        else: i+=1
    return met 
    
def plotseeing (X,Y, metY,start, end, rel):
    """ input: X (minutes), Y (seeing), date1, date2, rel.
        output: scatter plot saved to path and presented"""
    dates = mp.dates.date2num(X)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter = plt.plot_date(dates, Y, 'ro', markersize = 1)
    ax1.scatter =  plt.plot_date(dates, metY,'bo', markersize = 1)
    plt.ylabel('Seeing (arcsec)', fontsize = 10)
    plt.xlabel('Time (mm-dd HH)', fontsize =10)
    datestr = str(start.day) + '.' + str(start.month) + '.' + str(start.year)
    datestr2 = str(end.day) + '.' + str(end.month) + '.' + str(end.year)
    tit = site+ ' ' + datestr + ' ' + datestr2
    plt.yticks(np.arange(0, 1.1, step = 0.5), fontsize = 9)
    plt.xticks(fontsize = 9, rotation = 30)
    plt.title(tit, fontsize = 10)
   
    figname = datestr +'_'+Y.name +'_'+ metY.name+'_'+site+'.pdf'
    plt.tight_layout()
    lgd = ax1.legend(labels = [Y.name, metY.name], loc = 4, fontsize = 9)
    plt.savefig(figname, bbox_extra_artists=(lgd,))  
    return ax1

def oneplot(norm,site,datestr,datestr2,name):
    fig = plt.figure(figsize=(6,11))
    fig.suptitle(site+' '+datestr+'-'+datestr2, fontsize = 12) 
    x = 1
    for i in list(range(1,8)):
        S = norm.seeing
        Y = norm.iloc[:,x]
        X = mp.dates.date2num(av_10.time)
        ax1 = fig.add_subplot(420+i)
        ax1.scatter = plt.plot_date(X, S, 'ro', markersize = 1)
        ax1.scatter = plt.plot_date(X, Y, 'bo', markersize = 1)
        xformatter = mp.dates.DateFormatter('%H')
        ax1.xaxis.set_major_formatter(xformatter)
        plt.ylabel('Seeing (arcsec)', fontsize = 10)
        plt.xlabel('Time (hour)', fontsize =10)
        plt.yticks(np.arange(0, 1.1, step = 0.5), fontsize = 9)
        plt.xticks(fontsize = 9, rotation = 0)
        lgd = ax1.legend(labels = [S.name, Y.name], loc = 1, fontsize = 9)
        x += 1 
    plt.tight_layout()
    plt.savefig('plot '+name+'.pdf', bbox_extra_artists=(lgd,))  
    return fig 

################################# main ###################################

df = data(datafile)
rel = splicing(df.copy(),start,end)
rel = rel.set_index(rel.time)
av_5 = rel.resample('5T').mean()
av_10 = av_5.resample('10T').mean().dropna()
av_10 ['time'] = av_10.index
met = splicemet('ims_data_nov.csv', start,end, rel)
#av_10 = av_10.drop(0) # if the first row does not match first row in met
#met = met.drop(met.index[len(met)-1]) #if last row doesnt match row of av_10

norm = pd.DataFrame({      'seeing': av_10.seeing/max(av_10.seeing), 
                           'Temp(C)': met['Temp(C)']/max(met['Temp(C)']),
                           'RH(%)': met['RH(%)']/max(met['RH(%)']),
                           'Wind_speed': met['Wind_speed']/max(met['Wind_speed']),
                           'Wind_direction': met['Wind_dir']/max(met['Wind_dir']),
                           'std_wind_direction': met['std_wind_dir']/max(met['std_wind_dir']),
                           'Wind_gust': met['Wind_gust']/max(met['Wind_gust']),
                           'Wind_gust_dir': met['Wind_gust_dir']/max(met['Wind_gust_dir'])})

correlation = norm.corr(method='pearson') #options: ‘pearson’, ‘kendall’, ‘spearman’
print(correlation)
name ='correlation ' + str(start.day) + '.' + str(start.month) + '.' + str(start.year) 
correlation.to_csv(name+'.csv')
plot = oneplot(norm,site,datestr,datestr2,name)

# np.corrcoef(av_10.seeing,met['Temp(C)']) # pearson with numpy. should be same. 


"""
***if a pdf of each correlation is wanted***
ax1 = plotseeing(met.time,av_10.seeing,norm['Temp(C)'],start,end,rel)  # the plots of each parameter vs seeing
ax1 = plotseeing(met.time,av_10.seeing,norm['RH(%)'],start,end,rel)
ax1 = plotseeing(met.time,av_10.seeing,norm['Wind_speed'],start,end,rel)
ax1 = plotseeing(met.time,av_10.seeing,norm['Wind_direction'],start,end,rel)
ax1 = plotseeing(met.time,av_10.seeing,norm['std_wind_direction'],start,end,rel)
ax1 = plotseeing(met.time,av_10.seeing,norm['Wind_gust'],start,end,rel)
ax1 = plotseeing(met.time,av_10.seeing,norm['Wind_gust_dir'],start,end,rel)  
"""