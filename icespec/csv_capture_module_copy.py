# rumz // 9/3/24
# I don't think this should appear in the final product

import pandas as pd
import os

def getData(CoreID: str, Section: str, Light: str) -> str:
    #this function takes user input for desired core data, searches the csv, and returns proper path

    csv_path = os.path.abspath(os.path.join('..', 'processing', 'HSI_July2024.csv')) #set path to processing folder
    DataFrame = pd.read_csv(csv_path) #reading in file with core data

    Duplicates(DataFrame, Section) #call function to handle potential duplicates and avoid errors


    # conditionals to parse light settings
    if Light == "Top":
        light = '100'
    elif Light == "Bottom":
        light = 'Dot 4'
    else:
        print ("Light input invalid. / csv_capture - Line: 25")

    
    data = DataFrame[(DataFrame['CoreID'] == CoreID) & (DataFrame['Tube/Section'] == Section) & (DataFrame['Top Light'] == light)]
    df3 = data[["Top Light", "Bottom Light", "Exposure", 'DatasetName']]

    # conditionals to grab correct image based on light settings
    if light == 'Dot 4':
        dataName = getExp(df3)
    elif light == '100':
        dataName = df3.iloc[(0,3)]

    print(dataName)
    
    return dataName
        

    


def dataManage():
    # not complete

    #how many Top Lights are 100 vs Dot 4
    #ask input
    print("Error: There are duplicate images, cannot parse")
    return 0

def getExp(df3: pd.DataFrame) -> str:
    # function to handle exposure settings

    exposure = input("High or Low Exposure? (Type [h] or [l])")
    
    if exposure == 'h':
        dataName = hiEx(df3)
    elif exposure == 'l':
        dataName = loEx(df3)
    else:
        getExp(df3)

    return dataName

def hiEx(df3: pd.DataFrame) -> str:
    # grabs high exposure image

    if df3.iloc[(0,2)] > df3.iloc[(1,2)]:
        dataName = df3.iloc[(0,3)]
    elif df3.iloc[(0,2)] < df3.iloc[(1,2)]:
        dataName = df3.iloc[(1,3)]
    else:
        print("error")

    return dataName

def loEx(df3: pd.DataFrame) -> str:
    # grabs lower exposure image

    if df3.iloc[(0,2)] < df3.iloc[(1,2)]:
        dataName = df3.iloc[(0,3)]
    elif df3.iloc[(0,2)] > df3.iloc[(1,2)]:
        dataName = df3.iloc[(1,3)]
    else:
        print("error")

    return dataName

def Duplicates(DataFrame: pd.DataFrame, Section: str):
    # handles duplicates

    df2 = DataFrame.pivot_table(index='Tube/Section', aggfunc='size')
    #print("Get count of duplicate values in a single column:\n", df2)
    num = df2[Section]
    #print (num)
    if num > 3:
        dataManage()  #not finished