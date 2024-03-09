import pandas as pd
import numpy as np
 
#GDP per capita, Healthy life expectancy,Freedom to make life choices,Generosity,Perceptions of corruption


def data_merged():
    data_2015 = pd.read_csv(r"Data\2015.csv").drop(columns = ['Country', 'Region', 'Happiness Rank', 'Standard Error', 'Family', 'Dystopia Residual'])
    # ['Country', 'Region', 'Happiness Rank', 'Happiness Score',
    #        'Standard Error', 'Economy (GDP per Capita)', 'Family',
    #        'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
    #        'Generosity', 'Dystopia Residual']
    data_2015.rename(columns = {'Happiness Score': 'Score', 'Health (Life Expectancy)': 'Healthy life expectancy', 'Freedom': 'Freedom to make life choices',
                                'Trust (Government Corruption)': 'Perceptions of corruption', 'Economy (GDP per Capita)': "GDP per capita"}, inplace=True)
    data_2016 = pd.read_csv(r"Data\2016.csv").drop(columns = ['Country', 'Region', 'Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval', 
                                                                         'Family', 'Dystopia Residual'])
    # ['Country', 'Region', 'Happiness Rank', 'Happiness Score',        
    #        'Lower Confidence Interval', 'Upper Confidence Interval',        
    #        'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
    #        'Freedom', 'Trust (Government Corruption)', 'Generosity',        
    #        'Dystopia Residual']
    data_2016.rename(columns = {'Happiness Score': 'Score', 'Health (Life Expectancy)': 'Healthy life expectancy', 'Freedom': 'Freedom to make life choices',
                                'Trust (Government Corruption)': 'Perceptions of corruption', 'Economy (GDP per Capita)': "GDP per capita"}, inplace=True)
    data_2017 = pd.read_csv(r"Data\2017.csv").drop(columns = ['Country', 'Happiness.Rank', 'Whisker.high', 'Whisker.low','Family', 'Dystopia.Residual'])
    # ['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high',
    #        'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
    #        'Health..Life.Expectancy.', 'Freedom', 'Generosity',
    #        'Trust..Government.Corruption.', 'Dystopia.Residual']
    data_2017.rename(columns = {'Happiness.Score': 'Score', 'Health..Life.Expectancy.': 'Healthy life expectancy', 'Freedom': 'Freedom to make life choices',
                                'Trust..Government.Corruption.': 'Perceptions of corruption', 'Economy..GDP.per.Capita.': "GDP per capita"}, inplace=True)
    data_2018 = pd.read_csv(r"Data\2018.csv").drop(columns = ['Overall rank', 'Country or region', 'Social support'])
    # ['Overall rank', 'Country or region', 'Score', 'GDP per capita',
    #        'Social support', 'Healthy life expectancy',
    #        'Freedom to make life choices', 'Generosity',
    #        'Perceptions of corruption']

    merge = pd.concat([data_2015, data_2016, data_2017, data_2018], ignore_index = True)
    

    return merge.dropna()
    
