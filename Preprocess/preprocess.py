# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:08:57 2019

@author: anurag
"""
import pandas as pd
import numpy as np

#file containing all the data
MASTER_FILEPATH = "C:/Users/anurag/Desktop/Github_Repos/PredictingMiningInjuries/Data/Accidents.csv"
#columns which are required before splitting into sub tasks
MASTER_REQUIRED_COLUMNS = ("SUBUNIT	ACCIDENT_TIME DEGREE_INJURY	UG_LOCATION "
"UG_MINING_METHOD MINING_EQUIP SHIFT_BEGIN_TIME	CLASSIFICATION "
"ACCIDENT_TYPE	TOT_EXPER	MINE_EXPER	JOB_EXPER "
"OCCUPATION	ACTIVITY INJURY_SOURCE "
"NATURE_INJURY	INJ_BODY_PART DAYS_RESTRICT DAYS_LOST NARRATIVE COAL_METAL_IND").split()

#DATA for predicting DEGREE of INJURY
#columns REQUIRED to predict DEGREE OF INJURY using FIXED FIELDS
DOI_FF_REQUIRED_COLUMNS = ("SUBUNIT	ACCIDENT_TIME UG_LOCATION "
"UG_MINING_METHOD MINING_EQUIP SHIFT_BEGIN_TIME	CLASSIFICATION "
"ACCIDENT_TYPE	TOT_EXPER	MINE_EXPER	JOB_EXPER "
"OCCUPATION	ACTIVITY INJURY_SOURCE "
"NATURE_INJURY	INJ_BODY_PART  COAL_METAL_IND DEGREE_INJURY").split()
DOI_FF_REQUIRED_STRING_COLUMNS = ("UG_LOCATION "
"UG_MINING_METHOD MINING_EQUIP CLASSIFICATION "
"ACCIDENT_TYPE "
"OCCUPATION	ACTIVITY INJURY_SOURCE "
"NATURE_INJURY	INJ_BODY_PART  COAL_METAL_IND DEGREE_INJURY").split()
#columns REQUIRED to predict DEGREE OF INJURY using NARRATIVES
DOI_NAR_REQUIRED_COLUMNS = ("NARRATIVE DEGREE_INJURY").split()
#Target variable
TARGET_DOI = "DEGREE_INJURY"
#Taget classes that are not required
TARGET_DOI_NOT_REQUIRED = "".split()

#DATA for predicting DAYS AWAY FROM WORK
#columns REQUIRED to predict DAYS AWAY FROM WORK using FIXED FIELDS
DAW_FF_REQUIRED_COLUMNS = ("SUBUNIT	ACCIDENT_TIME DEGREE_INJURY	UG_LOCATION "
"UG_MINING_METHOD MINING_EQUIP SHIFT_BEGIN_TIME	CLASSIFICATION "
"ACCIDENT_TYPE	TOT_EXPER	MINE_EXPER	JOB_EXPER "
"OCCUPATION	ACTIVITY INJURY_SOURCE "
"NATURE_INJURY	INJ_BODY_PART  DAYS_LOST COAL_METAL_IND").split()
DAW_FF_REQUIRED_STRING_COLUMNS = ("SUBUNIT DEGREE_INJURY	UG_LOCATION "
"UG_MINING_METHOD MINING_EQUIP 	CLASSIFICATION "
"ACCIDENT_TYPE "
"OCCUPATION	ACTIVITY INJURY_SOURCE "
"NATURE_INJURY	INJ_BODY_PART COAL_METAL_IND").split()
#columns REQUIRED to predict DAYS AWAY FROM WORK using FIXED FIELDS
DAW_NAR_REQUIRED_COLUMNS = ("NARRATIVE DAYS_LOST").split()
DAW_NAR_REQUIRED_STRING_COLUMNS = ("NARRATIVE").split()
#Target variable
TARGET_DAW = "DAYS_LOST"
#VALUE of DOI
DOI_VALUES = "DAYS AWAY FROM WORK ONLY".split()


def import_data(filename):
    with open(filename, encoding="utf-8-sig") as f:
        data = pd.read_csv(f)
        f.close()
    return data

def remove_columns_with_null(data, columns):
    data.dropna()
    data=data.dropna(subset=columns)
    return data

def remove_rows_for_certainValues_column(data, column, values):
    for val in values:
        data = data[data[column] != val]
    return data

def keep_rows_with_col_values(df, column_name, some_values):
    df = df.loc[df[column_name].isin(some_values)]
    return df
    
#Preparing data for predicting DEGREE OF INJURY
def prepare_data_target_degree_injury(dataframe):
    dataframe_FF = dataframe[DOI_FF_REQUIRED_COLUMNS]
    dataframe_NAR = dataframe[DOI_NAR_REQUIRED_COLUMNS]
    return dataframe_FF, dataframe_NAR

#Preparing data for predicting DAYS AWAY FROM WORK
def prepare_data_target_days_away_from_work(dataframe):
    dataframe_FF = dataframe[DAW_FF_REQUIRED_COLUMNS]
    dataframe_NAR = dataframe[DAW_NAR_REQUIRED_COLUMNS]
    return dataframe_FF, dataframe_NAR

#read the file into a dataframe
master_dataframe = import_data(MASTER_FILEPATH)
#retaining only required columns
master_dataframe = master_dataframe[MASTER_REQUIRED_COLUMNS]

#TASK 1
#preparing data for predicting DEGREE OF INJURY
dataframe_DOI_FF, dataframe_DOI_NAR = prepare_data_target_degree_injury(master_dataframe)
#removing rows with null values in columns
dataframe_DOI_FF = remove_columns_with_null(dataframe_DOI_FF, DOI_FF_REQUIRED_COLUMNS)
dataframe_DOI_NAR = remove_columns_with_null(dataframe_DOI_NAR, DOI_NAR_REQUIRED_COLUMNS)
#removing rows with specific values in certain columns
specific_values = ["NO VALUE"]
for col in DOI_FF_REQUIRED_STRING_COLUMNS:
    dataframe_DOI_FF = remove_rows_for_certainValues_column(dataframe_DOI_FF, 
                                                            col, specific_values)
for col in DOI_NAR_REQUIRED_COLUMNS:
    dataframe_DOI_NAR = remove_rows_for_certainValues_column(dataframe_DOI_NAR, 
                                                            col, specific_values)
#for balancing data
dataframe_DOI_FF = remove_rows_for_certainValues_column(dataframe_DOI_FF, 
                                                            col, TARGET_DOI_NOT_REQUIRED)
dataframe_DOI_NAR = remove_rows_for_certainValues_column(dataframe_DOI_NAR, 
                                                            col, TARGET_DOI_NOT_REQUIRED)

#TASK 2
#preparing data for predicting DAYS AWAY FROM WORK
dataframe_DAW_FF, dataframe_DAW_NAR = prepare_data_target_days_away_from_work(master_dataframe)
#removing rows with null values in columns
dataframe_DAW_FF = remove_columns_with_null(dataframe_DAW_FF, DAW_FF_REQUIRED_COLUMNS)
dataframe_DAW_NAR = remove_columns_with_null(dataframe_DAW_NAR, DAW_NAR_REQUIRED_COLUMNS)

specific_values = ["NO VALUE"]
for col in DAW_FF_REQUIRED_STRING_COLUMNS:
    dataframe_DAW_FF = remove_rows_for_certainValues_column(dataframe_DAW_FF, 
                                                            col, specific_values)
for col in DAW_NAR_REQUIRED_STRING_COLUMNS:
    dataframe_DAW_NAR = remove_rows_for_certainValues_column(dataframe_DAW_NAR, 
                                                            col, specific_values)
    
#retaining only rows with specific DOI_VALUE
dataframe_DAW_FF = keep_rows_with_col_values(dataframe_DAW_FF, TARGET_DOI, DOI_VALUES)



    

