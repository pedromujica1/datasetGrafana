# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:26:08 2025

@author: Maruzka
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Versão alternativa mais robusta
def clean_invalid_values(df: pd.DataFrame, 
                               tags: list,
                               min_val: float, 
                               max_val: float,
                               inplace: bool = False,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Versão mais robusta da função de limpeza com parâmetros configuráveis.
    
    Args:
        df: DataFrame original
        voltage_tags: Lista de tags para identificar colunas de voltagem
        min_val: Valor mínimo válido
        max_val: Valor máximo válido
        
    Returns:
        DataFrame com valores inválidos substituídos por NaN
    """
    
    __print = print if verbose else lambda *args, **kwargs: None  # Funciona com múltiplos args
   
    if not inplace:
        df = df.copy()
        
    # Encontra colunas de voltagem
    voltage_columns = [col for col in df.columns 
                      if any(tag in col.lower() for tag in tags)]
    
    if not voltage_columns and not inplace:
        __print("None tags in column")
        return df
    
    __print(f"Columns to be cleaned: {voltage_columns}")
    
    total_invalid = 0
    
    for col in voltage_columns:
        if not pd.api.types.is_numeric_dtype(df[col]): # chaning only numeric columns
            __print(f"Aviso: Coluna '{col}' não é numérica, pulando...")
            continue
        
        invalid_mask = (df[col] > max_val) | (df[col] < min_val)
        col_invalid_count = invalid_mask.sum()
        
        if col_invalid_count > 0:
            df.loc[invalid_mask, col] = np.nan
            total_invalid += col_invalid_count
            __print(f"'{col}': {col_invalid_count} valores inválidos removidos")
    
    __print(f"Total de valores inválidos removidos: {total_invalid}")
    
    if not inplace:
        return df
    
    return True

# def clean_invalid_values(df: pd.DataFrame):
    
#     voltage_data = ["we", "ae"]
#     column_voltage_data = [col for col in df.columns if any(tag in col for tag in voltage_data)]
#     print(column_voltage_data)
    
#     data = df[column_voltage_data]
#     idx = (data > 6.14) | (data < 0.05)
#     print(idx.sum())
#     print(idx)
#     data.loc[idx] = np.nan
            
    # data = aqm[p + 'temp']
    # idx = (data > 50) | (data <= 1)
    # data.loc[idx] = np.nan

def drop_columns(columns: List[str], df: pd.DataFrame) -> pd.DataFrame:

    _df = df.copy(deep=True)
    
    # Encontra todas as colunas que contêm qualquer uma das strings
    columns_to_drop = [
        col for col in _df.columns 
        if any(tag in col for tag in columns)
    ]
    
    # Mostra quais colunas serão removidas
    if columns_to_drop:
        print(f"Deleting columns: {columns_to_drop}")
        _df = _df.drop(columns=columns_to_drop)
    else:
        print("No columns found to delete")
    
    # Mostra tags que não foram encontradas
    for tag in columns:
        if not any(tag in col for col in df.columns):
            print(f"Tag '{tag}' not found in any column")
    
    return _df

if __name__ == "__main__":
    
    aqm = pd.read_csv('envcity_aqm_df.csv')
    
    aqm = drop_columns(["header", "hardware", "sigma", "rx_time", "payload", "packet_id", 
                            "duplicate", "anem", "lora", "gateway","freq", 
                            "port", "_id", "modulation", "rssi", "size","rainfalltotal", 
                            "iag_solarradiation", "iag_wd", "iag_ws", "field",
                            "hct", "ch4", "snr", "location", "gps", "counter", 
                            "datarate", "delay", "pm", "e2_", "e1_", "datarate",
                            "anem", "e1_anem", "pressure"], aqm)
    
    print("limpando dados de tensão inválidos")
    clean_invalid_values(aqm, tags=["we", "ae"], min_val=0.05, max_val=6.1, inplace=True)
        
    print("limpando dados menores que 0")
    clean_invalid_values(aqm, tags=["e2sp_"], min_val=0, max_val=np.inf, inplace=True)
        
    aqm.set_index('time', inplace=True)
    aqm.index = pd.to_datetime(aqm.index)
    
    print(aqm.describe())
    print(aqm.shape)
    
    aqm.to_csv('envcity_df_sp_dataset_2023.csv', index=True)
         

