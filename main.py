from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import regex as re
from fastapi.encoders import jsonable_encoder

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    items: List[Item]

#Функции которые надо применить до pipline (Pre_pipline) далее:

#Функция чтобы обойти в трейне ячейку, где нет числа, есть только единица измерения, и в принципе не падать на таких кейсах
def check_reg(value):
    pattern_int = ('\d+')
    pattern_float = ('\d+\.\d+')
    match = re.search(pattern_int, value)
    if match:
        return value
    else:
        match = re.search(pattern_float, value)
        if match:
            return value
        else:
            return np.nan

#Функция чтобы из ячейки оставить только число, убрав единицу измерения
def del_ed_izmeren(value):
    type_value = type(value)
    if type_value == str:
        list_val = value.split(' ')
        rez = check_reg(list_val[0])
        return float(rez)

#Функция чтобы отрезать целевую перменную и переменную torque
def cut_cat(df):
    return df.drop(['torque', 'selling_price'], axis=1)

#Функция чтобы отрезать имя, зачем то мне надо было чтобы она была отдельно от предыдущей, когда я писал в ноутбуке
def cut_name(df):
    return df.drop('name', axis=1)

#Функция чтобы удалить единицы измерения в столбцах mileage, engine', max_power
def cut_izmeren_column(df):
    for column in ['mileage', 'engine', 'max_power']:
        df[column] = df[column].apply(del_ed_izmeren)
    return df

#Функция чтобы найти столбцы с пропусками
def cnt_loss(df):
    dictt = dict()
    for column in df.columns:
        count = df[column].apply(pd.isnull).sum()
        if count != 0:
            dictt[column] = count
    return dictt

#Функция чтобы заполнить столбцы с пропусками
def put_median(df):
    loss_dict = cnt_loss(df)
    for column in loss_dict.keys():
        median = df[column].median()
        df[column] = df[column].fillna(median)
    return df

#Функция чтобы привести к ИНТ стоблцы engine, seats
def change_to_int(df):
    for column in ['engine', 'seats']:
        df[column] = df[column].astype(int)
    return df

#Функция чтобы применить Pipline
def do_pipline(df):
    pipline_pickle = pickle.load(open('pipline.pkl', 'rb'))
    res_pickle = pipline_pickle.fit_transform(df)
    res_df_pickle = pd.DataFrame(res_pickle, columns=pipline_pickle.get_feature_names_out())
    return res_df_pickle

#Функция чтобы cобрать все другие воедино ~ One Ring to rule them all, One Ring to find them,
# One Ring to bring them all and in the darkness bind them In the Land of Mordor where the Shadows lie =)

# А еще она конкатенирует с трейном полученные данные, чтобы вложить правильные медианы и после OneHotE не получились разные столбцы
# И потом разконкатенирует обратно и отдает ДФ без строк трейна
def concat_and_do_pre_pipline_and_piplene_and_re_concat(df):
    cnt_str_our_df = df.shape[0]
    racshirenniy_df = pd.concat([df, train_df], axis=0, ignore_index=True)
    racshirenniy_df = cut_cat(racshirenniy_df)
    racshirenniy_df = cut_izmeren_column(racshirenniy_df)
    racshirenniy_df = cut_name(racshirenniy_df)
    racshirenniy_df = put_median(racshirenniy_df)
    racshirenniy_df = change_to_int(racshirenniy_df)
    racshirenniy_df = do_pipline(racshirenniy_df)
    return racshirenniy_df.head(cnt_str_our_df)

#-----------------------------------------------------------------------------
train_df = pd.read_csv('cars_train.csv')
pickled_model = pickle.load(open('model_new.pkl', 'rb'))

# Функции для эндпоинтов
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_item = pd.DataFrame(jsonable_encoder(item), index=[0])
    df_item = concat_and_do_pre_pipline_and_piplene_and_re_concat(df_item)
    y_pred = pickled_model.predict(df_item)[0]
    return y_pred

# В чатике писали, что можно трактовать неточности в свою сторону, потому в этот эндпоинт для массового прогноза надо давать на вход массив JSON-объектов, подходящих под класс Item, а на выход получится массив предсказаных цен
@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    items_index = list(range(len(items)))
    df_items = pd.DataFrame(jsonable_encoder(items), index=[items_index])
    df_items = concat_and_do_pre_pipline_and_piplene_and_re_concat(df_items)
    y_predS = pickled_model.predict(df_items)
    return y_predS
