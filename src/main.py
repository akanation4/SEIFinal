# ライブラリの読み込み
from math import dist
from matplotlib import type1font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from jeraconv import jeraconv

j2w = jeraconv.J2W()

# データの読み込み・加工
df = pd.read_csv("../real_estate/chitose.csv", encoding="cp932")
df.columns = ["no", "type", "region", "city_code", "prefecture_name", "city_name",
"district_name", "nearest_station_name", "nearest_station_distance",
"transaction_price","unit_price_per_tsubo","floor_area","area",
"transaction_price(unit_price_per_square_meter)","land_shape","frontage",
"total_floor_area","year_of_construction","structure_of_building","usage",
"purpose_of_future_use","front_road_direction","front_road_type",
"front_road_width","city_planning","building_coverage_ratio",
"floor_area_ratio","time_of_transaction","refurbishment",
"circumstances_of_transaction"]
# 数値に変換できない値を変換
df.loc[df['area']=='5000㎡以上', ['area']] = '5000'
df.loc[df['area']=='2000㎡以上', ['area']] = '2000'
df.loc[df['nearest_station_distance']=='30分?60分', ['nearest_station_distance']] = '45'
df.loc[df['nearest_station_distance']=='1H30?2H', ['nearest_station_distance']] = '105'
df.loc[df['nearest_station_distance']=='1H?1H30', ['nearest_station_distance']] = '75'
df.loc[df['nearest_station_distance']=='2H?', ['nearest_station_distance']] = '120'
df.loc[df['year_of_construction']=='戦前', ['year_of_construction']] = '昭和20年'
# 型の指定
df["area"] = df["area"].astype(int)
# 計算によって補完
df.loc[df["transaction_price(unit_price_per_square_meter)"].isnull(),
        ['transaction_price(unit_price_per_square_meter)']] = df['transaction_price'] / df['area']
# ワンホットエンコーディング
df_district = pd.get_dummies(df['district_name'], prefix='district_name')
df_type = pd.get_dummies(df['type'], prefix='type')
df_structure = pd.get_dummies(df['structure_of_building'], prefix='structure_of_building')
df_direction = pd.get_dummies(df['front_road_direction'], prefix='front_road_direction')

# print(df["type"].value_counts())
# print(df["structure_of_building"].value_counts())
# print(df["front_road_direction"].value_counts())

# データ結合
df = pd.concat([df["area"], df["nearest_station_distance"],
                df_district, df_structure, df_direction,
                df['year_of_construction'],
                df["transaction_price"]], axis=1)
# 欠損値の処理
df.dropna(subset=['year_of_construction'], inplace=True)
array = []
for item in df['year_of_construction']:
    array.append(j2w.convert(item))
df_year = pd.DataFrame(array, columns=['year'])
df.drop('year_of_construction', axis=1, inplace=True)
df = pd.concat([df_year, df], axis=1)
df.dropna(subset=['nearest_station_distance'], inplace=True)
df["nearest_station_distance"] = df["nearest_station_distance"].astype(int)
df.dropna(subset=['year'], inplace=True)
# pd.set_option('display.max_rows', None)
# print(df.isnull().sum())
# print(df['year'])

# 機械学習用のデータを作成
arr = df.values
X = arr[:,:-1]
y = arr[:,-1]
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 機械学習
# # Ridge Model
# ridge = Ridge(alpha=0.001)
# ridge.fit(X_train, y_train)

# y_train_pred = ridge.predict(X_train)

# y_test_pred = ridge.predict(X_test)

# print('RMSE 学習: %.2f, テスト: %.2f' % (
#     mean_squared_error(y_train, y_train_pred, squared=False),
#     mean_squared_error(y_test, y_test_pred, squared=False)
#     ))
# print('R2 学習: %.2f, テスト: %.2f' % (
#     r2_score(y_train, y_train_pred),
#     r2_score(y_test, y_test_pred)
#     ))

# # Lasso
# lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
# print(f"学習: {lasso.score(X_train, y_train):.2}")
# print(f"テスト: {lasso.score(X_test, y_test):.2}")
# print(np.sum(lasso.coef_ != 0))

# ニューラルネットワーク
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=10000)
mlp.fit(X_train, y_train)

pred = mlp.predict(X_test)
for i in range(len(X_test)):
    c = y_test[i]
    p = pred[i]
    print('[{0}] correct: {1:.3f}, predict: {2:.3f} ({3:.3f})'.format(i, c, p, c-p))

print('R^2 = {0}'.format(mlp.score(X_test, y_test)))

year = 2020
area = 313
nearest_station_distance = 9
district_name = '花園'
structure_of_building = '木造'
front_road_direction = '南'

test = pd.DataFrame({'year':[year], 'area':[area], 'nearest_station_distance':[nearest_station_distance], 'district_name':[district_name], 'structure_of_building':[structure_of_building], 'front_road_direction':[front_road_direction]})
test['district_name'] = pd.Categorical(test['district_name'],
categories=['あずさ', 'みどり台北', 'みどり台南', '上長都', '中央', '住吉',
'信濃', '勇舞', '北信濃', '北光', '北斗', '北栄', '北陽', '千代田町', '協和',
'大和', '富丘', '富士', '寿', '幌加', '平和', '幸町', '幸福', '弥生', '文京', '新富',
'新川', '新星', '日の出', '旭ケ丘', '春日町', '朝日町', '末広', '本町', '東丘',
'東郊', '東雲町', '柏台', '柏台南', '柏陽', '栄町', '根志越', '桂木', '桜木',
'梅ケ丘', '泉沢', '泉郷', '流通', '清水町', '清流', '白樺', '真々地', '祝梅',
'福住', '稲穂', '緑町', '美々', '自由ケ丘', '花園', '若草', '蘭越', '豊里',
'都', '里美', '釜加', '錦町', '長都', '長都駅前', '青葉', '青葉丘', '駒里', '高台'])
test['structure_of_building'] = pd.Categorical(test['structure_of_building'],
categories=['ブロック造', '木造', '軽量鉄骨造', '鉄骨造', '鉄骨造、木造',
'ＲＣ', 'ＲＣ、木造', 'ＳＲＣ'])
test['front_road_direction'] = pd.Categorical(test['front_road_direction'],
categories=['北', '北東', '北西', '南', '南東', '南西', '接面道路無', '東', '西'])
test = pd.get_dummies(test)
mlp.predict(test)
