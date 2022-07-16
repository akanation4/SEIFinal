# ライブラリの読み込み
from tkinter import Y
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
# 型の指定
df["area"] = df["area"].astype(int)
# 計算によって補完
df.loc[df["transaction_price(unit_price_per_square_meter)"].isnull(),
        ['transaction_price(unit_price_per_square_meter)']] = df['transaction_price'] / df['area']
# ワンホットエンコーディング
df_district = pd.get_dummies(df['district_name'], prefix='district_name')
df_type = pd.get_dummies(df['type'], prefix='type')
# 欠損値の処理
df.dropna(subset=['nearest_station_distance'], inplace=True)

# 機械学習用のデータを作成
df = pd.concat([df["area"], df["nearest_station_distance"],
                df_district, df_type, df["transaction_price"]], axis=1)
arr = df.values
X = arr[:,:-1]
Y = arr[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 機械学習
clf = MLPClassifier()
