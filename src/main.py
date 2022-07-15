import pandas as pd

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
print(df.head())

vc = df['area'].value_counts()
pd.set_option('display.max_rows', None)
print(vc)
