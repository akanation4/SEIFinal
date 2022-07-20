#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from fileinput import filename
from flask import Flask, jsonify, request
import pandas as pd
import pickle
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get():
    # JSON形式でデータ取得
    json = request.get_json()
    year = float(json.get('year'))
    area = float(json.get('area'))
    nearest_station_distance = float(json.get('nearest_station_distance'))
    district_name = json.get('district_name')
    structure_of_building = json.get('structure_of_building')
    front_road_direction = json.get('front_road_direction')

    # データフレームに変換
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

    # モデル読み込み
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    test_arr = test.values
    result = loaded_model.predict(test_arr)

    # JSON形式で返す
    return jsonify({'transaction_price':result[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
