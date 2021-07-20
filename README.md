# fer (Facial Expression Recognition )

## Overview 
画像から表情を機械学習(CNN)を用いて予測するプログラム（正答率は約53%)
0="angry",1="disgust",2="fear",3="happy",4="sad",5="surprise",6="neutral"
## Requirement
- Python3

## Usage
Kaggle(FER-2013)
[リンク先](https://datarepository.wolframcloud.com/resources/FER-2013)
のファイルをダウンロードして、csvファイルから画像ファイルに変換後、モデル構築、学習、予想させる。
画像変換はgene\_record.py、画像読み込みはread\_data.pyから行うことができる

## Features

# Model
![model_cnn_dropout](https://user-images.githubusercontent.com/44591782/108973212-70ebc080-76c7-11eb-90af-dcdb6c5018be.png)

# Accuracy
![face_cnn_dropout](https://user-images.githubusercontent.com/44591782/108973567-d50e8480-76c7-11eb-843b-669c03f006d0.png)

# heat-map
![mt](https://user-images.githubusercontent.com/44591782/108973620-e9eb1800-76c7-11eb-811f-b7e7fba093ed.png)

# 各層ごとの画像
![Fil1](https://user-images.githubusercontent.com/44591782/108973786-1dc63d80-76c8-11eb-8541-6253336b721b.png)
![Fil2](https://user-images.githubusercontent.com/44591782/108973813-23bc1e80-76c8-11eb-8f7c-d2f1b7b840f9.png)
![Fil3](https://user-images.githubusercontent.com/44591782/108973815-2454b500-76c8-11eb-9309-802f4655b874.png)
![Fil4](https://user-images.githubusercontent.com/44591782/108973816-24ed4b80-76c8-11eb-9ab4-fb84744ced49.png)
![Fil5](https://user-images.githubusercontent.com/44591782/108973818-2585e200-76c8-11eb-8f41-5a8d2f441138.png)

# 各層ごとのフィルター
![ker1](https://user-images.githubusercontent.com/44591782/108973950-4d754580-76c8-11eb-8f16-bf06c7338ee0.png)
![ker2](https://user-images.githubusercontent.com/44591782/108973960-4fd79f80-76c8-11eb-90d9-760f813c3a69.png)
![ker3](https://user-images.githubusercontent.com/44591782/108973973-51a16300-76c8-11eb-9036-775358e7e115.png)

#result
![Figure_5](https://user-images.githubusercontent.com/44591782/108974042-64b43300-76c8-11eb-8d4a-1357da23e0d1.png)

## Licence
MIT Licence