# fer (Facial Expression Recognition )

## Overview 
画像から表情を機械学習を用いて予測するプログラム（正答率は約30%)
0="angry",1="disgust",2="fear",3="happy",4="sad",5="surprise",6="neutral"
## Requirement
- Python3

## Usage
Kaggle(FER-2013)
[リンク先](https://datarepository.wolframcloud.com/resources/FER-2013)
のファイルをダウンロードして、csvファイルから画像ファイルに変換後、モデル構築、学習、予想させる。
画像変換はgene\_record.py、画像読み込みはread\_data.pyから行うことができる
