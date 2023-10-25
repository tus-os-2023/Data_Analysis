# 2023-10-25

## 今日やったこと
- csvファイルをPythonで読み込めるようにShift-JIS形式からutf-8形式に変換．ラベル等を整理し，2013/10/20~2023/10/19までのすべてのデータを一つにまとめるPythonコードを実行．

## 次回やること
- float64形式，int64形式のデータをBool形式に直す

## その他
Complified_2013-2023.csvのすべてのcolumnのdtypeを調べると以下のようになった．
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   年月日                        3652 non-null   object 
 1   日照時間(時間)                   3650 non-null   float64
 2   日照時間(時間).1 / 現象なし情報        3650 non-null   float64
 3   最深積雪(cm)                   3652 non-null   float64
 4   最深積雪(cm).1 / 現象なし情報        3652 non-null   int64  
 5   平均風速(m/s)                  3652 non-null   float64
 6   平均蒸気圧(hPa)                 3652 non-null   float64
 7   平均湿度(％)                    3652 non-null   float64
 8   平均海面気圧(hPa)                3652 non-null   float64
 9   平均現地気圧(hPa)                3652 non-null   float64
 10  平均雲量(10分比)                 3652 non-null   float64
 11  平均気温(℃)                    3652 non-null   float64
 12  合計全天日射量(MJ/㎡)              3652 non-null   float64
 13  降水量の合計(mm)                 3652 non-null   float64
 14  降水量の合計(mm).1 / 現象なし情報      3652 non-null   int64  
 15  降雪量合計(cm)                  3652 non-null   float64
 16  降雪量合計(cm).1 / 現象なし情報       3652 non-null   int64  
 17  最高気温(℃)                    3652 non-null   float64
 18  最低気温(℃)                    3652 non-null   float64
 19  最多風向(16方位)                 3652 non-null   object 
 20  最大風速(m/s)                  3652 non-null   float64
 21  最大風速(m/s).4 / 風向           3652 non-null   object 
 22  最低海面気圧(hPa)                3652 non-null   float64
 23  最低海面気圧(hPa).1 / 現象なし情報     3652 non-null   int64  
 24  最小相対湿度(％)                  3652 non-null   float64
 25  10分間降水量の最大(mm)             3652 non-null   float64
 26  10分間降水量の最大(mm).1 / 現象なし情報  3652 non-null   int64  
 27  最大瞬間風速(m/s)                3652 non-null   float64
 28  最大瞬間風速(m/s).4 / 風向         3652 non-null   object 
 29  天気概況(昼：06時～18時)            3652 non-null   object 
 30  天気概況(夜：18時～翌日06時)          3652 non-null   object 
