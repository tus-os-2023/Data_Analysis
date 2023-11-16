from statistics import LinearRegression
import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.signal import detrend
from matplotlib.backends.backend_pdf import PdfPages

# 日本語フォント
mpl.rcParams['font.family'] = 'Meiryo'

path = "/Users/kayou0117/Desktop/データ分析/Compiled_2013-2023_eng.csv"

df = pd.read_csv(path)

# 型確認
print(df.shape) 

# データの型を確認
print(df.dtypes)

# 以下のカラムがすべて True かどうか確認
all_true_1 = df['SunshineDurationNoPhenomenonInformation'].all()

all_true_2 = df['MaximumSnowDepthNoPhenomenonInformation'].all()

all_true_3 = df['TotalPrecipitationNoPhenomenonInformation'].all()

all_true_4 = df['TotalSnowfallNoPhenomenonInformation'].all()

all_true_5 = df['LowestSeaLevelPressureNoPhenomenonInformation'].all()

all_true_6 = df['MaximumPrecipitationin10MinutesNoPhenomenonInformation'].all()

# 結果を出力
print(all_true_1, all_true_2, all_true_3, all_true_4, all_true_5, all_true_6)

# すべて True のカラムを削除
columns_to_drop = ['SunshineDurationNoPhenomenonInformation', 'MaximumSnowDepthNoPhenomenonInformation', 'TotalPrecipitationNoPhenomenonInformation', 'TotalSnowfallNoPhenomenonInformation', 'LowestSeaLevelPressureNoPhenomenonInformation', 'MaximumPrecipitationin10MinutesNoPhenomenonInformation']
df = df.drop(columns=columns_to_drop)

# CSVファイルにデータフレームを保存
df.to_csv('Re-Compiled_2013-2023_eng.csv', index=False)

# 各行に欠損値が含まれるかどうかを示すブール型のデータフレームを作成
rows_with_missing_values = df.isnull().any(axis=1)

# 欠損値が含まれる行を表示
rows_with_missing_values = df[rows_with_missing_values]
print("欠損値が含まれる行:")
print(rows_with_missing_values)

# 欠損値を確認すると、'SunshineDuration'の2023/9/9、2023/9/10の2行が欠損。

# 相関行列の計算
correlation_matrix = df.corr()

# ヒートマップの作成
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 欠損地を、Heat-Mapより相関の高い変数('AverageCloudCover', 'TotalSolarRadiation')を説明変数として重回帰分析を行い補完。

df = pd.read_csv('Re-Compiled_2013-2023_eng.csv')

# 必要な列だけを抽出
data = df[['SunshineDuration', 'AverageCloudCover', 'TotalSolarRadiation']]

# 欠損値が含まれる行を除外
data = data.dropna(subset=['SunshineDuration', 'AverageCloudCover', 'TotalSolarRadiation'])

# 訓練データとテストデータに分割
X = data[['AverageCloudCover', 'TotalSolarRadiation']]
y = data['SunshineDuration']

# モデルの構築
model = LinearRegression()
model.fit(X, y)

# 欠損値が含まれる行のインデックスを取得
rows_with_missing_values = df[df['SunshineDuration'].isnull()].index

# 欠損値を補完
for row_index in rows_with_missing_values:
    # 説明変数の値を取得
    input_data = df.loc[row_index, ['AverageCloudCover', 'TotalSolarRadiation']].values.reshape(1, -1)
    
    # SunshineDurationの欠損を補完
    df.at[row_index, 'SunshineDuration'] = model.predict(input_data)

# 上記で欠損値が補完されたことを確認

# 各行に欠損値が含まれるかどうかを示すブール型のデータフレームを作成
rows_with_missing_values = df.isnull().any(axis=1)

# 欠損値が含まれる行を表示
rows_with_missing_values = df[rows_with_missing_values]
print("欠損値が含まれる行:")
print(rows_with_missing_values)

# CSVファイルを保存
df.to_csv('Re-Imputed_2013-2023_eng.csv', index=False)

# データ補完の様子をplot

# 補完前のデータ
file_path_before = "Re-Compiled_2013-2023_eng.csv"
df_before = pd.read_csv(file_path_before)
df_before['Date'] = pd.to_datetime(df_before['Date'])

# 補完後のデータ
file_path_after = "Re-Imputed_2013-2023_eng.csv"
df_after = pd.read_csv(file_path_after)
df_after['Date'] = pd.to_datetime(df_after['Date'])

# 2023年9月のデータを抽出
df_before_sep = df_before[(df_before['Date'] >= '2023-09-01') & (df_before['Date'] <= '2023-09-30')]
df_after_sep = df_after[(df_after['Date'] >= '2023-09-01') & (df_after['Date'] <= '2023-09-30')]

# プロット
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df_before_sep['Date'], df_before_sep['SunshineDuration'])
plt.title('Sunshine Duration - Before Imputation')
plt.xlabel('Date')
plt.ylabel('Sunshine Duration')

plt.subplot(2, 1, 2)
plt.plot(df_after_sep['Date'], df_after_sep['SunshineDuration'], color='orange')
plt.title('Sunshine Duration - After Imputation')
plt.xlabel('Date')
plt.ylabel('Sunshine Duration')

plt.tight_layout()
plt.show()

# ['MostFrequentWindDirection', 'MaximumWindSpeedWindDirection', 'MaximumInstantaneousWindSpeedWindDirection']のデータの種類を確認。
unique_values = df['MostFrequentWindDirection'].unique()
print(unique_values)

unique_values = df['MaximumWindSpeedWindDirection'].unique()
print(unique_values)

unique_values = df['MaximumInstantaneousWindSpeedWindDirection'].unique()
print(unique_values)

# ]や）などの余計な記号が付いているデータがあったので以下で余計な記号を削除。

# 不要な文字を削除する関数
def remove_unwanted_chars(value):
    unwanted_chars = set([')', ']'])
    return ''.join(char for char in value if char not in unwanted_chars)

# カテゴリカル変数を持つカラムを指定
wind_direction_columns = ['MostFrequentWindDirection', 'MaximumWindSpeedWindDirection', 'MaximumInstantaneousWindSpeedWindDirection']

# 不要な文字を削除
df[wind_direction_columns] = df[wind_direction_columns].applymap(remove_unwanted_chars)

# 以下でカテゴリカル変数を数字に変換。

wind_directions = ['北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東', '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西']

# カテゴリカル変数を持つカラムを指定
wind_direction_columns = ['MostFrequentWindDirection', 'MaximumWindSpeedWindDirection', 'MaximumInstantaneousWindSpeedWindDirection']

# カテゴリを順序に対応する整数に変換
df[wind_direction_columns] = df[wind_direction_columns].apply(lambda col: col.map({direction: idx for idx, direction in enumerate(wind_directions)}))

# LabelEncoderの適用
label_encoder = LabelEncoder()
df[wind_direction_columns] = df[wind_direction_columns].apply(label_encoder.fit_transform)

# 上記で欠損値が発生していないかを以下で確認。

# 各行に欠損値が含まれるかどうかを示すブール型のデータフレームを作成
rows_with_missing_values = df.isnull().any(axis=1)

# 欠損値が含まれる行を表示
rows_with_missing_values = df[rows_with_missing_values]
print("欠損値が含まれる行:")
print(rows_with_missing_values)

# CSVファイルにデータフレームを保存
df.to_csv('ねこ.csv', index=False)

# unique_valuesに含まれるそれぞれの要素に対して行数を数える関数
def count_rows_by_value(column, value):
    count = len(df[df[column] == value])
    return count

# WeatherSummaryDay に対して処理
unique_values_WeatherSummaryDay = df['WeatherSummaryDay'].unique()

# 各要素の行数を辞書に格納
count_dict = {value: count_rows_by_value('WeatherSummaryDay', value) for value in unique_values_WeatherSummaryDay}

# 行数が多い順にソート
sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

# ファイルに書き出すためのパスを指定
output_file_path_all = 'count_by_weather_summaryDay_all.txt'
output_file_path_top10 = 'count_by_weather_summaryDay_top10.txt'

# すべての結果を書き出す
with open(output_file_path_all, 'w') as output_file_all:
    for value, count in sorted_counts:
        output_file_all.write(f'{value}: {count}件\n')

# 上位10件だけを書き出す
with open(output_file_path_top10, 'w') as output_file_top10:
    for value, count in sorted_counts[:10]:
        output_file_top10.write(f'{value}: {count}件\n')

print(f'結果が {output_file_path_all} および {output_file_path_top10} に書き出されました。')


# WeatherSummaryNight に対して処理
unique_values_WeatherSummaryNight = df['WeatherSummaryNight'].unique()

# 各要素の行数を辞書に格納
count_dict = {value: count_rows_by_value('WeatherSummaryNight', value) for value in unique_values_WeatherSummaryNight}

# 行数が多い順にソート
sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

# ファイルに書き出すためのパスを指定
output_file_path_all = 'count_by_weather_WeatherSummaryNight_all.txt'
output_file_path_top10 = 'count_by_weather_WeatherSummaryNight_top10.txt'

# すべての結果を書き出す
with open(output_file_path_all, 'w') as output_file_all:
    for value, count in sorted_counts:
        output_file_all.write(f'{value}: {count}件\n')

# 上位10件だけを書き出す
with open(output_file_path_top10, 'w') as output_file_top10:
    for value, count in sorted_counts[:10]:
        output_file_top10.write(f'{value}: {count}件\n')

print(f'結果が {output_file_path_all} および {output_file_path_top10} に書き出されました。')

########################################################################################################

# WeatherSummaryDay にはあるが、WeatherSummaryNight には存在しない値を取得
missing_in_day = set(unique_values_WeatherSummaryDay) - set(unique_values_WeatherSummaryNight)

# 結果を表示
print("存在するが、WeatherSummaryNight には存在しない値:", missing_in_day)
print(len(missing_in_day))


# WeatherSummaryDay にはあるが、WeatherSummaryNight には存在しない値を取得
missing_in_night = set(unique_values_WeatherSummaryNight) - set(unique_values_WeatherSummaryDay)

# 結果を表示
print("存在するが、WeatherSummaryNight には存在しない値:", missing_in_night)
print(len(missing_in_night))

########################################################################################################

# ねこ.csvを読み込む
df = pd.read_csv('ねこ.csv')

# 天気をカテゴリにまとめる関数    
def categorize_weather(weather):
    if '雪' in weather:
        return 3
    elif '雨' in weather:
        return 2
    elif '曇' in weather:
        return 1
    elif '晴' in weather or '快晴' in weather:
        return 0
    else:
        return -1

# 天気をカテゴリにまとめる
df['WeatherSummaryDay'] = df['WeatherSummaryDay'].apply(categorize_weather)

# カテゴリごとの数を集計
weather_day_counts = df['WeatherSummaryDay'].value_counts()

# 夜間の天気をカテゴリにまとめる
df['WeatherSummaryNight'] = df['WeatherSummaryNight'].apply(categorize_weather)

# カテゴリごとの数を集計
weather_night_counts = df['WeatherSummaryNight'].value_counts()

# 結果の表示
print(weather_day_counts)
print(weather_night_counts)

# 変更を保存
df.to_csv('にゃん.csv', index=False)

########################################################################################################

# 各カラム毎の可視化と季節性の除去

# CSVファイルからデータを読み込む
df = pd.read_csv('にゃん_with_category.csv')

# 日付をdatetime型に変換
df['Date'] = pd.to_datetime(df['Date'])

# データのトレンドを削除する関数
def detrend_column(column):
    return detrend(column.values)

# グラフごとに1ページずつ表示
num_columns = len(df.columns)
graphs_per_page = 4  # 1ページに表示するグラフの数
num_pages = (num_columns // graphs_per_page) + (num_columns % graphs_per_page > 0)

# PDFファイルの作成
with PdfPages('output.pdf') as pdf:
    for page in range(num_pages):
        start_idx = page * graphs_per_page
        end_idx = (page + 1) * graphs_per_page

        # サブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for i, column in enumerate(df.columns[start_idx:end_idx]):
            row = i // 2
            col = i % 2
            axes[row, col].plot(df['Date'], df[column], label=f'Original {column}')
            axes[row, col].plot(df['Date'], detrend_column(df[column]), label=f'Detrended {column}', linestyle='--')
            axes[row, col].set_title(f'{column} with and without Trend')
            axes[row, col].set_xlabel('Date')
            axes[row, col].set_ylabel('Value')
            axes[row, col].legend()

        # レイアウト調整
        plt.tight_layout()

        # PDFに保存
        pdf.savefig()

        # 表示を閉じる
        plt.close()



# 相関行列の計算
correlation_matrix = df.corr()

# ヒートマップの作成し直し
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
plt.title('Correlation Heatmap with Reduced Covariance')
plt.tight_layout()

# フォントサイズの変更
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=8)

plt.show()

# データの読み込み
df = pd.read_csv('にゃん.csv')

# 統計情報の取得
statistics_df = df.describe().transpose()

# 四分位範囲の計算
statistics_df['IQR'] = statistics_df['75%'] - statistics_df['25%']

# 画像のサイズを設定
plt.figure(figsize=(12, 8))

# ヒートマップの作成
sns.heatmap(statistics_df[['min', '25%', '50%', '75%', 'max', 'mean', 'std', 'IQR']], annot=True, cmap='Blues', fmt=".2f", linewidths=.5)

# 画像の保存
plt.savefig('statistics_summary.png')

# 画像の表示
plt.show()
