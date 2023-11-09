import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def process_time_series(data, column_name, pdf):
    # 列に欠損値が含まれているかどうかをチェック
    if data[column_name].isnull().any():
        # 欠損値がある場合にエラーメッセージを出力
        print(f"Error: The column '{column_name}' contains missing values.")
        return

    # トレンド，季節性をプロット(seasonal_decomposeを使用)
    decomposition = seasonal_decompose(
        data[column_name], model='additive', period=365
        )  # 季節性、トレンド、残差成分にデータを分解
    fig = decomposition.plot()
    fig.set_size_inches(12, 6)
    pdf.savefig(fig)
    plt.close()

    # トレンド，季節性をプロット(STLを使用, 周期=365日)
    # 季節性、トレンド、残差成分にデータを分解
    stl = STL(data[column_name], period=365, robust=True)
    result = stl.fit()
    fig = result.plot()
    plt.gcf().set_size_inches(12, 6)
    pdf.savefig(fig)
    plt.close()

    # ACF，PACFのプロット
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_acf(data[column_name], lags=50, ax=ax1)  # 自己相関をプロット
    plot_pacf(data[column_name], lags=50, ax=ax2)  # 偏自己相関をプロット
    fig.set_size_inches(12, 6)
    pdf.savefig(fig)
    plt.close()


# メイン関数
if __name__ == "__main__":
    weather_parameters = [
        'SunshineDuration', 'AverageWindSpeed', 'AverageHumidity',
        'AverageSeaLevelPressure', 'AverageTemperature', 'TotalSolarRadiation',
        'TotalPrecipitation', 'MaximumTemperature', 'MinimumTemperature',
        'MaximumWindSpeed', 'MaximumInstantaneousWindSpeed'
    ]

    # データの読み込みと日付を変換
    data = pd.read_csv(
        './data/Tokyo_Weather_2013-2023/Compiled_2013-2023_eng.csv'
        )
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # すべての気象パラメータに対してグラフを作りPDFに変換
    with PdfPages("./outputs/notebook_analysis/all_plots.pdf") as pdf:
        for parameter in weather_parameters:
            process_time_series(data, parameter, pdf)
