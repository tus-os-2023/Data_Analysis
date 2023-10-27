import os
import pandas as pd

def convert_to_utf8(folder_path, current_year, years_back=10):
    for i in range(years_back):
        file_name = f"data{current_year-i-1}1020-{current_year-i}1019.csv"
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, file_name.replace(".csv", "_utf-8.csv"))
        
        if os.path.exists(old_path):
            with open(old_path, 'r', encoding='cp932') as old_file, \
                 open(new_path, 'w', encoding='utf-8') as new_file:
                new_file.write(old_file.read())
            print(f"Converted {old_path} -> {new_path}")
        else:
            print(f"File {old_path} does not exist!")

def preprocess_data(file_name):
    data = pd.read_csv(file_name, skiprows=3, encoding='utf-8')
    
    new_cols = []
    for col in data.columns:
        col_idx = data.columns.get_loc(col)
        new_col = col
        if pd.notna(data.iat[1, col_idx]):
            new_col += '/' + str(data.iat[1, col_idx])
        if pd.notna(data.iat[0, col_idx]):
            new_col += '/' + str(data.iat[0, col_idx])
        new_cols.append(new_col)
    
    data.columns = new_cols
    
    data = data.drop(columns=[col for col in data.columns if "品質情報" in str(data[col].iat[1]) or 
                                                         "均質番号" in str(data[col].iat[1]) or
                                                         "時分" in str(data[col].iat[0])])
    return data.drop([0, 1])

def translate_column_names(data):
    original_columns = [
        "年月日", "日照時間(時間)", "日照時間(時間).1/現象なし情報", "最深積雪(cm)", 
        "最深積雪(cm).1/現象なし情報", "平均風速(m/s)", "平均蒸気圧(hPa)", "平均湿度(％)",
        "平均海面気圧(hPa)", "平均現地気圧(hPa)", "平均雲量(10分比)", "平均気温(℃)", 
        "合計全天日射量(MJ/㎡)", "降水量の合計(mm)", "降水量の合計(mm).1/現象なし情報", 
        "降雪量合計(cm)", "降雪量合計(cm).1/現象なし情報", "最高気温(℃)", "最低気温(℃)", 
        "最多風向(16方位)", "最大風速(m/s)", "最大風速(m/s).4/風向", "最低海面気圧(hPa)", 
        "最低海面気圧(hPa).1/現象なし情報", "最小相対湿度(％)", "10分間降水量の最大(mm)", 
        "10分間降水量の最大(mm).1/現象なし情報", "最大瞬間風速(m/s)", "最大瞬間風速(m/s).4/風向", 
        "天気概況(昼：06時～18時)", "天気概況(夜：18時～翌日06時)"
    ]

    translated_columns = [
    "Date", "SunshineDuration", "SunshineDuration/NoPhenomenonInformation", 
    "MaximumSnowDepth", "MaximumSnowDepth/NoPhenomenonInformation", 
    "AverageWindSpeed", "AverageVaporPressure", "AverageHumidity", 
    "AverageSeaLevelPressure", "AverageGroundLevelPressure", "AverageCloudCover", 
    "AverageTemperature", "TotalSolarRadiation", "TotalPrecipitation", 
    "TotalPrecipitation/NoPhenomenonInformation", "TotalSnowfall", 
    "TotalSnowfall/NoPhenomenonInformation", "MaximumTemperature", "MinimumTemperature", 
    "MostFrequentWindDirection", "MaximumWindSpeed", "MaximumWindSpeed/WindDirection", 
    "LowestSeaLevelPressure", "LowestSeaLevelPressure/NoPhenomenonInformation", 
    "MinimumRelativeHumidity", "MaximumPrecipitationin10Minutes", 
    "MaximumPrecipitationin10Minutes/NoPhenomenonInformation", 
    "MaximumInstantaneousWindSpeed", "MaximumInstantaneousWindSpeed/WindDirection", 
    "WeatherSummaryDay", "WeatherSummaryNight"
]

    data.columns = [translated_columns[original_columns.index(col)] if col in original_columns else col for col in data.columns]
    return data

def aggregate_csv_files(folder_path, start_year, num_years):
    aggregated_data = []

    for i in range(num_years):
        file_name = f"data{start_year+i}1020-{start_year+i+1}1019_utf-8.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            data = preprocess_data(file_path)
            
            target_cols = [col for col in data.columns if "現象なし情報" in col]
            for col in target_cols:
                data[col] = data[col].astype(bool)
            
            
            data = translate_column_names(data)
            aggregated_data.append(data)
        else:
            print(f"File {file_name} does not exist!")
            
    result = pd.concat(aggregated_data, ignore_index=True)
    result.to_csv(os.path.join(folder_path, f'Compiled_{start_year}-{start_year+num_years}_eng.csv'), index=False, encoding='utf-8')
    print(f"Data aggregated and saved as Compiled_{start_year}-{start_year+num_years}_eng.csv")

if __name__ == "__main__":
    folder_path = './'
    current_year = 2023
    
    convert_to_utf8(folder_path, current_year)
    aggregate_csv_files(folder_path, 2013, 10) 
