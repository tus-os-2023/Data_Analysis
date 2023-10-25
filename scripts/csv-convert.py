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
            new_col += ' / ' + str(data.iat[1, col_idx])
        if pd.notna(data.iat[0, col_idx]):
            new_col += ' / ' + str(data.iat[0, col_idx])
        new_cols.append(new_col)
    
    data.columns = new_cols
    
    data = data.drop(columns=[col for col in data.columns if "品質情報" in str(data[col].iat[1]) or 
                                                         "均質番号" in str(data[col].iat[1]) or
                                                         "時分" in str(data[col].iat[0])])
    return data.drop([0, 1])


def aggregate_csv_files(folder_path, start_year, num_years):
    aggregated_data = []

    for i in range(num_years):
        file_name = f"data{start_year+i}1020-{start_year+i+1}1019_utf-8.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            data = preprocess_data(file_path)
            aggregated_data.append(data)
        else:
            print(f"File {file_name} does not exist!")

    result = pd.concat(aggregated_data, ignore_index=True)
    result.to_csv(os.path.join(folder_path, f'Compiled_{start_year}-{start_year+num_years}.csv'), index=False, encoding='utf-8')
    print(f"Data aggregated and saved as Compiled_{start_year}-{start_year+num_years}.csv")

if __name__ == "__main__":
    folder_path = './'
    current_year = 2023
    
    convert_to_utf8(folder_path, current_year)
    aggregate_csv_files(folder_path, 2013, 10)
