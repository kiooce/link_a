import os

# Set the number of threads for each relevant library
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
#import xarray as xr
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from aurora import Batch, Metadata

class WeatherDataset(Dataset):
    def __init__(self, df, sequence_length=2):
        self.df = df
        self.stations = df['station_name'].unique()  # This is a list of station names
        self.sequence_length = sequence_length
        self.features = ['pm10', 'so2', 'no2', 'pm25', 'o3', 
                            'temperature', 'humidity']  # This is a list of feature column names

        # Split data based on the station, combining data across all stations
        self.data = []
        for station in self.stations:
            station_data = self.df[self.df['station_name'] == station][self.features].values
            self.data.append(station_data)
        
        # Concatenate data across all stations
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        # Extract sequences for X (t-1, t) and y (t+1)
        X = self.data[idx:idx + self.sequence_length]  # Shape [seq_length, 7]
        y = self.data[idx + self.sequence_length]  # Shape [7]
        
        # Convert to tensor and ensure correct shape
        X = torch.tensor(X, dtype=torch.float32)  # Shape [seq_length, 7]
        y = torch.tensor(y, dtype=torch.float32)  # Shape [7]

        return X, y
    
    
def create_dataloaders(df, sequence_length, batch_size, test_size=0.2):
    dataset = WeatherDataset(df, sequence_length)

    # Use train_test_split to split the dataset indices
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, shuffle=False)

    # Ensure we have enough samples per batch in the training set
    train_indices = train_indices[:-(len(train_indices) % batch_size)]
    val_indices = val_indices[:-(len(val_indices) % batch_size)]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def calculate_mixing_ratio(temp_c, rh_percent, pressure_pa):
    # Calculate saturation vapor pressure (in hPa)
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    
    # Convert to Pa
    es_pa = es * 100
    
    # Calculate actual vapor pressure (in Pa)
    e = (rh_percent / 100.0) * es_pa
    
    # Calculate mixing ratio (in kg/kg)
    w = 0.622 * e / (pressure_pa - e)
    
    return w

def process_excel(filename: str) -> pd.DataFrame:

    df = pd.read_excel(io='data/Chongqing/'+filename, header=1, usecols="C:O")
    df.columns = ['date', 'time', 'PM10', 'SO2', 'NO2', 'PM25', 'O3', 'CO', 'pressure',
                'temperature', 'wind_dir', 'wind_velocity', 'humidity']

    df.replace('--', np.nan, inplace=True)

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Convert 'hours' column to timedelta (hours)
    df['time'] = pd.to_timedelta(df['time'], unit='h')

    # Combine date and hours into a single datetime column
    df['datetime'] = df['date'] + df['time']

    # Convert humidity (percentage) into mixing ratio
    pressure_pa = 101325  # Assuming sea-level pressure in Pascals
    df['humidity'] = df.apply(lambda row: calculate_mixing_ratio(row['temperature'], row['humidity'], pressure_pa), axis=1)

    # Convert temperature into Kelvin
    df['temperature'] = df['temperature'] + 273.15

    # assume the pressure is always 1000
    df['pressure'] =  1000

    df.drop(['date', 'time'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df = df.sort_values(by='datetime')
    cutoff_date = pd.to_datetime('2024-9-30 23:59:59')
    df = df[df['datetime'] <= cutoff_date]
    df = df.reset_index(drop=True)

    return df

def process_excel2(filename_weather: str, filename_airquality: str) -> pd.DataFrame:
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    weather_file_path = os.path.abspath(os.path.join(base_path, filename_weather))
    airquality_file_path = os.path.abspath(os.path.join(base_path, filename_airquality))

    # Print the full absolute paths to debug
    print(f"Weather file path: {weather_file_path}")
    print(f"Airquality file path: {airquality_file_path}")

    df_weather = pd.read_excel(io=weather_file_path, header=0, usecols="B,E,F,G,I")
    df_airquality = pd.read_excel(io=airquality_file_path, sheet_name=None, header=0, usecols="F,H,I,J,O,Q,S,U,W,AA")
    
    # 合并数据
    df_airquality = pd.concat(df_airquality.values(), ignore_index=True)
    df_airquality['time'] = df_airquality['monitoring_time']
    df_airquality.drop(['monitoring_time'], axis=1, inplace=True)
    
    # 处理时间
    df_weather['time'] = pd.to_datetime(df_weather['time'])  # 确保时间是日期格式
    df_airquality['time'] = pd.to_datetime(df_airquality['time'])  # 确保时间是日期格式

    df_weather.sort_values(by='time', inplace=True)
    df_airquality.sort_values(by='time', inplace=True)

    # 合并数据
    df_merged = pd.merge(df_weather, df_airquality, on=['station_name', 'time'], how='inner')

    df_merged.columns = ['station_name', 'time', 'temperature', 'humidity',
       'pressure', 'longitude', 'latitude', 'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    
    # 处理湿度为混合比
    df_merged['humidity'] = df_merged.apply(lambda row: calculate_mixing_ratio(row['temperature'], row['humidity'], row['pressure']), axis=1)
    df_merged['humidity'] = -df_merged['humidity']  # 假设湿度为负值
    df_merged['pressure'] = 1000

    df_merged = df_merged.sort_values(by='time')
    cutoff_date = pd.to_datetime('2024-9-30 23:59:59')
    df_merged = df_merged[df_merged['time'] <= cutoff_date]
    df_merged = df_merged.reset_index(drop=True)

    df_merged.dropna(inplace=True)

    return df_merged



def data_segmentation(df: pd.DataFrame, batch_size=16):
    """
    data.shape = (b, 2, 8, 4)
        2 timesptes
        8 features
        4 stations
    """
    columns = ['pm10', 'so2', 'no2', 'pm25', 'o3',
                'temperature', 'pressure', 'humidity']
    data = df[columns].values

    X, y = [], []

    for i in range(len(data) - 2):
        X.append(data[i:i+2])    # Two consecutive hours of data (没有考虑到4个station)
        y.append(data[i+2])      # The label is the next hour 

    # Convert lists to tensors
    X = torch.tensor(X, dtype=torch.float32)  # Shape: (n_samples, 2, n_features)
    y = torch.tensor(y, dtype=torch.float32)  # Shape: (n_samples, n_features)

    # Create a TensorDataset (to link X and y)
    dataset = TensorDataset(X, y)

    # Create a DataLoader with batch_size and shuffle the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # for batch_X, batch_y in dataloader:
    #     print("Batch X shape:", batch_X.shape)  # Should be (batch_size, 2, n_features)
    #     print("Batch y shape:", batch_y.shape)  # Should be (batch_size, n_features)

    return dataloader

def form_aurora_batch(df: pd.DataFrame):
    """
    Batch长什么样? => b, t, c, h, w
    (h, w) 就是不同的地点
    c 是 pressure level, 这里只有1个
    b 是 batch size, 不在单独batch的范围中
    t 是 time, 每次是两个
    """
    # Step 1: Sort the dataframe by 'time' and 'pressure' to ensure correct ordering
    df_sorted = df.sort_values(by=['time', 'pressure', 'station_name'])

    features = ['temperature', 'humidity', 'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    df_sorted = df_sorted.dropna()
    
    # Step 2: Pivot the dataframe to have 'longitude' and 'latitude' as a grid for each time and pressure
    df_pivoted = df_sorted.pivot_table(index=['time', 'pressure'], 
                                       columns=['station_name'], 
                                       values=features)
    
    # Step 3: Convert to a 4D NumPy array
    unique_times = df_pivoted.index.get_level_values('time').unique()
    t = df_pivoted.index.get_level_values('time').nunique()
    c = df_sorted['pressure'].nunique()  # number of pressure levels

    # Convert the pivoted DataFrame to a NumPy array and reshape it
    data_array = df_pivoted.values.reshape(t, c, -1, 2, 2) # feature_num = 8, station_num = 2*2

    # 我们需要产生n个X_batch和它们对应的y_batch
    # 时间到最后一个t之前两步 (t-2, using (t-2, t-1) to predict t+2)
    X_batches = []
    y_batches = []

    for i in range(1, t-2):
        if np.isnan(data_array[i-1:i+3]).any():
            continue

        X_batch = Batch(
            surf_vars={
                # 使用时间点 i-1 和 i 作为输入
                "2t": torch.from_numpy(data_array[i-1:i+1, 0, 7][None]).repeat(1, 1, 6, 6),
                "msl": torch.full((1, 2, 12, 12), 1000),
                "pm10": torch.from_numpy(data_array[i-1:i+1, 0, 4][None]).repeat(1, 1, 6, 6), 
                "pm25": torch.from_numpy(data_array[i-1:i+1, 0, 5][None]).repeat(1, 1, 6, 6), 
                "so2": torch.from_numpy(data_array[i-1:i+1, 0, 6][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array[i-1:i+1, 0, 2][None]).repeat(1, 1, 6, 6), 
                "o3": torch.from_numpy(data_array[i-1:i+1, 0, 3][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array[i-1:i+1,0,7][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array[i-1:i+1,0,1][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                # ✅ 修复：输入时间应该是当前时间点 i
                time=(unique_times[i],),
                atmos_levels=(1000,)
            ),
        )

        y_batch = Batch(
            surf_vars={
                # 预测时间点 i+2 的数据
                "2t": torch.from_numpy(data_array[i+2, 0, 7][None]).repeat(1, 1, 6, 6),
                "msl": torch.full((1, 1, 12, 12), 1000),
                "pm10": torch.from_numpy(data_array[i+2, 0, 4][None]).repeat(1, 1, 6, 6), 
                "pm25": torch.from_numpy(data_array[i+2, 0, 5][None]).repeat(1, 1, 6, 6), 
                "so2": torch.from_numpy(data_array[i+2, 0, 6][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array[i+2, 0, 2][None]).repeat(1, 1, 6, 6), 
                "o3": torch.from_numpy(data_array[i+2, 0, 3][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array[i+2,0,7][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array[i+2,0,1][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                # ✅ 修复：预测时间应该是未来时间点 i+2
                time=(unique_times[i+2],),
                atmos_levels=(1000,)
            ),
        )

        X_batches.append(X_batch)
        y_batches.append(y_batch)

    return X_batches, y_batches

def plot_predictions_vs_labels(all_preds, all_labels, feature_names, output_file='predictions_vs_labels.png',
                               order=[0, 1, 2, 3, 4]):
    """
    Plots the predictions vs. labels for all features and saves the graph as an image file.
    
    Parameters:
        all_preds (numpy array): Predicted values, shape [num_samples, num_features].
        all_labels (numpy array): Ground truth labels, shape [num_samples, num_features].
        feature_names (list): List of feature names, length should be num_features.
    """

    num_features = len(feature_names)

    # Create subplots for each feature
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 5 * num_features), sharex=True)

    # feature_order = [1, 3, 2, 4, 0]
    # feature_order = [0, 1, 2, 3, 4, 5, 6]

    # Iterate over each feature
    # for feature_index in range(num_features):
    for i, feature_index in enumerate(order):
        # Prepare the x-axis values (assuming continuous time steps)
        x_values = np.arange(all_preds.shape[0])

        # Plot predictions and labels for the current feature
        axes[i].plot(x_values, all_preds[:, feature_index], label='Predictions', color='blue')
        axes[i].plot(x_values, all_labels[:, feature_index], label='Labels', color='orange')
        
        # Set labels and title
        axes[i].set_ylabel(feature_names[feature_index])
        axes[i].legend()
        axes[i].grid(True)

    # Set the common x-axis label
    axes[-1].set_xlabel('Time Steps')
    
    # Save the figure to the output directory
    plt.savefig(output_file)
    plt.close(fig)  # Close the figure to avoid displaying it inline
