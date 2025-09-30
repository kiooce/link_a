# utils_clean.py
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch
from aurora import Batch, Metadata

def calculate_specific_humidity(temp_k, rh_percent, pressure_pa):
    """计算比湿度"""
    temp_c = temp_k - 273.15
    es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e = (rh_percent / 100.0) * es
    q = 0.622 * e / (pressure_pa - 0.378 * e)
    return q

def wind_speed_dir_to_uv(wind_speed, wind_dir):
    """风速风向转u/v分量"""
    wind_dir_rad = np.radians(wind_dir)
    u = -wind_speed * np.sin(wind_dir_rad)
    v = -wind_speed * np.cos(wind_dir_rad)
    return u, v

def uv_to_wind_speed_dir(u, v):
    """u/v分量转风速风向"""
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = np.degrees(np.arctan2(-u, -v)) % 360
    return wind_speed, wind_dir

def specific_humidity_to_rh(q, temp_k, pressure_pa):
    """比湿度转相对湿度"""
    temp_c = temp_k - 273.15
    es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e = q * pressure_pa / (0.622 + 0.378 * q)
    rh = (e / es) * 100.0
    return np.clip(rh, 0, 100)

def process_excel2(filename_weather: str, filename_airquality: str) -> pd.DataFrame:
    base_path = '/home/zhepingliu/aurora_code/aurora_weather/data/Chongqing/'
    weather_file_path = os.path.join(base_path, filename_weather)
    airquality_file_path = os.path.join(base_path, filename_airquality)

    # 加载气象数据
    df_weather = pd.read_excel(weather_file_path, header=0)
    df_weather = df_weather[['station_name', 'time', '温度 单位开尔文K--减去273.15换算为摄氏度', 
                           '湿度', '小时降雨量 mm', '气压', '风速', '风向']].copy()
    df_weather.columns = ['station_name', 'time', 'temperature', 'humidity', 'rainfall', 
                          'pressure', 'wind_speed', 'wind_direction']
    
    # 加载空气质量数据
    df_airquality = pd.read_excel(airquality_file_path, sheet_name=None, header=0)
    df_airquality = pd.concat(df_airquality.values(), ignore_index=True)
    df_airquality = df_airquality[['station_name', 'monitoring_time', 'longitude', 'latitude', 
                                   'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']].copy()
    df_airquality.rename(columns={'monitoring_time': 'time'}, inplace=True)
    
    # 处理时间格式
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_airquality['time'] = pd.to_datetime(df_airquality['time'])
    
    # 合并数据
    df_merged = pd.merge(df_weather, df_airquality, on=['station_name', 'time'], how='inner')
    
    # 数据预处理
    df_merged['specific_humidity'] = calculate_specific_humidity(
        df_merged['temperature'], df_merged['humidity'], df_merged['pressure'])
    
    u_wind, v_wind = wind_speed_dir_to_uv(df_merged['wind_speed'], df_merged['wind_direction'])
    df_merged['u_wind'] = u_wind
    df_merged['v_wind'] = v_wind
    df_merged['pressure_hpa'] = df_merged['pressure'] / 100.0
    
    # 过滤时间
    cutoff_date = pd.to_datetime('2024-9-30 23:59:59')
    df_merged = df_merged[df_merged['time'] <= cutoff_date]
    df_merged = df_merged.reset_index(drop=True)

    # 清理NaN
    cleaned_data_list = []
    for station in ['上清寺', '唐家沱', '天生', '龙井湾', '鱼新街']:  # 固定顺序
        if station in df_merged['station_name'].unique():
            station_data = df_merged[df_merged['station_name'] == station].copy()
            station_data = station_data.dropna(subset=['temperature', 'pressure', 'pm25', 'pm10'])
            cleaned_data_list.append(station_data)

    df_merged = pd.concat(cleaned_data_list, ignore_index=True)
    
    return df_merged

def form_aurora_batch(df: pd.DataFrame):
    """构建Aurora批次"""
    stations = df['station_name'].unique()
    desired_stations = ['上清寺', '唐家沱', '天生', '龙井湾']
    selected_stations = [s for s in desired_stations if s in stations]
    
    if len(selected_stations) != 4:
        raise ValueError(f"需要4个站点，当前只有{len(selected_stations)}个")
    
    df_filtered = df[df['station_name'].isin(selected_stations)].copy()
    
    # 排序
    df_filtered['station_order'] = df_filtered['station_name'].map({
        '上清寺': 0, '唐家沱': 1, '天生': 2, '龙井湾': 3
    })
    df_sorted = df_filtered.sort_values(by=['time', 'station_order'])
    df_sorted = df_sorted.drop('station_order', axis=1).dropna()
    
    features = ['temperature', 'specific_humidity', 'rainfall', 'pressure_hpa', 'u_wind', 'v_wind',
                'pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    
    # 构建数据数组
    unique_times = sorted(df_sorted['time'].unique())
    n_times = len(unique_times)
    n_features = len(features)
    n_stations = 4
    
    data_array_3d = np.full((n_times, n_features, n_stations), np.nan)
    valid_times = []
    
    for t_idx, time_point in enumerate(unique_times):
        time_data = df_sorted[df_sorted['time'] == time_point]
        
        if len(time_data['station_name'].unique()) != 4:
            continue
            
        all_stations_present = True
        for s_idx, station in enumerate(selected_stations):
            station_data = time_data[time_data['station_name'] == station]
            if len(station_data) != 1:
                all_stations_present = False
                break
                
            for f_idx, feature in enumerate(features):
                if feature in station_data.columns:
                    value = station_data[feature].iloc[0]
                    if not np.isnan(value):
                        data_array_3d[t_idx, f_idx, s_idx] = value
                    else:
                        all_stations_present = False
                        break
            
            if not all_stations_present:
                break
        
        if all_stations_present:
            valid_times.append(time_point)
    
    valid_indices = [i for i, t in enumerate(unique_times) if t in valid_times]
    data_array_3d = data_array_3d[valid_indices]
    valid_times = np.array(valid_times)
    
    # 重塑为2x2网格
    data_array_4d = data_array_3d.reshape(len(valid_times), n_features, 2, 2)
    
    # 生成批次
    X_batches = []
    y_batches = []
    feat_map = {feature: idx for idx, feature in enumerate(features)}
    
    for i in range(1, len(valid_times)-2):
        if np.isnan(data_array_4d[i-1:i+3]).any():
            continue
            
        X_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['temperature']][None]).repeat(1, 1, 6, 6),
                "10u": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['u_wind']][None]).repeat(1, 1, 6, 6),
                "10v": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['v_wind']][None]).repeat(1, 1, 6, 6),
                "msl": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['pressure_hpa']][None]).repeat(1, 1, 6, 6) * 100,
                "pm10": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['pm10']][None]).repeat(1, 1, 6, 6),
                "pm25": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['pm25']][None]).repeat(1, 1, 6, 6),
                "so2": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['so2']][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['no2']][None]).repeat(1, 1, 6, 6),
                "o3": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['o3']][None]).repeat(1, 1, 6, 6),
                "co": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['co']][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['temperature']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['specific_humidity']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "u": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['u_wind']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "v": torch.from_numpy(data_array_4d[i-1:i+1, feat_map['v_wind']][None]).view((1, 2, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                time=(valid_times[i],),
                atmos_levels=(1000,)
            ),
        )

        y_batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(data_array_4d[i+2, feat_map['temperature']][None]).repeat(1, 1, 6, 6),
                "10u": torch.from_numpy(data_array_4d[i+2, feat_map['u_wind']][None]).repeat(1, 1, 6, 6),
                "10v": torch.from_numpy(data_array_4d[i+2, feat_map['v_wind']][None]).repeat(1, 1, 6, 6),
                "msl": torch.from_numpy(data_array_4d[i+2, feat_map['pressure_hpa']][None]).repeat(1, 1, 6, 6) * 100,
                "pm10": torch.from_numpy(data_array_4d[i+2, feat_map['pm10']][None]).repeat(1, 1, 6, 6),
                "pm25": torch.from_numpy(data_array_4d[i+2, feat_map['pm25']][None]).repeat(1, 1, 6, 6),
                "so2": torch.from_numpy(data_array_4d[i+2, feat_map['so2']][None]).repeat(1, 1, 6, 6),
                "no2": torch.from_numpy(data_array_4d[i+2, feat_map['no2']][None]).repeat(1, 1, 6, 6),
                "o3": torch.from_numpy(data_array_4d[i+2, feat_map['o3']][None]).repeat(1, 1, 6, 6),
                "co": torch.from_numpy(data_array_4d[i+2, feat_map['co']][None]).repeat(1, 1, 6, 6),
            },
            static_vars={
                "slt": torch.full((12, 12), 1),
                "lsm": torch.full((12, 12), 1),
            },
            atmos_vars={
                "t": torch.from_numpy(data_array_4d[i+2, feat_map['temperature']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "q": torch.from_numpy(data_array_4d[i+2, feat_map['specific_humidity']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "u": torch.from_numpy(data_array_4d[i+2, feat_map['u_wind']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
                "v": torch.from_numpy(data_array_4d[i+2, feat_map['v_wind']][None]).view((1, 1, 1, 2, 2)).repeat(1, 1, 1, 6, 6),
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 12),
                lon=torch.linspace(0, 360, 12 + 1)[:-1],
                time=(valid_times[i+2],),
                atmos_levels=(1000,)
            ),
        )

        X_batches.append(X_batch)
        y_batches.append(y_batch)
    
    print(f"生成批次数: {len(X_batches)}")
    return X_batches, y_batches
