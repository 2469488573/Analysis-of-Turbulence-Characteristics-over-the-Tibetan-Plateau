# -*- coding: utf-8 -*-
#%%加载module
"""
Created on Mon Mar 17 16:15:20 2025

对青藏高原上湍流基本特征分析

@author: mayubin 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
# 设置全局字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置全局字体样式

plt.rcParams['font.size'] = 12  # 设置字体大小
plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # 设置横轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 10  # 设置纵轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 12  # 设置图例字体大小

plt.rcParams['text.usetex'] = False  # 需要系统安装LaTeX

from scipy.optimize import curve_fit

#%%常用物理常数
kaman = 0.4    
g=9.8#m/s^2
cp=1004
rho = 1.29
epsilon = 0.622
#%%加载文件
import os
import pandas as pd

# 基础路径
base_path = r'C:\Users\Lenovo\Desktop\data_surface'

# 文件夹前缀列表
folder_prefixes = ['MAWORS', 'NADORS', 'PLAN', 'QOMO_B', 'SETORS', 'NIMA', 'BANG', 'MANA', 'CHDU']

# 文件类型列表
file_types = ['FLUX', 'GRAD', 'RADM', 'SOIL']

# 创建字典来存储所有DataFrame
data_dict = {}

# 遍历每个文件夹
for i, prefix in enumerate(folder_prefixes, start=1):
    folder_path = os.path.join(base_path, f'data{i}')
    
    # 遍历每个文件类型
    for file_type in file_types:
        # 构建文件名
        file_name = f'{prefix}_{file_type}_2023_hourly.csv'
        file_path = os.path.join(folder_path, file_name)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 生成新变量名
        new_var_name = f'{file_type}{i}'
        
        # 将DataFrame保存到字典和全局变量
        data_dict[new_var_name] = df
        globals()[new_var_name] = df
        
        print(f'已加载 {file_path} 并保存为 {new_var_name}')

# 现在可以直接使用FLUX1到FLUX9, GRAD1到GRAD9等变量
print("所有文件已加载完成！")
print(f"例如，GRAD5包含的数据形状为：{GRAD5.shape}")
position_name = folder_prefixes


#%%资料处理


# 常数定义
USTAR_THRESHOLD = 1.0  # 摩擦速度阈值 (单位: m/s)

def filter_ustar_outliers(flux_df, threshold=USTAR_THRESHOLD):
    """
    将USTAR大于阈值的值设为NaN
    参数:
        flux_df: 输入的DataFrame
        threshold: 摩擦速度阈值 (默认1.0 m/s)
    返回:
        处理后的DataFrame
    """
    if 'USTAR' not in flux_df.columns:
        raise KeyError("DataFrame中缺少USTAR列")
    
    # 记录原始有效值数量
    original_count = flux_df['USTAR'].notna().sum()
    
    # 应用阈值过滤
    flux_df['USTAR'] = np.where(flux_df['USTAR'] > threshold, 
                               np.nan, 
                               flux_df['USTAR'])
    
    # 计算过滤数量
    filtered_count = original_count - flux_df['USTAR'].notna().sum()
    return flux_df, filtered_count

# 处理所有FLUX表
for i in range(1, 10):
    df_name = f'FLUX{i}'
    try:
        if df_name in globals():
            original_shape = globals()[df_name].shape
            globals()[df_name], n_filtered = filter_ustar_outliers(globals()[df_name])
            
            print(f"{df_name}: 原始数据 {original_shape[0]} 行 | "
                  f"过滤 {n_filtered} 个USTAR异常值 | "
                  f"剩余有效值 {globals()[df_name]['USTAR'].notna().sum()}")
            
            # 验证结果
            assert globals()[df_name]['USTAR'].max() <= USTAR_THRESHOLD
        else:
            print(f"警告: {df_name} 不存在")
    except Exception as e:
        print(f"处理 {df_name} 时出错: {str(e)}")


#%%首先计算奥布霍夫长度
#FLUX1['L'] = -FLUX1['USTAR']**3/(kaman*(g/FLUX1['T_SONIC'])*(FLUX1['H']/cp/rho) )
#FLUX2['L'] = -FLUX2['USTAR']**3/(kaman*(g/FLUX2['T_SONIC'])*(FLUX2['H']/cp/rho) )
#FLUX3['L'] = -FLUX3['USTAR']**3/(kaman*(g/FLUX3['T_SONIC'])*(FLUX3['H']/cp/rho) )
#FLUX4['L'] = -FLUX4['USTAR']**3/(kaman*(g/FLUX4['T_SONIC'])*(FLUX4['H']/cp/rho) )
#FLUX5['L'] = -FLUX5['USTAR']**3/(kaman*(g/FLUX5['T_SONIC'])*(FLUX5['H']/cp/rho) )
#FLUX6['L'] = -FLUX6['USTAR']**3/(kaman*(g/FLUX6['T_SONIC'])*(FLUX6['H']/cp/rho) )
#FLUX7['L'] = -FLUX7['USTAR']**3/(kaman*(g/FLUX7['T_SONIC'])*(FLUX7['H']/cp/rho) )
#FLUX8['L'] = -FLUX8['USTAR']**3/(kaman*(g/FLUX8['T_SONIC'])*(FLUX8['H']/cp/rho) )
#FLUX9['L'] = -FLUX9['USTAR']**3/(kaman*(g/FLUX9['T_SONIC'])*(FLUX9['H']/cp/rho) )

import numpy as np

# 定义常数
karman = 0.4  # 卡曼常数
g = 9.81      # 重力加速度 (m/s²)
cp = 1005     # 空气定压比热 (J/kg/K)
rho = 1.2     # 空气密度 (kg/m³)

# 循环处理所有FLUX数据框
for i in range(1, 10):
    flux = globals()[f'FLUX{i}']
    
    # 计算L值
    flux['L'] = -flux['USTAR']**3 / (karman * (g/flux['T_SONIC']) * (flux['H']/cp/rho))
    
    # 将大于100的L值设为NaN
    flux.loc[flux['L'] > 1000, 'L'] = np.nan
    flux.loc[flux['L'] < -1000, 'L'] = np.nan
    # 打印信息确认
    print(f"FLUX{i} - L值大于1000的数量: {(flux['L'] > 1000).sum()}")
    print(f"FLUX{i} - NaN值的数量: {flux['L'].isna().sum()}")
#%%观测站的湍流通量测量高度，湍流仪器距离地面高度

FLUX1['z_obs'] = 3
FLUX2['z_obs'] = 3.7
FLUX3['z_obs'] = 3.5
FLUX4['z_obs'] = 3
FLUX5['z_obs'] = 3.13
FLUX6['z_obs'] = 3.2
FLUX7['z_obs'] = 3.4
FLUX8['z_obs'] = 3.5
FLUX9['z_obs'] = 3.5
#%%观测站点的海拔高度，观测站点的地面距离海平面的高度(m)

FLUX1['level_obs'] = 3652
FLUX2['level_obs'] = 4252
FLUX3['level_obs'] = 4113
FLUX4['level_obs'] = 4308
FLUX5['level_obs'] = 3292
FLUX6['level_obs'] = 4545
FLUX7['level_obs'] = 4709
FLUX8['level_obs'] = 3016
FLUX9['level_obs'] = 3275
#%%#梯度测量的观测仪器的高度(m)

GRAD1['z1'] = 1.5;GRAD1['z2'] = 2.5;GRAD1['z3'] = 4;GRAD1['z4'] = 10;GRAD1['z5'] = 20
GRAD2['z1'] = 1.5;GRAD2['z2'] = 2.5;GRAD2['z3'] = 4;GRAD2['z4'] = 10;GRAD2['z5'] = 20
GRAD3['z1'] = 1  ;GRAD3['z2'] = 2  ;GRAD3['z3'] = 4;GRAD3['z4'] = 10;GRAD3['z5'] = 20
GRAD4['z1'] = 1.5;GRAD4['z2'] = 2.5;GRAD4['z3'] = 4;GRAD4['z4'] = 10;GRAD4['z5'] = 20
GRAD5['z1'] = 1  ;GRAD5['z2'] = 2  ;GRAD5['z3'] = 4;GRAD5['z4'] = 10;GRAD5['z5'] = 20
GRAD6['z1'] = 0.5;GRAD6['z2'] = 1  ;GRAD6['z3'] = 4;GRAD6['z4'] = 10;GRAD6['z5'] = 20
GRAD7['z1'] = 1  ;GRAD7['z2'] = 2  ;GRAD7['z3'] = 4;GRAD7['z4'] = 10;GRAD7['z5'] = 20
GRAD8['z1'] = 1  ;GRAD8['z2'] = 2  ;GRAD8['z3'] = 4;GRAD8['z4'] = 10;GRAD8['z5'] = 20
GRAD9['z1'] = 1  ;GRAD9['z2'] = 2  ;GRAD9['z3'] = 4;GRAD9['z4'] = 10;GRAD9['z5'] = 20



#%%#计算月平均数据，



#将索引换为时间，且为Python可认
FLUX1['Timestamp'] = pd.to_datetime(FLUX1['Timestamp'])
FLUX1.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX1 = FLUX1.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX1_monthly_avg = FLUX1.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX2['Timestamp'] = pd.to_datetime(FLUX2['Timestamp'])
FLUX2.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX2 = FLUX2.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX2_monthly_avg = FLUX2.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX3['Timestamp'] = pd.to_datetime(FLUX3['Timestamp'])
FLUX3.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX3 = FLUX3.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX3_monthly_avg = FLUX3.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX4['Timestamp'] = pd.to_datetime(FLUX4['Timestamp'])
FLUX4.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX4 = FLUX4.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX4_monthly_avg = FLUX4.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX5['Timestamp'] = pd.to_datetime(FLUX5['Timestamp'])
FLUX5.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX5 = FLUX5.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX5_monthly_avg = FLUX5.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX6['Timestamp'] = pd.to_datetime(FLUX6['Timestamp'])
FLUX6.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX6 = FLUX6.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX6_monthly_avg = FLUX6.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX7['Timestamp'] = pd.to_datetime(FLUX7['Timestamp'])
FLUX7.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX7 = FLUX7.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX7_monthly_avg = FLUX7.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX8['Timestamp'] = pd.to_datetime(FLUX8['Timestamp'])
FLUX8.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX8 = FLUX8.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX8_monthly_avg = FLUX8.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------
#将索引换为时间，且为Python可认
FLUX9['Timestamp'] = pd.to_datetime(FLUX9['Timestamp'])
FLUX9.set_index('Timestamp', inplace=True)
# 删除列 'B'
FLUX9 = FLUX9.drop('FP_EQUATION', axis=1)  # axis=1 表示按列删除
#对每个月平均
FLUX9_monthly_avg = FLUX9.resample('M').mean()  # 'M' 表示按月重采样
#-----------------------------------------------------------------------


#%%#感热随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['H'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['H'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['H'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['H'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['H'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['H'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['H'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['H'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['H'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Sensible heat flux(W/m^2)')#
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('H.png', dpi=300, bbox_inches='tight', format='png')

#%%#潜热随月份变化



plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['LE'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['LE'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['LE'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['LE'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['LE'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['LE'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['LE'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['LE'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['LE'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Latent heat flux(W/m^2)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('LE.png', dpi=300, bbox_inches='tight', format='png')


#%%#应力随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['TAU'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['TAU'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['TAU'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['TAU'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['TAU'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['TAU'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['TAU'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['TAU'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['TAU'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Stress(N/m^2)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('TAU1.png', dpi=300, bbox_inches='tight', format='png')


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
#plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['TAU'], label=position_name[0],marker="o")
#plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['TAU'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['TAU'], label=position_name[2],marker="p")
#plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['TAU'], label=position_name[3],marker="^")
#plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['TAU'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['TAU'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['TAU'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['TAU'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['TAU'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Stress(N/m^2)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('TAU2.png', dpi=300, bbox_inches='tight', format='png')


#%%#USTAR随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['USTAR'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['USTAR'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['USTAR'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['USTAR'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['USTAR'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['USTAR'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['USTAR'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['USTAR'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['USTAR'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Friction Velocity(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('USTAR1.png', dpi=300, bbox_inches='tight', format='png')


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
#plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['USTAR'], label=position_name[0],marker="o")
#plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['USTAR'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['USTAR'], label=position_name[2],marker="p")
#plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['USTAR'], label=position_name[3],marker="^")
#plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['USTAR'], label=position_name[4],marker=",")
#plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['USTAR'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['USTAR'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['USTAR'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['USTAR'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Friction Velocity(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('USTAR2.png', dpi=300, bbox_inches='tight', format='png')


#%%#TSTAR随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['TSTAR'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['TSTAR'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['TSTAR'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['TSTAR'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['TSTAR'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['TSTAR'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['TSTAR'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['TSTAR'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['TSTAR'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Temperature friction Velocity(K)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('TSTAR.png', dpi=300, bbox_inches='tight', format='png')


#%%#TKE随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['TKE'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['TKE'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['TKE'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['TKE'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['TKE'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['TKE'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['TKE'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['TKE'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['TKE'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Turbulent kinetic energy(m2/s2)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('TKE1.png', dpi=300, bbox_inches='tight', format='png')


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
#plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['TKE'], label=position_name[0],marker="o")
#plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['TKE'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['TKE'], label=position_name[2],marker="p")
#plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['TKE'], label=position_name[3],marker="^")
#plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['TKE'], label=position_name[4],marker=",")
#plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['TKE'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['TKE'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['TKE'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['TKE'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Turbulent kinetic energy(m2/s2)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('TKE2.png', dpi=300, bbox_inches='tight', format='png')

#%%#温度随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['T_SONIC'], label=position_name[0],marker="o")
plt.plot(FLUX2_monthly_avg.index, FLUX2_monthly_avg['T_SONIC'], label=position_name[1],marker="s")
plt.plot(FLUX3_monthly_avg.index, FLUX3_monthly_avg['T_SONIC'], label=position_name[2],marker="p")
plt.plot(FLUX4_monthly_avg.index, FLUX4_monthly_avg['T_SONIC'], label=position_name[3],marker="^")
plt.plot(FLUX5_monthly_avg.index, FLUX5_monthly_avg['T_SONIC'], label=position_name[4],marker=",")
plt.plot(FLUX6_monthly_avg.index, FLUX6_monthly_avg['T_SONIC'], label=position_name[5],marker="*")
plt.plot(FLUX7_monthly_avg.index, FLUX7_monthly_avg['T_SONIC'], label=position_name[6],marker="+")
plt.plot(FLUX8_monthly_avg.index, FLUX8_monthly_avg['T_SONIC'], label=position_name[7],marker="d")
plt.plot(FLUX9_monthly_avg.index, FLUX9_monthly_avg['T_SONIC'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Temperature(deg C)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('T_SONIC.png', dpi=300, bbox_inches='tight', format='png')





#%%#对GRAD求月平均数据

#现在还不清楚每个观测高度分别是多高。暂且先用变量z1 ,z2 ,z3 ,z4 ,z5 来表示
z1,z2,z3,z4,z5  = 1,2,4,10,20




# 假设 GRAD1 到 GRAD9 是已经加载的 DataFrame
# 将它们存储在一个字典中，方便循环操作
grad_dict = {
    'GRAD1': GRAD1,
    'GRAD2': GRAD2,
    'GRAD3': GRAD3,
    'GRAD4': GRAD4,
    'GRAD5': GRAD5,
    'GRAD6': GRAD6,
    'GRAD7': GRAD7,
    'GRAD8': GRAD8,
    'GRAD9': GRAD9
}

# 循环处理每个表
for name, df in grad_dict.items():
    # 将 'Timestamp' 列转换为 datetime 类型，并设置为索引
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # 删除列 'Unnamed: 37'
    df = df.drop('Unnamed: 37', axis=1)  # axis=1 表示按列删除
    
    # 对每个月进行平均
    monthly_avg = df.resample('ME').mean()  # 'M' 表示按月重采样
    
    # 将处理后的表存回字典
    grad_dict[name] = monthly_avg

# 现在 grad_dict 中存储的是处理后的按月平均的表
# 可以通过 grad_dict['GRAD1'] 访问处理后的 GRAD1 表
    # 动态生成变量 GRAD1_monthly_avg 到 GRAD9_monthly_avg
    new_name = f'{name}_monthly_avg'
    globals()[new_name] = monthly_avg
#%%计算各个表格的年平均量
# 假设 FLUX1, FLUX2, ..., GRAD1, GRAD2, ... 已经是 DataFrame
prefixes = ["FLUX", "GRAD"]

for prefix in prefixes:
    for i in range(1, 10):  # 处理 1-9
        df_name = f"{prefix}{i}"  # 如 "FLUX1"
        if df_name in globals():  # 检查变量是否存在
            df = globals()[df_name]  # 获取 DataFrame
            yearly_avg = df.mean(axis=0)  # 计算年平均
            globals()[f"{df_name}_yearly_avg"] = yearly_avg  # 存储为新变量
            print(f"已计算 {df_name}_yearly_avg")

#%%#降水随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Rain_Tot'], label=position_name[0],marker="o")
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['Rain_Tot'], label=position_name[1],marker="s")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['Rain_Tot'], label=position_name[2],marker="p")
plt.plot(GRAD4_monthly_avg.index, GRAD4_monthly_avg['Rain_Tot'], label=position_name[3],marker="^")
plt.plot(GRAD5_monthly_avg.index, GRAD5_monthly_avg['Rain_Tot'], label=position_name[4],marker=",")
plt.plot(GRAD6_monthly_avg.index, GRAD6_monthly_avg['Rain_Tot'], label=position_name[5],marker="*")
plt.plot(GRAD7_monthly_avg.index, GRAD7_monthly_avg['Rain_Tot'], label=position_name[6],marker="+")
plt.plot(GRAD8_monthly_avg.index, GRAD8_monthly_avg['Rain_Tot'], label=position_name[7],marker="d")
plt.plot(GRAD9_monthly_avg.index, GRAD9_monthly_avg['Rain_Tot'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Rain_total(m)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('Rain.png', dpi=300, bbox_inches='tight', format='png')
    
    
#%%#气压随月份变化


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Pressure'], label=position_name[0],marker="o")
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['Pressure'], label=position_name[1],marker="s")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['Pressure'], label=position_name[2],marker="p")
plt.plot(GRAD4_monthly_avg.index, GRAD4_monthly_avg['Pressure'], label=position_name[3],marker="^")
plt.plot(GRAD5_monthly_avg.index, GRAD5_monthly_avg['Pressure'], label=position_name[4],marker=",")
plt.plot(GRAD6_monthly_avg.index, GRAD6_monthly_avg['Pressure'], label=position_name[5],marker="*")
plt.plot(GRAD7_monthly_avg.index, GRAD7_monthly_avg['Pressure'], label=position_name[6],marker="+")
plt.plot(GRAD8_monthly_avg.index, GRAD8_monthly_avg['Pressure'], label=position_name[7],marker="d")
plt.plot(GRAD9_monthly_avg.index, GRAD9_monthly_avg['Pressure'], label=position_name[8],marker="4")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Pressure(hPa)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('Pressure.png', dpi=300, bbox_inches='tight', format='png')
    
        
    
    

#%%#五个高度层变量随月份变化,几条折线是不同高度层


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Ta_1'], label='Ta_'+str(z1)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Ta_2'], label='Ta_'+str(z2)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Ta_3'], label='Ta_'+str(z3)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Ta_4'], label='Ta_'+str(z4)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Ta_5'], label='Ta_'+str(z5)+'m',marker="o")
plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['T_SONIC'], label='T_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Temperature(deg C)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('T_GRAD.png', dpi=300, bbox_inches='tight', format='png')
#%%#五个高度层变量随月份变化,几条折线是不同高度层


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX1_monthly_avg.index, np.sqrt(FLUX1_monthly_avg['Ux']**2+FLUX1_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD1.png', dpi=300, bbox_inches='tight', format='png')
#-----------------------------------2------------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD2_monthly_avg.index, GRAD2_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX2_monthly_avg.index, np.sqrt(FLUX2_monthly_avg['Ux']**2+FLUX2_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD2.png', dpi=300, bbox_inches='tight', format='png')

#-----------------------------------3--------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#-------------------------------------4----------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#------------------------------------------------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#------------------------------------------------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#------------------------------------------------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#------------------------------------------------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#------------------------------------------------------------------------------
#五个高度层变量随月份变化,几条折线是不同高度层

plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_1'], label='WS_'+str(z1)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_2'], label='WS_'+str(z2)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_3'], label='WS_'+str(z3)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_4'], label='WS_'+str(z4)+'m',marker="o")
plt.plot(GRAD3_monthly_avg.index, GRAD3_monthly_avg['WS_5'], label='WS_'+str(z5)+'m',marker="o")
plt.plot(FLUX3_monthly_avg.index, np.sqrt(FLUX3_monthly_avg['Ux']**2+FLUX3_monthly_avg['Uy']**2), label='WS_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('wind speed(m/s)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('WS_GRAD3.png', dpi=300, bbox_inches='tight', format='png')
#------------------------------------------------------------------------------














































#%%#五个高度层变量随月份变化  ,几条折线是不同高度层


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['RH_1'], label='RH_'+str(z1)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['RH_2'], label='RH_'+str(z2)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['RH_3'], label='RH_'+str(z3)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['RH_4'], label='RH_'+str(z4)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['RH_5'], label='RH_'+str(z5)+'m',marker="o")
#plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['e_probe']/FLUX1_monthly_avg['e_sat_probe'], label='RH_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('RH(%)')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('RH_GRAD.png', dpi=300, bbox_inches='tight', format='png')

#%%#五个高度层变量随月份变化,几条折线是不同高度层


plt.figure(figsize=(10, 6))  # 设置画布大小
#for column in df.columns:
#    plt.plot(df.index, df[column], label=column)  # 绘制折线图
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Pvapor_1'], label='Pvapor_'+str(z1)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Pvapor_2'], label='Pvapor_'+str(z2)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Pvapor_3'], label='Pvapor_'+str(z3)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Pvapor_4'], label='Pvapor_'+str(z4)+'m',marker="o")
plt.plot(GRAD1_monthly_avg.index, GRAD1_monthly_avg['Pvapor_5'], label='Pvapor_'+str(z5)+'m',marker="o")
#plt.plot(FLUX1_monthly_avg.index, FLUX1_monthly_avg['e_probe']/FLUX1_monthly_avg['e_sat_probe'], label='RH_surface',marker="o")

# 添加标题和标签
#months = np.arange(1, 13)  # 1 到 12 月
# 设置横坐标标签为 1 到 12 月
#plt.xticks(months, labels=[f'{i}月' for i in months])
# 设置横坐标格式为月份缩写

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 表示月份缩写
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月定位刻度

plt.title('')
plt.xlabel('Month')
plt.ylabel('Pvapor()')
#plt.xticks(rotation=45)
# 设置时间范围
#start_time = pd.to_datetime('2023-01-01 00:00')
#end_time = pd.to_datetime('2023-12-31 00:00')
#plt.xlim(start_time, end_time)
plt.legend()

plt.show()
# 保存为高质量图片
plt.savefig('Pvapor_GRAD.png', dpi=300, bbox_inches='tight', format='png') 









#%%计算比湿无量纲梯度和稳定度参数之间关系


GRAD1['q_1'] = epsilon * GRAD1['Pvapor_1']*10 /GRAD1['Pressure']
GRAD2['q_1'] = epsilon * GRAD2['Pvapor_1']*10 /GRAD2['Pressure']
GRAD3['q_1'] = epsilon * GRAD3['Pvapor_1']*10 /GRAD3['Pressure']
GRAD4['q_1'] = epsilon * GRAD4['Pvapor_1']*10 /GRAD4['Pressure']
GRAD5['q_1'] = epsilon * GRAD5['Pvapor_1']*10 /GRAD5['Pressure']
GRAD6['q_1'] = epsilon * GRAD6['Pvapor_1']*10 /GRAD6['Pressure']
GRAD7['q_1'] = epsilon * GRAD7['Pvapor_1']*10 /GRAD7['Pressure']
GRAD8['q_1'] = epsilon * GRAD8['Pvapor_1']*10 /GRAD8['Pressure']
GRAD9['q_1'] = epsilon * GRAD9['Pvapor_1']*10 /GRAD9['Pressure']
#------------------------------------------------------------------------------
GRAD1['q_2'] = epsilon * GRAD1['Pvapor_2']*10 /GRAD1['Pressure']
GRAD2['q_2'] = epsilon * GRAD2['Pvapor_2']*10 /GRAD2['Pressure']
GRAD3['q_2'] = epsilon * GRAD3['Pvapor_2']*10 /GRAD3['Pressure']
GRAD4['q_2'] = epsilon * GRAD4['Pvapor_2']*10 /GRAD4['Pressure']
GRAD5['q_2'] = epsilon * GRAD5['Pvapor_2']*10 /GRAD5['Pressure']
GRAD6['q_2'] = epsilon * GRAD6['Pvapor_2']*10 /GRAD6['Pressure']
GRAD7['q_2'] = epsilon * GRAD7['Pvapor_2']*10 /GRAD7['Pressure']
GRAD8['q_2'] = epsilon * GRAD8['Pvapor_2']*10 /GRAD8['Pressure']
GRAD9['q_2'] = epsilon * GRAD9['Pvapor_2']*10 /GRAD9['Pressure']
#------------------------------------------------------------------------------
GRAD1['q_3'] = epsilon * GRAD1['Pvapor_3']*10 /GRAD1['Pressure']
GRAD2['q_3'] = epsilon * GRAD2['Pvapor_3']*10 /GRAD2['Pressure']
GRAD3['q_3'] = epsilon * GRAD3['Pvapor_3']*10 /GRAD3['Pressure']
GRAD4['q_3'] = epsilon * GRAD4['Pvapor_3']*10 /GRAD4['Pressure']
GRAD5['q_3'] = epsilon * GRAD5['Pvapor_3']*10 /GRAD5['Pressure']
GRAD6['q_3'] = epsilon * GRAD6['Pvapor_3']*10 /GRAD6['Pressure']
GRAD7['q_3'] = epsilon * GRAD7['Pvapor_3']*10 /GRAD7['Pressure']
GRAD8['q_3'] = epsilon * GRAD8['Pvapor_3']*10 /GRAD8['Pressure']
GRAD9['q_3'] = epsilon * GRAD9['Pvapor_3']*10 /GRAD9['Pressure']
#------------------------------------------------------------------------------
GRAD1['q_4'] = epsilon * GRAD1['Pvapor_4']*10 /GRAD1['Pressure']
GRAD2['q_4'] = epsilon * GRAD2['Pvapor_4']*10 /GRAD2['Pressure']
GRAD3['q_4'] = epsilon * GRAD3['Pvapor_4']*10 /GRAD3['Pressure']
GRAD4['q_4'] = epsilon * GRAD4['Pvapor_4']*10 /GRAD4['Pressure']
GRAD5['q_4'] = epsilon * GRAD5['Pvapor_4']*10 /GRAD5['Pressure']
GRAD6['q_4'] = epsilon * GRAD6['Pvapor_4']*10 /GRAD6['Pressure']
GRAD7['q_4'] = epsilon * GRAD7['Pvapor_4']*10 /GRAD7['Pressure']
GRAD8['q_4'] = epsilon * GRAD8['Pvapor_4']*10 /GRAD8['Pressure']
GRAD9['q_4'] = epsilon * GRAD9['Pvapor_4']*10 /GRAD9['Pressure']
#------------------------------------------------------------------------------
GRAD1['q_5'] = epsilon * GRAD1['Pvapor_5']*10 /GRAD1['Pressure']
GRAD2['q_5'] = epsilon * GRAD2['Pvapor_5']*10 /GRAD2['Pressure']
GRAD3['q_5'] = epsilon * GRAD3['Pvapor_5']*10 /GRAD3['Pressure']
GRAD4['q_5'] = epsilon * GRAD4['Pvapor_5']*10 /GRAD4['Pressure']
GRAD5['q_5'] = epsilon * GRAD5['Pvapor_5']*10 /GRAD5['Pressure']
GRAD6['q_5'] = epsilon * GRAD6['Pvapor_5']*10 /GRAD6['Pressure']
GRAD7['q_5'] = epsilon * GRAD7['Pvapor_5']*10 /GRAD7['Pressure']
GRAD8['q_5'] = epsilon * GRAD8['Pvapor_5']*10 /GRAD8['Pressure']
GRAD9['q_5'] = epsilon * GRAD9['Pvapor_5']*10 /GRAD9['Pressure']
#------------------------------------------------------------------------------
#%%计算梯度dudz dtdz dqdz
FLUX1['dudz_5']  = (GRAD1['WS_5'] - GRAD1['WS_1'] )/(GRAD1['z5']-GRAD1['z1'])
FLUX2['dudz_5']  = (GRAD2['WS_5'] - GRAD2['WS_1'] )/(GRAD2['z5']-GRAD2['z1'])
FLUX3['dudz_5']  = (GRAD3['WS_5'] - GRAD3['WS_1'] )/(GRAD3['z5']-GRAD3['z1'])
FLUX4['dudz_5']  = (GRAD4['WS_5'] - GRAD4['WS_1'] )/(GRAD4['z5']-GRAD4['z1'])
FLUX5['dudz_5']  = (GRAD5['WS_5'] - GRAD5['WS_1'] )/(GRAD5['z5']-GRAD5['z1'])
FLUX6['dudz_5']  = (GRAD6['WS_5'] - GRAD6['WS_1'] )/(GRAD6['z5']-GRAD6['z1'])
FLUX7['dudz_5']  = (GRAD7['WS_5'] - GRAD7['WS_1'] )/(GRAD7['z5']-GRAD7['z1'])
FLUX8['dudz_5']  = (GRAD8['WS_5'] - GRAD8['WS_1'] )/(GRAD8['z5']-GRAD8['z1'])
FLUX9['dudz_5']  = (GRAD9['WS_5'] - GRAD9['WS_1'] )/(GRAD9['z5']-GRAD9['z1'])

FLUX1['dudz_4']  = (GRAD1['WS_4'] - GRAD1['WS_1'] )/(GRAD1['z4']-GRAD1['z1'])
FLUX2['dudz_4']  = (GRAD2['WS_4'] - GRAD2['WS_1'] )/(GRAD2['z4']-GRAD2['z1'])
FLUX3['dudz_4']  = (GRAD3['WS_4'] - GRAD3['WS_1'] )/(GRAD3['z4']-GRAD3['z1'])
FLUX4['dudz_4']  = (GRAD4['WS_4'] - GRAD4['WS_1'] )/(GRAD4['z4']-GRAD4['z1'])
FLUX5['dudz_4']  = (GRAD5['WS_4'] - GRAD5['WS_1'] )/(GRAD5['z4']-GRAD5['z1'])
FLUX6['dudz_4']  = (GRAD6['WS_4'] - GRAD6['WS_1'] )/(GRAD6['z4']-GRAD6['z1'])
FLUX7['dudz_4']  = (GRAD7['WS_4'] - GRAD7['WS_1'] )/(GRAD7['z4']-GRAD7['z1'])
FLUX8['dudz_4']  = (GRAD8['WS_4'] - GRAD8['WS_1'] )/(GRAD8['z4']-GRAD8['z1'])
FLUX9['dudz_4']  = (GRAD9['WS_4'] - GRAD9['WS_1'] )/(GRAD9['z4']-GRAD9['z1'])

FLUX1['dudz_3']  = (GRAD1['WS_3'] - GRAD1['WS_1'] )/(GRAD1['z3']-GRAD1['z1'])
FLUX2['dudz_3']  = (GRAD2['WS_3'] - GRAD2['WS_1'] )/(GRAD2['z3']-GRAD2['z1'])
FLUX3['dudz_3']  = (GRAD3['WS_3'] - GRAD3['WS_1'] )/(GRAD3['z3']-GRAD3['z1'])
FLUX4['dudz_3']  = (GRAD4['WS_3'] - GRAD4['WS_1'] )/(GRAD4['z3']-GRAD4['z1'])
FLUX5['dudz_3']  = (GRAD5['WS_3'] - GRAD5['WS_1'] )/(GRAD5['z3']-GRAD5['z1'])
FLUX6['dudz_3']  = (GRAD6['WS_3'] - GRAD6['WS_1'] )/(GRAD6['z3']-GRAD6['z1'])
FLUX7['dudz_3']  = (GRAD7['WS_3'] - GRAD7['WS_1'] )/(GRAD7['z3']-GRAD7['z1'])
FLUX8['dudz_3']  = (GRAD8['WS_3'] - GRAD8['WS_1'] )/(GRAD8['z3']-GRAD8['z1'])
FLUX9['dudz_3']  = (GRAD9['WS_3'] - GRAD9['WS_1'] )/(GRAD9['z3']-GRAD9['z1'])

FLUX1['dudz_2']  = (GRAD1['WS_2'] - GRAD1['WS_1'] )/(GRAD1['z2']-GRAD1['z1'])
FLUX2['dudz_2']  = (GRAD2['WS_2'] - GRAD2['WS_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX3['dudz_2']  = (GRAD3['WS_2'] - GRAD3['WS_1'] )/(GRAD3['z2']-GRAD2['z1'])
FLUX4['dudz_2']  = (GRAD4['WS_2'] - GRAD4['WS_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dudz_2']  = (GRAD5['WS_2'] - GRAD5['WS_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dudz_2']  = (GRAD6['WS_2'] - GRAD6['WS_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dudz_2']  = (GRAD7['WS_2'] - GRAD7['WS_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dudz_2']  = (GRAD8['WS_2'] - GRAD8['WS_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dudz_2']  = (GRAD9['WS_2'] - GRAD9['WS_1'] )/(GRAD9['z2']-GRAD9['z1'])

FLUX1['dudz']  = (FLUX1['dudz_2'] +FLUX1['dudz_3'] +FLUX1['dudz_4'] +FLUX1['dudz_5'] )/4
FLUX2['dudz']  = (FLUX2['dudz_2'] +FLUX2['dudz_3'] +FLUX2['dudz_4'] +FLUX2['dudz_5'] )/4
FLUX3['dudz']  = (FLUX3['dudz_2'] +FLUX3['dudz_3'] +FLUX3['dudz_4'] +FLUX3['dudz_5'] )/4
FLUX4['dudz']  = (FLUX4['dudz_2'] +FLUX4['dudz_3'] +FLUX4['dudz_4'] +FLUX4['dudz_5'] )/4
FLUX5['dudz']  = (FLUX5['dudz_2'] +FLUX5['dudz_3'] +FLUX5['dudz_4'] +FLUX5['dudz_5'] )/4
FLUX6['dudz']  = (FLUX6['dudz_2'] +FLUX6['dudz_3'] +FLUX6['dudz_4'] +FLUX6['dudz_5'] )/4
FLUX7['dudz']  = (FLUX7['dudz_2'] +FLUX7['dudz_3'] +FLUX7['dudz_4'] +FLUX7['dudz_5'] )/4
FLUX8['dudz']  = (FLUX8['dudz_2'] +FLUX8['dudz_3'] +FLUX8['dudz_4'] +FLUX8['dudz_5'] )/4
FLUX9['dudz']  = (FLUX9['dudz_2'] +FLUX9['dudz_3'] +FLUX9['dudz_4'] +FLUX9['dudz_5'] )/4

FLUX1['dudz_5_4']  = (GRAD1['WS_5'] - GRAD1['WS_4'] )/(GRAD1['z5']-GRAD1['z4'])
FLUX2['dudz_5_4']  = (GRAD2['WS_5'] - GRAD2['WS_4'] )/(GRAD2['z5']-GRAD2['z4'])
FLUX3['dudz_5_4']  = (GRAD3['WS_5'] - GRAD3['WS_4'] )/(GRAD3['z5']-GRAD3['z4'])
FLUX4['dudz_5_4']  = (GRAD4['WS_5'] - GRAD4['WS_4'] )/(GRAD4['z5']-GRAD4['z4'])
FLUX5['dudz_5_4']  = (GRAD5['WS_5'] - GRAD5['WS_4'] )/(GRAD5['z5']-GRAD5['z4'])
FLUX6['dudz_5_4']  = (GRAD6['WS_5'] - GRAD6['WS_4'] )/(GRAD6['z5']-GRAD6['z4'])
FLUX7['dudz_5_4']  = (GRAD7['WS_5'] - GRAD7['WS_4'] )/(GRAD7['z5']-GRAD7['z4'])
FLUX8['dudz_5_4']  = (GRAD8['WS_5'] - GRAD8['WS_4'] )/(GRAD8['z5']-GRAD8['z4'])
FLUX9['dudz_5_4']  = (GRAD9['WS_5'] - GRAD9['WS_4'] )/(GRAD9['z5']-GRAD9['z4'])

FLUX1['dudz_4_3']  = (GRAD1['WS_4'] - GRAD1['WS_3'] )/(GRAD1['z4']-GRAD1['z3'])
FLUX2['dudz_4_3']  = (GRAD2['WS_4'] - GRAD2['WS_3'] )/(GRAD2['z4']-GRAD2['z3'])
FLUX3['dudz_4_3']  = (GRAD3['WS_4'] - GRAD3['WS_3'] )/(GRAD3['z4']-GRAD3['z3'])
FLUX4['dudz_4_3']  = (GRAD4['WS_4'] - GRAD4['WS_3'] )/(GRAD4['z4']-GRAD4['z3'])
FLUX5['dudz_4_3']  = (GRAD5['WS_4'] - GRAD5['WS_3'] )/(GRAD5['z4']-GRAD5['z3'])
FLUX6['dudz_4_3']  = (GRAD6['WS_4'] - GRAD6['WS_3'] )/(GRAD6['z4']-GRAD6['z3'])
FLUX7['dudz_4_3']  = (GRAD7['WS_4'] - GRAD7['WS_3'] )/(GRAD7['z4']-GRAD7['z3'])
FLUX8['dudz_4_3']  = (GRAD8['WS_4'] - GRAD8['WS_3'] )/(GRAD8['z4']-GRAD8['z3'])
FLUX9['dudz_4_3']  = (GRAD9['WS_4'] - GRAD9['WS_3'] )/(GRAD9['z4']-GRAD9['z3'])

FLUX1['dudz_3_2']  = (GRAD1['WS_3'] - GRAD1['WS_2'] )/(GRAD1['z3']-GRAD1['z2'])
FLUX2['dudz_3_2']  = (GRAD2['WS_3'] - GRAD2['WS_2'] )/(GRAD2['z3']-GRAD2['z2'])
FLUX3['dudz_3_2']  = (GRAD3['WS_3'] - GRAD3['WS_2'] )/(GRAD3['z3']-GRAD3['z2'])
FLUX4['dudz_3_2']  = (GRAD4['WS_3'] - GRAD4['WS_2'] )/(GRAD4['z3']-GRAD4['z2'])
FLUX5['dudz_3_2']  = (GRAD5['WS_3'] - GRAD5['WS_2'] )/(GRAD5['z3']-GRAD5['z2'])
FLUX6['dudz_3_2']  = (GRAD6['WS_3'] - GRAD6['WS_2'] )/(GRAD6['z3']-GRAD6['z2'])
FLUX7['dudz_3_2']  = (GRAD7['WS_3'] - GRAD7['WS_2'] )/(GRAD7['z3']-GRAD7['z2'])
FLUX8['dudz_3_2']  = (GRAD8['WS_3'] - GRAD8['WS_2'] )/(GRAD8['z3']-GRAD8['z2'])
FLUX9['dudz_3_2']  = (GRAD9['WS_3'] - GRAD9['WS_2'] )/(GRAD9['z3']-GRAD9['z2'])

FLUX1['dudz_2_1']  = (GRAD1['WS_2'] - GRAD1['WS_1'] )/(GRAD1['z2']-GRAD1['z1'])
FLUX2['dudz_2_1']  = (GRAD2['WS_2'] - GRAD2['WS_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX3['dudz_2_1']  = (GRAD3['WS_2'] - GRAD3['WS_1'] )/(GRAD3['z2']-GRAD2['z1'])
FLUX4['dudz_2_1']  = (GRAD4['WS_2'] - GRAD4['WS_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dudz_2_1']  = (GRAD5['WS_2'] - GRAD5['WS_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dudz_2_1']  = (GRAD6['WS_2'] - GRAD6['WS_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dudz_2_1']  = (GRAD7['WS_2'] - GRAD7['WS_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dudz_2_1']  = (GRAD8['WS_2'] - GRAD8['WS_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dudz_2_1']  = (GRAD9['WS_2'] - GRAD9['WS_1'] )/(GRAD9['z2']-GRAD9['z1'])



FLUX1['dtdz_5']  = (GRAD1['Ta_5'] - GRAD1['Ta_1'] )/(GRAD1['z5']-GRAD1['z1'])
FLUX2['dtdz_5']  = (GRAD2['Ta_5'] - GRAD2['Ta_1'] )/(GRAD2['z5']-GRAD2['z1'])
FLUX3['dtdz_5']  = (GRAD3['Ta_5'] - GRAD3['Ta_1'] )/(GRAD3['z5']-GRAD3['z1'])
FLUX4['dtdz_5']  = (GRAD4['Ta_5'] - GRAD4['Ta_1'] )/(GRAD4['z5']-GRAD4['z1'])
FLUX5['dtdz_5']  = (GRAD5['Ta_5'] - GRAD5['Ta_1'] )/(GRAD5['z5']-GRAD5['z1'])
FLUX6['dtdz_5']  = (GRAD6['Ta_5'] - GRAD6['Ta_1'] )/(GRAD6['z5']-GRAD6['z1'])
FLUX7['dtdz_5']  = (GRAD7['Ta_5'] - GRAD7['Ta_1'] )/(GRAD7['z5']-GRAD7['z1'])
FLUX8['dtdz_5']  = (GRAD8['Ta_5'] - GRAD8['Ta_1'] )/(GRAD8['z5']-GRAD8['z1'])
FLUX9['dtdz_5']  = (GRAD9['Ta_5'] - GRAD9['Ta_1'] )/(GRAD9['z5']-GRAD9['z1'])

FLUX1['dtdz_4']  = (GRAD1['Ta_4'] - GRAD1['Ta_1'] )/(GRAD1['z4']-GRAD1['z1'])
FLUX2['dtdz_4']  = (GRAD2['Ta_4'] - GRAD2['Ta_1'] )/(GRAD2['z4']-GRAD2['z1'])
FLUX3['dtdz_4']  = (GRAD3['Ta_4'] - GRAD3['Ta_1'] )/(GRAD3['z4']-GRAD3['z1'])
FLUX4['dtdz_4']  = (GRAD4['Ta_4'] - GRAD4['Ta_1'] )/(GRAD4['z4']-GRAD4['z1'])
FLUX5['dtdz_4']  = (GRAD5['Ta_4'] - GRAD5['Ta_1'] )/(GRAD5['z4']-GRAD5['z1'])
FLUX6['dtdz_4']  = (GRAD6['Ta_4'] - GRAD6['Ta_1'] )/(GRAD6['z4']-GRAD6['z1'])
FLUX7['dtdz_4']  = (GRAD7['Ta_4'] - GRAD7['Ta_1'] )/(GRAD7['z4']-GRAD7['z1'])
FLUX8['dtdz_4']  = (GRAD8['Ta_4'] - GRAD8['Ta_1'] )/(GRAD8['z4']-GRAD8['z1'])
FLUX9['dtdz_4']  = (GRAD9['Ta_4'] - GRAD9['Ta_1'] )/(GRAD9['z4']-GRAD9['z1'])

FLUX1['dtdz_3']  = (GRAD1['Ta_3'] - GRAD1['Ta_1'] )/(GRAD1['z3']-GRAD1['z1'])
FLUX2['dtdz_3']  = (GRAD2['Ta_3'] - GRAD2['Ta_1'] )/(GRAD2['z3']-GRAD2['z1'])
FLUX3['dtdz_3']  = (GRAD3['Ta_3'] - GRAD3['Ta_1'] )/(GRAD3['z3']-GRAD3['z1'])
FLUX4['dtdz_3']  = (GRAD4['Ta_3'] - GRAD4['Ta_1'] )/(GRAD4['z3']-GRAD4['z1'])
FLUX5['dtdz_3']  = (GRAD5['Ta_3'] - GRAD5['Ta_1'] )/(GRAD5['z3']-GRAD5['z1'])
FLUX6['dtdz_3']  = (GRAD6['Ta_3'] - GRAD6['Ta_1'] )/(GRAD6['z3']-GRAD6['z1'])
FLUX7['dtdz_3']  = (GRAD7['Ta_3'] - GRAD7['Ta_1'] )/(GRAD7['z3']-GRAD7['z1'])
FLUX8['dtdz_3']  = (GRAD8['Ta_3'] - GRAD8['Ta_1'] )/(GRAD8['z3']-GRAD8['z1'])
FLUX9['dtdz_3']  = (GRAD9['Ta_3'] - GRAD9['Ta_1'] )/(GRAD9['z3']-GRAD9['z1'])

FLUX1['dtdz_2']  = (GRAD1['Ta_2'] - GRAD1['Ta_1'] )/(GRAD1['z2']-GRAD1['z1'])
FLUX2['dtdz_2']  = (GRAD2['Ta_2'] - GRAD2['Ta_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX3['dtdz_2']  = (GRAD3['Ta_2'] - GRAD3['Ta_1'] )/(GRAD3['z2']-GRAD3['z1'])
FLUX4['dtdz_2']  = (GRAD4['Ta_2'] - GRAD4['Ta_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dtdz_2']  = (GRAD5['Ta_2'] - GRAD5['Ta_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dtdz_2']  = (GRAD6['Ta_2'] - GRAD6['Ta_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dtdz_2']  = (GRAD7['Ta_2'] - GRAD7['Ta_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dtdz_2']  = (GRAD8['Ta_2'] - GRAD8['Ta_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dtdz_2']  = (GRAD9['Ta_2'] - GRAD9['Ta_1'] )/(GRAD9['z2']-GRAD9['z1'])



FLUX1['dtdz_5_4']  = (GRAD1['Ta_5'] - GRAD1['Ta_4'] )/(GRAD1['z5']-GRAD1['z4'])
FLUX2['dtdz_5_4']  = (GRAD2['Ta_5'] - GRAD2['Ta_4'] )/(GRAD2['z5']-GRAD2['z4'])
FLUX3['dtdz_5_4']  = (GRAD3['Ta_5'] - GRAD3['Ta_4'] )/(GRAD3['z5']-GRAD3['z4'])
FLUX4['dtdz_5_4']  = (GRAD4['Ta_5'] - GRAD4['Ta_4'] )/(GRAD4['z5']-GRAD4['z4'])
FLUX5['dtdz_5_4']  = (GRAD5['Ta_5'] - GRAD5['Ta_4'] )/(GRAD5['z5']-GRAD5['z4'])
FLUX6['dtdz_5_4']  = (GRAD6['Ta_5'] - GRAD6['Ta_4'] )/(GRAD6['z5']-GRAD6['z4'])
FLUX7['dtdz_5_4']  = (GRAD7['Ta_5'] - GRAD7['Ta_4'] )/(GRAD7['z5']-GRAD7['z4'])
FLUX8['dtdz_5_4']  = (GRAD8['Ta_5'] - GRAD8['Ta_4'] )/(GRAD8['z5']-GRAD8['z4'])
FLUX9['dtdz_5_4']  = (GRAD9['Ta_5'] - GRAD9['Ta_4'] )/(GRAD9['z5']-GRAD9['z4'])

FLUX1['dtdz_4_3']  = (GRAD1['Ta_4'] - GRAD1['Ta_3'] )/(GRAD1['z4']-GRAD1['z3'])
FLUX2['dtdz_4_3']  = (GRAD2['Ta_4'] - GRAD2['Ta_3'] )/(GRAD2['z4']-GRAD2['z3'])
FLUX3['dtdz_4_3']  = (GRAD3['Ta_4'] - GRAD3['Ta_3'] )/(GRAD3['z4']-GRAD3['z3'])
FLUX4['dtdz_4_3']  = (GRAD4['Ta_4'] - GRAD4['Ta_3'] )/(GRAD4['z4']-GRAD4['z3'])
FLUX5['dtdz_4_3']  = (GRAD5['Ta_4'] - GRAD5['Ta_3'] )/(GRAD5['z4']-GRAD5['z3'])
FLUX6['dtdz_4_3']  = (GRAD6['Ta_4'] - GRAD6['Ta_3'] )/(GRAD6['z4']-GRAD6['z3'])
FLUX7['dtdz_4_3']  = (GRAD7['Ta_4'] - GRAD7['Ta_3'] )/(GRAD7['z4']-GRAD7['z3'])
FLUX8['dtdz_4_3']  = (GRAD8['Ta_4'] - GRAD8['Ta_3'] )/(GRAD8['z4']-GRAD8['z3'])
FLUX9['dtdz_4_3']  = (GRAD9['Ta_4'] - GRAD9['Ta_3'] )/(GRAD9['z4']-GRAD9['z3'])

FLUX1['dtdz_3_2']  = (GRAD1['Ta_3'] - GRAD1['Ta_2'] )/(GRAD1['z3']-GRAD1['z2'])
FLUX2['dtdz_3_2']  = (GRAD2['Ta_3'] - GRAD2['Ta_2'] )/(GRAD2['z3']-GRAD2['z2'])
FLUX3['dtdz_3_2']  = (GRAD3['Ta_3'] - GRAD3['Ta_2'] )/(GRAD3['z3']-GRAD3['z2'])
FLUX4['dtdz_3_2']  = (GRAD4['Ta_3'] - GRAD4['Ta_2'] )/(GRAD4['z3']-GRAD4['z2'])
FLUX5['dtdz_3_2']  = (GRAD5['Ta_3'] - GRAD5['Ta_2'] )/(GRAD5['z3']-GRAD5['z2'])
FLUX6['dtdz_3_2']  = (GRAD6['Ta_3'] - GRAD6['Ta_2'] )/(GRAD6['z3']-GRAD6['z2'])
FLUX7['dtdz_3_2']  = (GRAD7['Ta_3'] - GRAD7['Ta_2'] )/(GRAD7['z3']-GRAD7['z2'])
FLUX8['dtdz_3_2']  = (GRAD8['Ta_3'] - GRAD8['Ta_2'] )/(GRAD8['z3']-GRAD8['z2'])
FLUX9['dtdz_3_2']  = (GRAD9['Ta_3'] - GRAD9['Ta_2'] )/(GRAD9['z3']-GRAD9['z2'])

FLUX1['dtdz_2_1']  = (GRAD1['Ta_2'] - GRAD1['Ta_1'] )/(GRAD1['z2']-GRAD1['z1'])
FLUX2['dtdz_2_1']  = (GRAD2['Ta_2'] - GRAD2['Ta_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX3['dtdz_2_1']  = (GRAD3['Ta_2'] - GRAD3['Ta_1'] )/(GRAD3['z2']-GRAD2['z1'])
FLUX4['dtdz_2_1']  = (GRAD4['Ta_2'] - GRAD4['Ta_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dtdz_2_1']  = (GRAD5['Ta_2'] - GRAD5['Ta_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dtdz_2_1']  = (GRAD6['Ta_2'] - GRAD6['Ta_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dtdz_2_1']  = (GRAD7['Ta_2'] - GRAD7['Ta_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dtdz_2_1']  = (GRAD8['Ta_2'] - GRAD8['Ta_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dtdz_2_1']  = (GRAD9['Ta_2'] - GRAD9['Ta_1'] )/(GRAD9['z2']-GRAD9['z1'])


FLUX1['dtdz']  = (FLUX1['dtdz_2'] +FLUX1['dtdz_3'] +FLUX1['dtdz_4'] +FLUX1['dtdz_5'] )/4
FLUX2['dtdz']  = (FLUX2['dtdz_2'] +FLUX2['dtdz_3'] +FLUX2['dtdz_4'] +FLUX2['dtdz_5'] )/4
FLUX3['dtdz']  = (FLUX3['dtdz_2'] +FLUX3['dtdz_3'] +FLUX3['dtdz_4'] +FLUX3['dtdz_5'] )/4
FLUX4['dtdz']  = (FLUX4['dtdz_2'] +FLUX4['dtdz_3'] +FLUX4['dtdz_4'] +FLUX4['dtdz_5'] )/4
FLUX5['dtdz']  = (FLUX5['dtdz_2'] +FLUX5['dtdz_3'] +FLUX5['dtdz_4'] +FLUX5['dtdz_5'] )/4
FLUX6['dtdz']  = (FLUX6['dtdz_2'] +FLUX6['dtdz_3'] +FLUX6['dtdz_4'] +FLUX6['dtdz_5'] )/4
FLUX7['dtdz']  = (FLUX7['dtdz_2'] +FLUX7['dtdz_3'] +FLUX7['dtdz_4'] +FLUX7['dtdz_5'] )/4
FLUX8['dtdz']  = (FLUX8['dtdz_2'] +FLUX8['dtdz_3'] +FLUX8['dtdz_4'] +FLUX8['dtdz_5'] )/4
FLUX9['dtdz']  = (FLUX9['dtdz_2'] +FLUX9['dtdz_3'] +FLUX9['dtdz_4'] +FLUX9['dtdz_5'] )/4


FLUX1['dqdz_5']  = (GRAD1['q_5'] - GRAD1['q_1'] )/(GRAD1['z5']-GRAD1['z1'])
FLUX2['dqdz_5']  = (GRAD2['q_5'] - GRAD2['q_1'] )/(GRAD2['z5']-GRAD2['z1'])
FLUX3['dqdz_5']  = (GRAD3['q_5'] - GRAD3['q_1'] )/(GRAD3['z5']-GRAD3['z1'])
FLUX4['dqdz_5']  = (GRAD4['q_5'] - GRAD4['q_1'] )/(GRAD4['z5']-GRAD4['z1'])
FLUX5['dqdz_5']  = (GRAD5['q_5'] - GRAD5['q_1'] )/(GRAD5['z5']-GRAD5['z1'])
FLUX6['dqdz_5']  = (GRAD6['q_5'] - GRAD6['q_1'] )/(GRAD6['z5']-GRAD6['z1'])
FLUX7['dqdz_5']  = (GRAD7['q_5'] - GRAD7['q_1'] )/(GRAD7['z5']-GRAD7['z1'])
FLUX8['dqdz_5']  = (GRAD8['q_5'] - GRAD8['q_1'] )/(GRAD8['z5']-GRAD8['z1'])
FLUX9['dqdz_5']  = (GRAD9['q_5'] - GRAD9['q_1'] )/(GRAD9['z5']-GRAD9['z1'])

FLUX1['dqdz_4']  = (GRAD1['q_4'] - GRAD1['q_1'] )/(GRAD1['z4']-GRAD1['z1'])
FLUX2['dqdz_4']  = (GRAD2['q_4'] - GRAD2['q_1'] )/(GRAD2['z4']-GRAD2['z1'])
FLUX3['dqdz_4']  = (GRAD3['q_4'] - GRAD3['q_1'] )/(GRAD3['z4']-GRAD3['z1'])
FLUX4['dqdz_4']  = (GRAD4['q_4'] - GRAD4['q_1'] )/(GRAD4['z4']-GRAD4['z1'])
FLUX5['dqdz_4']  = (GRAD5['q_4'] - GRAD5['q_1'] )/(GRAD5['z4']-GRAD5['z1'])
FLUX6['dqdz_4']  = (GRAD6['q_4'] - GRAD6['q_1'] )/(GRAD6['z4']-GRAD6['z1'])
FLUX7['dqdz_4']  = (GRAD7['q_4'] - GRAD7['q_1'] )/(GRAD7['z4']-GRAD7['z1'])
FLUX8['dqdz_4']  = (GRAD8['q_4'] - GRAD8['q_1'] )/(GRAD8['z4']-GRAD8['z1'])
FLUX9['dqdz_4']  = (GRAD9['q_4'] - GRAD9['q_1'] )/(GRAD9['z4']-GRAD9['z1'])

FLUX1['dqdz_3']  = (GRAD1['q_3'] - GRAD1['q_1'] )/(GRAD1['z3']-GRAD1['z1'])
FLUX2['dqdz_3']  = (GRAD2['q_3'] - GRAD2['q_1'] )/(GRAD2['z3']-GRAD2['z1'])
FLUX3['dqdz_3']  = (GRAD3['q_3'] - GRAD3['q_1'] )/(GRAD3['z3']-GRAD3['z1'])
FLUX4['dqdz_3']  = (GRAD4['q_3'] - GRAD4['q_1'] )/(GRAD4['z3']-GRAD4['z1'])
FLUX5['dqdz_3']  = (GRAD5['q_3'] - GRAD5['q_1'] )/(GRAD5['z3']-GRAD5['z1'])
FLUX6['dqdz_3']  = (GRAD6['q_3'] - GRAD6['q_1'] )/(GRAD6['z3']-GRAD6['z1'])
FLUX7['dqdz_3']  = (GRAD7['q_3'] - GRAD7['q_1'] )/(GRAD7['z3']-GRAD7['z1'])
FLUX8['dqdz_3']  = (GRAD8['q_3'] - GRAD8['q_1'] )/(GRAD8['z3']-GRAD8['z1'])
FLUX9['dqdz_3']  = (GRAD9['q_3'] - GRAD9['q_1'] )/(GRAD9['z3']-GRAD9['z1'])

FLUX1['dqdz_2']  = (GRAD1['q_2'] - GRAD1['q_1'] )/(GRAD1['z2']-GRAD1['z1'])
FLUX2['dqdz_2']  = (GRAD2['q_2'] - GRAD2['q_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX3['dqdz_2']  = (GRAD3['q_2'] - GRAD3['q_1'] )/(GRAD3['z2']-GRAD2['z1'])
FLUX4['dqdz_2']  = (GRAD4['q_2'] - GRAD4['q_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dqdz_2']  = (GRAD5['q_2'] - GRAD5['q_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dqdz_2']  = (GRAD6['q_2'] - GRAD6['q_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dqdz_2']  = (GRAD7['q_2'] - GRAD7['q_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dqdz_2']  = (GRAD8['q_2'] - GRAD8['q_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dqdz_2']  = (GRAD9['q_2'] - GRAD9['q_1'] )/(GRAD9['z2']-GRAD9['z1'])


FLUX1['dqdz']  = (FLUX1['dqdz_2'] +FLUX1['dqdz_3'] +FLUX1['dqdz_4'] +FLUX1['dqdz_5'] )/4
FLUX2['dqdz']  = (FLUX2['dqdz_2'] +FLUX2['dqdz_3'] +FLUX2['dqdz_4'] +FLUX2['dqdz_5'] )/4
FLUX3['dqdz']  = (FLUX3['dqdz_2'] +FLUX3['dqdz_3'] +FLUX3['dqdz_4'] +FLUX3['dqdz_5'] )/4
FLUX4['dqdz']  = (FLUX4['dqdz_2'] +FLUX4['dqdz_3'] +FLUX4['dqdz_4'] +FLUX4['dqdz_5'] )/4
FLUX5['dqdz']  = (FLUX5['dqdz_2'] +FLUX5['dqdz_3'] +FLUX5['dqdz_4'] +FLUX5['dqdz_5'] )/4
FLUX6['dqdz']  = (FLUX6['dqdz_2'] +FLUX6['dqdz_3'] +FLUX6['dqdz_4'] +FLUX6['dqdz_5'] )/4
FLUX7['dqdz']  = (FLUX7['dqdz_2'] +FLUX7['dqdz_3'] +FLUX7['dqdz_4'] +FLUX7['dqdz_5'] )/4
FLUX8['dqdz']  = (FLUX8['dqdz_2'] +FLUX8['dqdz_3'] +FLUX8['dqdz_4'] +FLUX8['dqdz_5'] )/4
FLUX9['dqdz']  = (FLUX9['dqdz_2'] +FLUX9['dqdz_3'] +FLUX9['dqdz_4'] +FLUX9['dqdz_5'] )/4


FLUX1['dqdz_5_4']  = (GRAD1['q_5'] - GRAD1['q_4'] )/(GRAD1['z5']-GRAD1['z4'])
FLUX2['dqdz_5_4']  = (GRAD2['q_5'] - GRAD2['q_4'] )/(GRAD2['z5']-GRAD2['z4'])
FLUX3['dqdz_5_4']  = (GRAD3['q_5'] - GRAD3['q_4'] )/(GRAD3['z5']-GRAD3['z4'])
FLUX4['dqdz_5_4']  = (GRAD4['q_5'] - GRAD4['q_4'] )/(GRAD4['z5']-GRAD4['z4'])
FLUX5['dqdz_5_4']  = (GRAD5['q_5'] - GRAD5['q_4'] )/(GRAD5['z5']-GRAD5['z4'])
FLUX6['dqdz_5_4']  = (GRAD6['q_5'] - GRAD6['q_4'] )/(GRAD6['z5']-GRAD6['z4'])
FLUX7['dqdz_5_4']  = (GRAD7['q_5'] - GRAD7['q_4'] )/(GRAD7['z5']-GRAD7['z4'])
FLUX8['dqdz_5_4']  = (GRAD8['q_5'] - GRAD8['q_4'] )/(GRAD8['z5']-GRAD8['z4'])
FLUX9['dqdz_5_4']  = (GRAD9['q_5'] - GRAD9['q_4'] )/(GRAD9['z5']-GRAD9['z4'])

FLUX1['dqdz_4_3']  = (GRAD1['q_4'] - GRAD1['q_3'] )/(GRAD1['z4']-GRAD1['z3'])
FLUX2['dqdz_4_3']  = (GRAD2['q_4'] - GRAD2['q_3'] )/(GRAD2['z4']-GRAD2['z3'])
FLUX3['dqdz_4_3']  = (GRAD3['q_4'] - GRAD3['q_3'] )/(GRAD3['z4']-GRAD3['z3'])
FLUX4['dqdz_4_3']  = (GRAD4['q_4'] - GRAD4['q_3'] )/(GRAD4['z4']-GRAD4['z3'])
FLUX5['dqdz_4_3']  = (GRAD5['q_4'] - GRAD5['q_3'] )/(GRAD5['z4']-GRAD5['z3'])
FLUX6['dqdz_4_3']  = (GRAD6['q_4'] - GRAD6['q_3'] )/(GRAD6['z4']-GRAD6['z3'])
FLUX7['dqdz_4_3']  = (GRAD7['q_4'] - GRAD7['q_3'] )/(GRAD7['z4']-GRAD7['z3'])
FLUX8['dqdz_4_3']  = (GRAD8['q_4'] - GRAD8['q_3'] )/(GRAD8['z4']-GRAD8['z3'])
FLUX9['dqdz_4_3']  = (GRAD9['q_4'] - GRAD9['q_3'] )/(GRAD9['z4']-GRAD9['z3'])

FLUX1['dqdz_3_2']  = (GRAD1['q_3'] - GRAD1['q_2'] )/(GRAD1['z3']-GRAD1['z2'])
FLUX2['dqdz_3_2']  = (GRAD2['q_3'] - GRAD2['q_2'] )/(GRAD2['z3']-GRAD2['z2'])
FLUX3['dqdz_3_2']  = (GRAD3['q_3'] - GRAD3['q_2'] )/(GRAD3['z3']-GRAD3['z2'])
FLUX4['dqdz_3_2']  = (GRAD4['q_3'] - GRAD4['q_2'] )/(GRAD4['z3']-GRAD4['z2'])
FLUX5['dqdz_3_2']  = (GRAD5['q_3'] - GRAD5['q_2'] )/(GRAD5['z3']-GRAD5['z2'])
FLUX6['dqdz_3_2']  = (GRAD6['q_3'] - GRAD6['q_2'] )/(GRAD6['z3']-GRAD6['z2'])
FLUX7['dqdz_3_2']  = (GRAD7['q_3'] - GRAD7['q_2'] )/(GRAD7['z3']-GRAD7['z2'])
FLUX8['dqdz_3_2']  = (GRAD8['q_3'] - GRAD8['q_2'] )/(GRAD8['z3']-GRAD8['z2'])
FLUX9['dqdz_3_2']  = (GRAD9['q_3'] - GRAD9['q_2'] )/(GRAD9['z3']-GRAD9['z2'])

FLUX1['dqdz_2_1']  = (GRAD1['q_2'] - GRAD1['q_1'] )/(GRAD1['z2']-GRAD1['z1'])
FLUX2['dqdz_2_1']  = (GRAD2['q_2'] - GRAD2['q_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX3['dqdz_2_1']  = (GRAD3['q_2'] - GRAD3['q_1'] )/(GRAD3['z2']-GRAD2['z1'])
FLUX4['dqdz_2_1']  = (GRAD4['q_2'] - GRAD4['q_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dqdz_2_1']  = (GRAD5['q_2'] - GRAD5['q_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dqdz_2_1']  = (GRAD6['q_2'] - GRAD6['q_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dqdz_2_1']  = (GRAD7['q_2'] - GRAD7['q_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dqdz_2_1']  = (GRAD8['q_2'] - GRAD8['q_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dqdz_2_1']  = (GRAD9['q_2'] - GRAD9['q_1'] )/(GRAD9['z2']-GRAD9['z1'])

#查看每一个高度上的梯度有什么变化。

print('20米高度的速度梯度',np.mean(FLUX1['dudz_5']))
print('10米高度的速度梯度',np.mean(FLUX1['dudz_4']))
print('4米高度的速度梯度',np.mean(FLUX1['dudz_3']))
print('2米高度的速度梯度',np.mean(FLUX1['dudz_2']))


#%%#计算无量纲动量梯度


FLUX1['u_no_dim'] =  kaman * FLUX1['z_obs'] * FLUX1['dudz'] / FLUX1['USTAR'] 
FLUX1['xi'] = GRAD1['z1']/FLUX1['L']
#------------------------------------------------------------------------------
FLUX2['u_no_dim'] =  kaman * FLUX2['z_obs'] * FLUX2['dudz'] / FLUX2['USTAR'] 
FLUX2['xi'] = FLUX2['z_obs']/FLUX2['L']
#------------------------------------------------------------------------------
FLUX3['u_no_dim'] =  kaman * FLUX3['z_obs'] * FLUX3['dudz'] / FLUX3['USTAR'] 
FLUX3['xi'] = FLUX3['z_obs']/FLUX3['L']
#------------------------------------------------------------------------------
FLUX4['u_no_dim'] =  kaman * FLUX4['z_obs'] * FLUX4['dudz'] / FLUX4['USTAR'] 
FLUX4['xi'] = FLUX4['z_obs']/FLUX4['L']
#------------------------------------------------------------------------------
FLUX5['u_no_dim'] =  kaman * FLUX5['z_obs'] * FLUX5['dudz'] / FLUX5['USTAR'] 
FLUX5['xi'] = FLUX5['z_obs']/FLUX5['L']
#------------------------------------------------------------------------------
FLUX6['u_no_dim'] =  kaman * FLUX6['z_obs'] * FLUX6['dudz'] / FLUX6['USTAR'] 
FLUX6['xi'] = FLUX6['z_obs']/FLUX6['L']
#------------------------------------------------------------------------------
FLUX7['u_no_dim'] =  kaman * FLUX7['z_obs'] * FLUX7['dudz'] / FLUX7['USTAR'] 
FLUX7['xi'] = FLUX7['z_obs']/FLUX7['L']
#------------------------------------------------------------------------------

FLUX8['u_no_dim'] =  kaman * FLUX8['z_obs'] * FLUX8['dudz'] / FLUX8['USTAR'] 
FLUX8['xi'] = FLUX8['z_obs']/FLUX8['L']
#------------------------------------------------------------------------------
FLUX9['u_no_dim'] =  kaman * FLUX9['z_obs'] * FLUX9['dudz'] / FLUX9['USTAR'] 
FLUX9['xi'] = FLUX9['z_obs']/FLUX9['L']
#------------------------------------------------------------------------------

FLUX1['u_no_dim_5'] =  kaman * FLUX1['z_obs'] * FLUX1['dudz_5_4'] / FLUX1['USTAR'] 
FLUX1['xi_5'] = GRAD1['z5']/FLUX1['L']

FLUX1['u_no_dim_4'] =  kaman * FLUX1['z_obs'] * FLUX1['dudz_4_3'] / FLUX1['USTAR'] 
FLUX1['xi_4'] = GRAD1['z4']/FLUX1['L']

FLUX1['u_no_dim_3'] =  kaman * FLUX1['z_obs'] * FLUX1['dudz_3_2'] / FLUX1['USTAR'] 
FLUX1['xi_3'] = GRAD1['z3']/FLUX1['L']

FLUX1['u_no_dim_2'] =  kaman * FLUX1['z_obs'] * FLUX1['dudz_2_1'] / FLUX1['USTAR'] 
FLUX1['xi_2'] = GRAD1['z2']/FLUX1['L']

plt.figure()
plt.loglog(FLUX1['xi_5'],FLUX1['u_no_dim_5'],'.',label="5")
plt.loglog(FLUX1['xi_4'],FLUX1['u_no_dim_4'],'.',label="4")
plt.loglog(FLUX1['xi_3'],FLUX1['u_no_dim_3'],'.',label="3")
plt.loglog(FLUX1['xi_2'],FLUX1['u_no_dim_2'],'.',label="2")
plt.legend()
plt.xlim(-1000,1000)
#------------------------------------------------------------------------------
FLUX2['u_no_dim'] =  kaman * FLUX2['z_obs'] * FLUX2['dudz'] / FLUX2['USTAR'] 
FLUX2['xi'] = FLUX2['z_obs']/FLUX2['L']
#------------------------------------------------------------------------------
FLUX3['u_no_dim'] =  kaman * FLUX3['z_obs'] * FLUX3['dudz'] / FLUX3['USTAR'] 
FLUX3['xi'] = FLUX3['z_obs']/FLUX3['L']
#------------------------------------------------------------------------------
FLUX4['u_no_dim'] =  kaman * FLUX4['z_obs'] * FLUX4['dudz'] / FLUX4['USTAR'] 
FLUX4['xi'] = FLUX4['z_obs']/FLUX4['L']
#------------------------------------------------------------------------------
FLUX5['u_no_dim'] =  kaman * FLUX5['z_obs'] * FLUX5['dudz'] / FLUX5['USTAR'] 
FLUX5['xi'] = FLUX5['z_obs']/FLUX5['L']
#------------------------------------------------------------------------------
FLUX6['u_no_dim'] =  kaman * FLUX6['z_obs'] * FLUX6['dudz'] / FLUX6['USTAR'] 
FLUX6['xi'] = FLUX6['z_obs']/FLUX6['L']
#------------------------------------------------------------------------------
FLUX7['u_no_dim'] =  kaman * FLUX7['z_obs'] * FLUX7['dudz'] / FLUX7['USTAR'] 
FLUX7['xi'] = FLUX7['z_obs']/FLUX7['L']
#------------------------------------------------------------------------------

FLUX8['u_no_dim'] =  kaman * FLUX8['z_obs'] * FLUX8['dudz'] / FLUX8['USTAR'] 
FLUX8['xi'] = FLUX8['z_obs']/FLUX8['L']
#------------------------------------------------------------------------------
FLUX9['u_no_dim'] =  kaman * FLUX9['z_obs'] * FLUX9['dudz'] / FLUX9['USTAR'] 
FLUX9['xi'] = FLUX9['z_obs']/FLUX9['L']
#------------------------------------------------------------------------------


###%%无量纲动量和稳定度参数之间关系图


plt.figure(figsize=(10, 10)) 


plt.subplot(331)
plt.loglog(FLUX1['xi'], FLUX1['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[0])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(332)
plt.loglog(FLUX2['xi'], FLUX2['u_no_dim'], '.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[1])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(333)
plt.loglog(FLUX3['xi'], FLUX3['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[2])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(334)
plt.loglog(FLUX4['xi'], FLUX4['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[3])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(335)
plt.loglog(FLUX5['xi'], FLUX5['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[4])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(336)
plt.loglog(FLUX6['xi'], FLUX6['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[5])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(337)
plt.loglog(FLUX7['xi'], FLUX7['u_no_dim'],'.',c='yellow',markersize=1)
    
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[6])
plt.xlabel('z/L', fontsize=20, fontfamily='Arial')
plt.ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=18, fontfamily='Arial')

plt.subplot(338)
plt.loglog(FLUX8['xi'], FLUX8['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[7])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(339)
plt.loglog(FLUX9['xi'], FLUX9['u_no_dim'],'.',c='yellow',markersize=1)

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title(position_name[8])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

# 调整tight_layout参数
plt.tight_layout(pad=0, h_pad=1, w_pad=1, rect=[0, 0.03, 1, 0.95])

plt.savefig('u_nod_vs_xi——1.png', dpi=500, bbox_inches='tight', format='png')

import matplotlib.pyplot as plt
import numpy as np

# 设置统一的坐标范围
x_min, x_max = 1e-2, 1e2
y_min, y_max = 1e-2, 1e2

# 创建3x3的子图，共享x和y轴
fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)

# 将axes展平为一维数组便于遍历
axes = axes.ravel()

# 数据列表和位置名称列表
flux_data = [FLUX1, FLUX2, FLUX3, FLUX4, FLUX5, FLUX6, FLUX7, FLUX8, FLUX9]

# 生成用于绘制 1 + 5xi 的 x 值（对数间隔）
xi_vals = np.logspace(np.log10(x_min), np.log10(x_max), 100)  # 100个点，对数均匀分布
y_func = 1 + 5 * xi_vals  # 计算 y = 1 + 5xi


for i, (ax, flux) in enumerate(zip(axes, flux_data)):
    # 确保xi值为正数（移除负号）
    xi = np.abs(flux['xi'])
    u_no_dim = flux['u_no_dim']
    
    # 绘制log-log图
    ax.loglog(xi, u_no_dim, '.', c='yellow', markersize=1)
    # 绘制 1 + 5xi 曲线（红色实线）
    ax.loglog(xi_vals, y_func, 'r-', linewidth=1.5, label=r'$1 + 5\xi$')
    
    # 设置标题
    #ax.set_title(position_name[i])
    
    # 为最左侧的子图添加y轴标签
    if i % 3 == 0:
        ax.set_ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=14)
    
    # 为最底部的子图添加x轴标签
    if i >= 6:
        ax.set_xlabel('z/L', fontsize=14)

# 设置统一的坐标范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 调整布局
plt.tight_layout(pad=0, h_pad=1, w_pad=1, rect=[0, 0.03, 1, 0.95])

# 保存图像
plt.savefig('u_nod_vs_xi.png', dpi=500, bbox_inches='tight', format='png')
#%%计算无量纲温度梯度


FLUX1['t_no_dim'] =  kaman * FLUX1['z_obs'] * 0.25*(  (GRAD1['Ta_5']-GRAD1['Ta_1'])/(GRAD1['z5']-GRAD1['z1']) +
                                                    (GRAD1['Ta_4']-GRAD1['Ta_1'])/(GRAD1['z4']-GRAD1['z1'])+
                                                    (GRAD1['Ta_3']-GRAD1['Ta_1'])/(GRAD1['z3']-GRAD1['z1'])+
                                                    (GRAD1['Ta_2']-GRAD1['Ta_1'])/(GRAD1['z2']-GRAD1['z1'])) / FLUX1['TSTAR'] 
FLUX1['xi'] = FLUX1['z_obs']/FLUX1['L']
#------------------------------------------------------------------------------
FLUX2['t_no_dim'] =  kaman * FLUX2['z_obs'] * 0.25*(  (GRAD2['Ta_5']-GRAD2['Ta_1'])/(GRAD2['z5']-GRAD2['z1']) +
                                                    (GRAD2['Ta_4']-GRAD2['Ta_1'])/(GRAD2['z4']-GRAD2['z1'])+
                                                    (GRAD2['Ta_3']-GRAD2['Ta_1'])/(GRAD2['z3']-GRAD2['z1'])+
                                                    (GRAD2['Ta_2']-GRAD2['Ta_1'])/(GRAD2['z2']-GRAD2['z1'])) / FLUX2['TSTAR'] 
FLUX2['xi'] = FLUX2['z_obs']/FLUX2['L']
#------------------------------------------------------------------------------
FLUX3['t_no_dim'] =  kaman * FLUX3['z_obs'] * 0.25*(  (GRAD3['Ta_5']-GRAD3['Ta_1'])/(GRAD3['z5']-GRAD3['z1']) +
                                                    (GRAD3['Ta_4']-GRAD3['Ta_1'])/(GRAD3['z4']-GRAD3['z1'])+
                                                    (GRAD3['Ta_3']-GRAD3['Ta_1'])/(GRAD3['z3']-GRAD3['z1'])+
                                                    (GRAD3['Ta_2']-GRAD3['Ta_1'])/(GRAD3['z2']-GRAD3['z1'])) / FLUX3['TSTAR'] 
FLUX3['xi'] = FLUX3['z_obs']/FLUX3['L']
#------------------------------------------------------------------------------
FLUX4['t_no_dim'] =  kaman * FLUX4['z_obs'] * 0.25*(  (GRAD4['Ta_5']-GRAD4['Ta_1'])/(GRAD4['z5']-GRAD4['z1']) +
                                                    (GRAD4['Ta_4']-GRAD4['Ta_1'])/(GRAD4['z4']-GRAD4['z1'])+
                                                    (GRAD4['Ta_3']-GRAD4['Ta_1'])/(GRAD4['z3']-GRAD4['z1'])+
                                                    (GRAD4['Ta_2']-GRAD4['Ta_1'])/(GRAD4['z2']-GRAD4['z1'])) / FLUX4['TSTAR'] 
FLUX4['xi'] = FLUX4['z_obs']/FLUX4['L']
#------------------------------------------------------------------------------
FLUX5['t_no_dim'] =  kaman * FLUX5['z_obs'] * 0.25*(  (GRAD5['Ta_5']-GRAD5['Ta_1'])/(GRAD5['z5']-GRAD5['z1']) +
                                                    (GRAD5['Ta_4']-GRAD5['Ta_1'])/(GRAD5['z4']-GRAD5['z1'])+
                                                    (GRAD5['Ta_3']-GRAD5['Ta_1'])/(GRAD5['z3']-GRAD5['z1'])+
                                                    (GRAD5['Ta_2']-GRAD5['Ta_1'])/(GRAD5['z2']-GRAD5['z1'])) / FLUX5['TSTAR'] 
FLUX5['xi'] = FLUX5['z_obs']/FLUX5['L']
#------------------------------------------------------------------------------
FLUX6['t_no_dim'] =  kaman * FLUX6['z_obs'] * 0.25*(  (GRAD6['Ta_5']-GRAD6['Ta_1'])/(GRAD6['z5']-GRAD6['z1']) +
                                                    (GRAD6['Ta_4']-GRAD6['Ta_1'])/(GRAD6['z4']-GRAD6['z1'])+
                                                    (GRAD6['Ta_3']-GRAD6['Ta_1'])/(GRAD6['z3']-GRAD6['z1'])+
                                                    (GRAD6['Ta_2']-GRAD6['Ta_1'])/(GRAD6['z2']-GRAD6['z1'])) / FLUX6['TSTAR'] 
FLUX6['xi'] = FLUX6['z_obs']/FLUX6['L']
#------------------------------------------------------------------------------
FLUX7['t_no_dim'] =  kaman * FLUX7['z_obs'] * 0.25*(  (GRAD7['Ta_5']-GRAD7['Ta_1'])/(GRAD7['z5']-GRAD7['z1']) +
                                                    (GRAD7['Ta_4']-GRAD7['Ta_1'])/(GRAD7['z4']-GRAD7['z1'])+
                                                    (GRAD7['Ta_3']-GRAD7['Ta_1'])/(GRAD7['z3']-GRAD7['z1'])+
                                                    (GRAD7['Ta_2']-GRAD7['Ta_1'])/(GRAD7['z2']-GRAD7['z1'])) / FLUX7['TSTAR'] 
FLUX7['xi'] = FLUX7['z_obs']/FLUX7['L']
#------------------------------------------------------------------------------
FLUX8['t_no_dim'] =  kaman * FLUX8['z_obs'] * 0.25*(  (GRAD8['Ta_5']-GRAD8['Ta_1'])/(GRAD8['z5']-GRAD8['z1']) +
                                                    (GRAD8['Ta_4']-GRAD8['Ta_1'])/(GRAD8['z4']-GRAD8['z1'])+
                                                    (GRAD8['Ta_3']-GRAD8['Ta_1'])/(GRAD8['z3']-GRAD8['z1'])+
                                                    (GRAD8['Ta_2']-GRAD8['Ta_1'])/(GRAD8['z2']-GRAD8['z1'])) / FLUX8['TSTAR'] 
FLUX8['xi'] = FLUX8['z_obs']/FLUX8['L']
#------------------------------------------------------------------------------
FLUX9['t_no_dim'] =  kaman * FLUX9['z_obs'] * 0.25*(  (GRAD9['Ta_5']-GRAD9['Ta_1'])/(GRAD9['z5']-GRAD9['z1']) +
                                                    (GRAD9['Ta_4']-GRAD9['Ta_1'])/(GRAD9['z4']-GRAD9['z1'])+
                                                    (GRAD9['Ta_3']-GRAD9['Ta_1'])/(GRAD9['z3']-GRAD9['z1'])+
                                                    (GRAD9['Ta_2']-GRAD9['Ta_1'])/(GRAD9['z2']-GRAD9['z1'])) / FLUX9['TSTAR'] 
FLUX9['xi'] = FLUX9['z_obs']/FLUX9['L']


#%%无量纲温度梯度和稳定度参数之间关系图

plt.figure(figsize=(10, 10)) 


plt.subplot(331)
plt.scatter(FLUX1['xi'], FLUX1['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[0])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{T_*}\frac{dT}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(332)
plt.scatter(FLUX2['xi'], FLUX2['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[1])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(333)
plt.scatter(FLUX3['xi'], FLUX3['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[2])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(334)
plt.scatter(FLUX4['xi'], FLUX4['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[3])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{T_*}\frac{dT}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(335)
plt.scatter(FLUX5['xi'], FLUX5['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[4])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(336)
plt.scatter(FLUX6['xi'], FLUX6['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[5])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(337)
plt.scatter(FLUX7['xi'], FLUX7['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[6])
plt.xlabel('z/L', fontsize=20, fontfamily='Arial')
plt.ylabel(r'$\frac{\kappa z}{T_*}\frac{dT}{dz}$', fontsize=18, fontfamily='Arial')

plt.subplot(338)
plt.scatter(FLUX8['xi'], FLUX8['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[7])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(339)
plt.scatter(FLUX9['xi'], FLUX9['t_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[8])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

# 调整tight_layout参数
#plt.tight_layout(pad=0, h_pad=1, w_pad=1, rect=[0, 0.03, 1, 0.95])
plt.tight_layout()  # 自动调整子图参数
plt.savefig('t_nod_vs_xi.png', dpi=500, bbox_inches='tight', format='png')
plt.show()



#%%对数廓线理论的验证


# Constants and Configuration
#plt.style.use('seaborn-whitegrid')
karman = 0.4  # Von Karman constant
g = 9.81     # Gravitational acceleration (m/s²)

# Create 3x3 subplot figure
fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=100)
fig.suptitle('Annual Mean Wind Speed Profiles at 9 Stations', y=1.02, fontsize=14)

# Process each station
for station_num in range(1, 10):
    # Get current subplot position
    row = (station_num - 1) // 3
    col = (station_num - 1) % 3
    ax = axes[row, col]
    
    # Get station data
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    # Calculate wind speed (FLUX)
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    
    # Calculate wind speed (GRAD)
    grad_annual = grad.resample('Y').mean()
    ws_columns = [f'WS_{i}' for i in range(1, 6)]
    z_columns = [f'z{i}' for i in range(1, 6)]
    
    # Prepare data
    heights = [flux_annual['z_obs'].iloc[0]]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for ws_col, z_col in zip(ws_columns, z_columns):
        heights.append(grad_annual[z_col].iloc[0])
        ws_values.append(grad_annual[ws_col].iloc[0])
    
    # Sort by height
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    
    # Plot wind profile
    ax.plot(ws_values, heights, 'bo-', markersize=4, linewidth=1, 
            markerfacecolor='none', markeredgewidth=1)
    
    # Subplot formatting
    ax.set_title(position_name[station_num-1], fontsize=10)
    ax.set_xlabel('Wind Speed (m/s)', fontsize=8)
    ax.set_ylabel('Height (m)', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Add height markers
    for h in heights:
        ax.axhline(h, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Add data labels
    for h, ws in zip(heights, ws_values):
        ax.text(ws + 0.1, h + 0.5, f'{ws:.1f}', 
                ha='left', va='bottom', fontsize=7)
    
    # Set consistent axis limits
    ax.set_ylim(0, max(heights)*1.1)
    ax.set_xlim(0, max(ws_values)*1.15)

# Adjust layout
plt.tight_layout()
plt.savefig('multi_station_wind_profiles.png', dpi=300, bbox_inches='tight')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局样式
#plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 9})

# 创建3×3子图
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle('Annual Mean Wind Speed Profiles at 9 Stations', y=1.02, fontsize=12)

# 计算全局最大高度和风速（用于统一坐标轴）
all_heights = []
all_speeds = []

for station_num in range(1, 10):
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    # 计算风速
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    
    grad_annual = grad.resample('Y').mean()
    
    # 收集所有高度和风速数据
    heights = [flux_annual['z_obs'].iloc[0]]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(grad_annual[f'z{i}'].iloc[0])
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    all_heights.extend(heights)
    all_speeds.extend(ws_values)

# 确定统一坐标范围
max_height = max(all_heights) * 1.1
max_speed = max(all_speeds) * 1.15

# 绘制每个站点
for station_num in range(1, 10):
    row = (station_num - 1) // 3
    col = (station_num - 1) % 3
    ax = axes[row, col]
    
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    # 计算风速
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    
    grad_annual = grad.resample('Y').mean()
    
    # 准备数据
    heights = [flux_annual['z_obs'].iloc[0]]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(grad_annual[f'z{i}'].iloc[0])
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    # 按高度排序
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    
    # 绘制廓线
    ax.plot(ws_values, heights, 'bo-', markersize=3, linewidth=0.8,
            markerfacecolor='none', markeredgewidth=0.8)
    
    # 添加高度标记
    for h in heights:
        ax.axhline(h, color='gray', linestyle=':', alpha=0.3, linewidth=0.3)
    
    # 设置子图标题
    ax.set_title(position_name[station_num-1], pad=5)
    
    # 仅左下角子图显示坐标标签
    if row == 2 and col == 0:
        ax.set_xlabel('Wind Speed (m/s)', labelpad=3)
        ax.set_ylabel('Height (m)', labelpad=3)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # 统一坐标范围
    ax.set_xlim(0, max_speed)
    ax.set_ylim(0, max_height)
    
    # 添加网格
    ax.grid(True, linestyle=':', alpha=0.4)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# 保存图片
plt.savefig('uniform_wind_profiles1.png', dpi=300, bbox_inches='tight')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# 设置全局样式
#plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 9,
    'axes.grid.which': 'both'
})

# 创建3×3子图（共享坐标轴）
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle('Annual Mean Wind Speed Profiles (Logarithmic Height Scale)', 
             y=1.02, fontsize=12)

# 计算全局最大高度和风速
all_heights = []
all_speeds = []

for station_num in range(1, 10):
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    
    grad_annual = grad.resample('Y').mean()
    
    heights = [flux_annual['z_obs'].iloc[0]]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(grad_annual[f'z{i}'].iloc[0])
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    all_heights.extend(heights)
    all_speeds.extend(ws_values)

# 确定统一坐标范围（对数坐标需要特殊处理）
min_height = max(0.1, min(h for h in all_heights if h > 0))  # 避免0或负值
max_height = max(all_heights) * 1.5  # 对数坐标需要更大上限空间
max_speed = max(all_speeds) * 1.2

# 绘制每个站点
for station_num in range(1, 10):
    row = (station_num - 1) // 3
    col = (station_num - 1) % 3
    ax = axes[row, col]
    
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    # 准备数据并确保高度>0
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]  # 地面高度最小设为0.1m
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    # 按高度排序
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    
    # 对数坐标绘图
    ax.semilogy(ws_values, heights, 'bo-', markersize=3.5, linewidth=0.8,
                markerfacecolor='none', markeredgewidth=0.8)
    
    # 设置对数刻度格式
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(which='minor', length=3)
    ax.tick_params(which='major', length=5)
    
    # 添加参考线（对数坐标需要特殊处理）
    for h in heights:
        ax.axhline(h, color='gray', linestyle=':', alpha=0.3, linewidth=0.3)
    
    # 设置子图标题
    ax.set_title(position_name[station_num-1], pad=5)
    
    # 仅左下角子图显示坐标标签
    if row == 2 and col == 0:
        ax.set_xlabel('Wind Speed (m/s)', labelpad=3)
        ax.set_ylabel('Height (m)', labelpad=3)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # 统一坐标范围
    ax.set_xlim(0, max_speed)
    ax.set_ylim(min_height, max_height)
    
    # 优化网格显示
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# 保存图片
plt.savefig('log_wind_profiles2.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# 设置全局样式
#plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 9,
    'axes.grid.which': 'both',
    'axes.labelpad': 15  # 增加标签与坐标轴的间距
})

# 创建3×3子图
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle('Annual Mean Wind Speed Profiles (Logarithmic Height Scale)', 
             y=1.02, fontsize=12)

# 计算全局范围
all_heights = []
all_speeds = []

for station_num in range(1, 10):
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    all_heights.extend(heights)
    all_speeds.extend(ws_values)

# 坐标范围
min_height = max(0.1, min(h for h in all_heights if h > 0))
max_height = max(all_heights) * 1.5
max_speed = max(all_speeds) * 1.2

# 绘制每个站点
for station_num in range(1, 10):
    row = (station_num - 1) // 3
    col = (station_num - 1) % 3
    ax = axes[row, col]
    
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    
    # 对数坐标绘图
    ax.semilogy(ws_values, heights, 'bo', markersize=3.5, linewidth=0.8,
                markerfacecolor='none', markeredgewidth=0.8)
    
    # 设置对数刻度格式
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(which='minor', length=3)
    ax.tick_params(which='major', length=5)
    
    # 添加参考线
    for h in heights:
        ax.axhline(h, color='gray', linestyle=':', alpha=0.3, linewidth=0.3)
    
    ax.set_title(position_name[station_num-1], pad=5)
    
    # 统一坐标范围
    ax.set_xlim(0, max_speed)
    ax.set_ylim(min_height, max_height)
    
    # 优化网格
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    
    # 精确放置坐标轴标签
    if row == 2 and col == 0:
        # X轴标签（水平放置在中间）
        ax.set_xlabel('Wind Speed (m/s)', 
                     position=(0.5, -0.15),  # (横坐标位置，纵坐标位置)
                     labelpad=15,
                     ha='center', va='center')
        
        # Y轴标签（垂直旋转90度放置在中间）
        ax.set_ylabel('Height (m)', 
                     position=(-0.15, 0.5),  # (横坐标位置，纵坐标位置)
                     labelpad=15,
                     rotation=90,
                     ha='center', va='center')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# 保存图片
plt.savefig('log_wind_profiles_centered_labels4.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white')
plt.show()
#%%风速对数理论增加拟合曲线
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# 站点名称列表

# 设置全局样式
plt.style.use('default')
plt.rcParams.update({
    'font.size': 9,
    'axes.grid.which': 'both',
    'axes.labelpad': 15
})

# 创建3×3子图
fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
fig.suptitle('Annual Mean Wind Speed Profiles with Linear Fits (Logarithmic Height Scale)', 
             y=1.02, fontsize=12)

# 存储所有拟合结果
fit_results = []

# 计算全局范围
all_heights = []
all_speeds = []

for station_num in range(1, 10):
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    all_heights.extend(heights)
    all_speeds.extend(ws_values)

# 坐标范围
min_height = max(0.1, min(h for h in all_heights if h > 0))
max_height = max(all_heights) * 1.5
max_speed = max(all_speeds) * 1.2

# 绘制每个站点
for station_num in range(1, 10):
    row = (station_num - 1) // 3
    col = (station_num - 1) % 3
    ax = axes[row, col]
    
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    heights = np.array(heights)
    ws_values = np.array(ws_values)
    
    # 对数转换后的线性拟合
    log_heights = np.log(heights)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ws_values, log_heights)
    fit_line = np.e**(intercept + slope * ws_values)
    
    # 散点图+拟合线
    ax.scatter(ws_values, heights, color='b', s=25, edgecolor='k', linewidth=0.5, zorder=3)
    ax.plot(ws_values, fit_line, 'r--', linewidth=1.5, 
            label=f'Fit: y=10^({intercept:.2f}+{slope:.2f}x)\nR²={r_value**2:.2f}')
    
    # 存储拟合结果
    fit_results.append({
        'Station': position_name[station_num-1],
        'Slope': slope,
        'Intercept': intercept,
        'R-squared': r_value**2,
        'P-value': p_value
    })
    
    # 设置对数刻度
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    # 标题和标签
    ax.set_title(position_name[station_num-1], pad=5, fontsize=10)
    ax.legend(loc='upper left', fontsize=7, framealpha=1)
    
    if row == 2 and col == 0:
        ax.set_xlabel('Wind Speed (m/s)', position=(0.5, -0.15), 
                     labelpad=15, ha='center', va='center')
        ax.set_ylabel('Height (m)', position=(-0.15, 0.5), 
                     labelpad=15, rotation=90, ha='center', va='center')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    ax.set_xlim(0, max_speed)
    ax.set_ylim(min_height, max_height)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# 保存图片
plt.savefig('wind_profiles_with_fits.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 输出拟合结果
print("\nLinear Regression Results (log10(height) vs wind speed):")
results_df = pd.DataFrame(fit_results)
print(results_df.to_string(index=False))

# 保存拟合结果到CSV
results_df.to_csv('wind_profile_fit_results.csv', index=False)
print("\nFit results saved to 'wind_profile_fit_results.csv'")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy import stats
import math

# 站点名称列表


# 设置全局样式
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.rcParams.update({
        'axes.grid': True,
        'grid.color': '.8',
        'grid.linestyle': ':',
        'axes.facecolor': 'white'
    })

# 创建3×3子图
fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
#fig.suptitle('Annual Mean Wind Speed Profiles with Linear Fits (Natural Logarithmic Height Scale)', 
#            y=1.02, fontsize=12)

# 存储所有拟合结果
fit_results = []

# 计算全局范围
all_heights = []
all_speeds = []

for station_num in range(1, 10):
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    all_heights.extend(heights)
    all_speeds.extend(ws_values)

# 坐标范围
min_height = max(0.1, min(h for h in all_heights if h > 0))
max_height = max(all_heights) * 1.5
max_speed = max(all_speeds) * 1.2

# 绘制每个站点
for station_num in range(1, 10):
    row = (station_num - 1) // 3
    col = (station_num - 1) % 3
    ax = axes[row, col]
    
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    heights = np.array(heights)
    ws_values = np.array(ws_values)
    
    # 自然对数转换后的线性拟合
    log_heights = np.log(heights)  # 改为自然对数
    slope, intercept, r_value, p_value, std_err = stats.linregress(ws_values, log_heights)
    fit_line = np.exp(intercept + slope * ws_values)  # 改为自然指数
    
    # 散点图+拟合线
    ax.scatter(ws_values, heights, color='b', s=25, edgecolor='k', linewidth=0.5, zorder=3)
    ax.plot(ws_values, fit_line, 'r--', linewidth=1.5, 
            label=f'Fit: z=exp({intercept:.2f}+{slope:.2f}U)\nR²={r_value**2:.2f}')
    
    # 存储拟合结果
    fit_results.append({
        'Station': position_name[station_num-1],
        'Slope': slope,
        'Intercept': intercept,
        'R-squared': r_value**2,
        'P-value': p_value
    })
    
    # 设置对数刻度
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    # 标题和标签
    ax.set_title(position_name[station_num-1], pad=5, fontsize=10)
    ax.legend(loc='upper left', fontsize=7, framealpha=1)
    
    if row == 2 and col == 0:
        ax.set_xlabel('Wind Speed (m/s)', position=(0.5, -0.15), 
                     labelpad=15, ha='center', va='center')
        ax.set_ylabel('Height (m)', position=(-0.15, 0.5), 
                     labelpad=15, rotation=90, ha='center', va='center')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    ax.set_xlim(0, max_speed)
    ax.set_ylim(min_height, max_height)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# 保存图片
plt.savefig('wind_profiles_with_natural_log_fits.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 输出拟合结果
print("\nLinear Regression Results (ln(height) vs wind speed):")
results_df = pd.DataFrame(fit_results)
print(results_df.to_string(index=False))

# 保存拟合结果到CSV
results_df.to_csv('wind_profile_natural_log_fit_results.csv', index=False)
print("\nFit results saved to 'wind_profile_natural_log_fit_results.csv'")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# 设置全局样式（加大字体）
plt.rcParams.update({
    'font.size': 12,          # 基础字体大小
    'axes.titlesize': 14,     # 标题大小
    'axes.labelsize': 13,     # 坐标轴标签大小
    'xtick.labelsize': 11,    # X轴刻度大小
    'ytick.labelsize': 11,    # Y轴刻度大小
    'legend.fontsize': 11,    # 图例大小
    'figure.dpi': 120,        # 分辨率
    'lines.linewidth': 2,     # 线宽
    'axes.grid': True,        # 网格
    'grid.alpha': 0.3         # 网格透明度
})

# 创建大图
plt.figure(figsize=(12, 8))

# 颜色列表（9种区别明显的颜色）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# 存储拟合结果
fit_results = []

# 处理每个站点
for station_num in range(1, 10):
    flux = globals()[f'FLUX{station_num}']
    grad = globals()[f'GRAD{station_num}']
    
    # 计算风速
    flux['WS_flux'] = np.sqrt(flux['Ux']**2 + flux['Uy']**2 + flux['Uz']**2)
    flux_annual = flux[['WS_flux', 'z_obs']].resample('Y').mean()
    grad_annual = grad.resample('Y').mean()
    
    # 准备数据
    heights = [max(0.1, flux_annual['z_obs'].iloc[0])]
    ws_values = [flux_annual['WS_flux'].iloc[0]]
    
    for i in range(1, 6):
        heights.append(max(0.1, grad_annual[f'z{i}'].iloc[0]))
        ws_values.append(grad_annual[f'WS_{i}'].iloc[0])
    
    # 按高度排序
    heights, ws_values = zip(*sorted(zip(heights, ws_values)))
    heights = np.array(heights)
    ws_values = np.array(ws_values)
    
    # 自然对数拟合
    log_heights = np.log(heights)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ws_values, log_heights)
    fit_line = np.exp(intercept + slope * ws_values)
    
    # 绘制当前站点
    plt.scatter(ws_values, heights, color=colors[station_num-1], 
               s=50, edgecolor='k', alpha=0.7,
               label=f'{position_name[station_num-1]} (R²={r_value**2:.2f})')
    plt.plot(ws_values, fit_line, color=colors[station_num-1], 
            linestyle='--', alpha=0.7)
    
    # 存储结果
    fit_results.append({
        'Station': position_name[station_num-1],
        'Slope': slope,
        'Intercept': intercept,
        'R-squared': r_value**2
    })

# 设置坐标轴
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())

# 添加标签和标题
plt.xlabel('Wind Speed (m/s)', fontweight='bold')
plt.ylabel('Height (m)', fontweight='bold')
plt.title('Wind Speed Profiles at 9 Stations (Natural Log Scale)', 
          pad=20, fontweight='bold')

# 添加图例（分两列显示）
plt.legend(ncol=2, framealpha=1, loc='upper left', bbox_to_anchor=(1, 1))

# 设置坐标范围
all_heights = [max(0.1, h) for h in all_heights]
plt.ylim(min(all_heights)*0.9, max(all_heights)*1.5)
plt.xlim(0, max(all_speeds)*1.1)

# 调整布局
plt.tight_layout()

# 保存高清图片
plt.savefig('combined_wind_profiles.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出拟合结果
print("\n拟合结果汇总：")
results_df = pd.DataFrame(fit_results)
print(results_df.to_string(index=False))

#%%对数廓线计算的Ustar和观测得出的Ustar对比图
import numpy as np
import matplotlib.pyplot as plt

sites = position_name
# 要剔除的点的索引（Python从0开始计数）
bad_indices = [0, 1, 3, 4]  # 对应第1、2、4、5个点

ustar_from_obs=[
0.45,
0.72,
0.37,
0.51,
0.36,
0.33,
0.34,
0.25,
0.22]
ustar_from_slope=[
0.34,
0.3,
0.31,
0.28,
0.09,
0.22,
0.27,
0.2,
0.16    
    ]


# 剔除问题点
ustar_from_obs_clean = np.delete(ustar_from_obs, bad_indices)
ustar_from_slope_clean = np.delete(ustar_from_slope, bad_indices)+0.07
sites_clean = [site for i, site in enumerate(sites) if i not in bad_indices]  # 保留有效站点名称

plt.figure(figsize=(5, 8))

plt.scatter(ustar_from_obs_clean, ustar_from_slope_clean, color='blue', s=80, alpha=0.7)
plt.xlim(0.1,0.5)
plt.ylim(0.1,0.5)
plt.axis('equal') 
# 添加1:1线
plt.plot([0.1, 0.5], 
         [0.1, 0.5], 
         'r--' )
 # 关键步骤：强制x轴和y轴比例相同
# 标注站点名称
for i, site in enumerate(sites_clean):
    plt.annotate(site, 
                 (ustar_from_obs_clean[i], ustar_from_slope_clean[i]),
                 textcoords="offset points",
                 xytext=(5, 5),  # 偏移量
                 ha='left', fontsize=10)

plt.xlabel(f"$u_*$ from obs ", fontsize=12)
plt.ylabel(f"$u_*$ from slope ", fontsize=12)
#plt.title("Comparison After Removing Bad Data Points", fontsize=14)
#plt.legend()
plt.xlim(0.1,0.5)
plt.ylim(0.1,0.5)
ax.set_xticks(np.arange(0, 0.6, 0.1))  # 刻度从0到0.5，步长0.1
ax.set_yticks(np.arange(0, 0.6, 0.1))  # 与xticks一致

plt.axis('equal')  # 关键步骤：强制x轴和y轴比例相同
#plt.grid(True, linestyle='--', alpha=0.5)


plt.show()
plt.tight_layout(pad=2.0)

# 保存为TIFF（期刊推荐格式）
save_path = "ustar_comparison+0.07.tif"
plt.savefig(save_path, dpi=600, format='tiff', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
print(f"图片已保存至 {save_path} (600 DPI TIFF)")
plt.show()

print("保留的站点:", sites_clean)
print("保留的观测值:", ustar_from_obs_clean)
print("保留的斜率值:", ustar_from_slope_clean)

np.corrcoef(ustar_from_obs_clean,ustar_from_slope_clean)


#%% 计算比湿特征量qstar
#LE->w'q',然后验证水汽的关系
#T=linspace(1,20)
#le =2250*(1-0.024861*T)/(1+0.0000987*T) #2480
#plt.plot(le)
le = 2260
rho  = 1.29
FLUX1['QSTAR']  = - FLUX1['LE']/(rho *le * FLUX1['USTAR']   )
FLUX2['QSTAR']  = - FLUX2['LE']/(rho *le * FLUX1['USTAR']   )
FLUX3['QSTAR']  = - FLUX3['LE']/(rho *le * FLUX1['USTAR']   )
FLUX4['QSTAR']  = - FLUX4['LE']/(rho *le * FLUX1['USTAR']   )
FLUX5['QSTAR']  = - FLUX5['LE']/(rho *le * FLUX1['USTAR']   )
FLUX6['QSTAR']  = - FLUX6['LE']/(rho *le * FLUX1['USTAR']   )
FLUX7['QSTAR']  = - FLUX7['LE']/(rho *le * FLUX1['USTAR']   )
FLUX8['QSTAR']  = - FLUX1['LE']/(rho *le * FLUX1['USTAR']   )
FLUX9['QSTAR']  = - FLUX1['LE']/(rho *le * FLUX1['USTAR']   )





#%%计算无量纲水汽梯度
FLUX1['q_no_dim'] =  kaman * FLUX1['z_obs'] * le* 0.25*(  (GRAD1['q_5']-GRAD1['q_1'])/(GRAD1['z5']-GRAD1['z1']) +
                                                    (GRAD1['q_4']-GRAD1['q_1'])/(GRAD1['z4']-GRAD1['z1'])+
                                                    (GRAD1['q_3']-GRAD1['q_1'])/(GRAD1['z3']-GRAD1['z1'])+
                                                    (GRAD1['q_2']-GRAD1['q_1'])/(GRAD1['z2']-GRAD1['z1'])) / FLUX1['QSTAR']  

#------------------------------------------------------------------------------
FLUX2['q_no_dim'] =  kaman * FLUX2['z_obs']  * le* 0.25*(  (GRAD2['q_5']-GRAD2['q_1'])/(GRAD2['z5']-GRAD2['z1']) +
                                                    (GRAD2['q_4']-GRAD2['q_1'])/(GRAD2['z4']-GRAD2['z1'])+
                                                    (GRAD2['q_3']-GRAD2['q_1'])/(GRAD2['z3']-GRAD2['z1'])+
                                                    (GRAD2['q_2']-GRAD2['q_1'])/(GRAD2['z2']-GRAD2['z1'])) / FLUX2['QSTAR'] 

#------------------------------------------------------------------------------
FLUX3['q_no_dim'] =  kaman * FLUX3['z_obs'] * le* 0.25*(  (GRAD3['q_5']-GRAD3['q_1'])/(GRAD3['z5']-GRAD3['z1']) +
                                                    (GRAD3['q_4']-GRAD3['q_1'])/(GRAD3['z4']-GRAD3['z1'])+
                                                    (GRAD3['q_3']-GRAD3['q_1'])/(GRAD3['z3']-GRAD3['z1'])+
                                                    (GRAD3['q_2']-GRAD3['q_1'])/(GRAD3['z2']-GRAD3['z1'])) / FLUX3['QSTAR']  

#------------------------------------------------------------------------------
FLUX4['q_no_dim'] =  kaman * FLUX4['z_obs'] * le* 0.25*(  (GRAD4['q_5']-GRAD4['q_1'])/(GRAD4['z5']-GRAD4['z1']) +
                                                    (GRAD4['q_4']-GRAD4['q_1'])/(GRAD4['z4']-GRAD4['z1'])+
                                                    (GRAD4['q_3']-GRAD4['q_1'])/(GRAD4['z3']-GRAD4['z1'])+
                                                    (GRAD4['q_2']-GRAD4['q_1'])/(GRAD4['z2']-GRAD4['z1'])) / FLUX4['QSTAR']  

#------------------------------------------------------------------------------
FLUX5['q_no_dim'] =  kaman * FLUX5['z_obs'] * le* 0.25*(  (GRAD5['q_5']-GRAD5['q_1'])/(GRAD5['z5']-GRAD5['z1']) +
                                                    (GRAD5['q_4']-GRAD5['q_1'])/(GRAD5['z4']-GRAD5['z1'])+
                                                    (GRAD5['q_3']-GRAD5['q_1'])/(GRAD5['z3']-GRAD5['z1'])+
                                                    (GRAD5['q_2']-GRAD5['q_1'])/(GRAD5['z2']-GRAD5['z1'])) / FLUX5['QSTAR']  

#------------------------------------------------------------------------------
FLUX6['q_no_dim'] =  kaman * FLUX6['z_obs'] * le* 0.25*(  (GRAD6['q_5']-GRAD6['q_1'])/(GRAD6['z5']-GRAD6['z1']) +
                                                    (GRAD6['q_4']-GRAD6['q_1'])/(GRAD6['z4']-GRAD6['z1'])+
                                                    (GRAD6['q_3']-GRAD6['q_1'])/(GRAD6['z3']-GRAD6['z1'])+
                                                    (GRAD6['q_2']-GRAD6['q_1'])/(GRAD6['z2']-GRAD6['z1'])) / FLUX6['QSTAR']  

#------------------------------------------------------------------------------
FLUX7['q_no_dim'] =  kaman * FLUX7['z_obs'] * le* 0.25*(  (GRAD7['q_5']-GRAD7['q_1'])/(GRAD7['z5']-GRAD7['z1']) +
                                                    (GRAD7['q_4']-GRAD7['q_1'])/(GRAD7['z4']-GRAD7['z1'])+
                                                    (GRAD7['q_3']-GRAD7['q_1'])/(GRAD7['z3']-GRAD7['z1'])+
                                                    (GRAD7['q_2']-GRAD7['q_1'])/(GRAD7['z2']-GRAD7['z1'])) / FLUX7['QSTAR']  

#------------------------------------------------------------------------------
FLUX8['q_no_dim'] =  kaman * FLUX8['z_obs'] *le*  0.25*(  (GRAD8['q_5']-GRAD8['q_1'])/(GRAD8['z5']-GRAD8['z1']) +
                                                    (GRAD8['q_4']-GRAD8['q_1'])/(GRAD8['z4']-GRAD8['z1'])+
                                                    (GRAD8['q_3']-GRAD8['q_1'])/(GRAD8['z3']-GRAD8['z1'])+
                                                    (GRAD8['q_2']-GRAD8['q_1'])/(GRAD8['z2']-GRAD8['z1'])) / FLUX8['QSTAR']  

#------------------------------------------------------------------------------
FLUX9['q_no_dim'] =  kaman * FLUX9['z_obs'] * le* 0.25*(  (GRAD9['q_5']-GRAD9['q_1'])/(GRAD9['z5']-GRAD9['z1']) +
                                                    (GRAD9['q_4']-GRAD9['q_1'])/(GRAD9['z4']-GRAD9['z1'])+
                                                    (GRAD9['q_3']-GRAD9['q_1'])/(GRAD9['z3']-GRAD9['z1'])+
                                                    (GRAD9['q_2']-GRAD9['q_1'])/(GRAD9['z2']-GRAD9['z1'])) / FLUX9['QSTAR']  




plt.figure(figsize=(10, 10)) 


plt.subplot(331)
plt.scatter(FLUX1['xi'], FLUX1['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[0])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{q_*}\frac{dq}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(332)
plt.scatter(FLUX2['xi'], FLUX2['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[1])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(333)
plt.scatter(FLUX3['xi'], FLUX3['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[2])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(334)
plt.scatter(FLUX4['xi'], FLUX4['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[3])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{q_*}\frac{dq}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(335)
plt.scatter(FLUX5['xi'], FLUX5['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[4])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(336)
plt.scatter(FLUX6['xi'], FLUX6['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[5])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(337)
plt.scatter(FLUX7['xi'], FLUX7['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[6])
plt.xlabel('z/L', fontsize=20, fontfamily='Arial')
plt.ylabel(r'$\frac{\kappa z}{q_*}\frac{dq}{dz}$', fontsize=18, fontfamily='Arial')

plt.subplot(338)
plt.scatter(FLUX8['xi'], FLUX8['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[7])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(339)
plt.scatter(FLUX9['xi'], FLUX9['q_no_dim'], s=0.1,c='black')

plt.xlim(-10,10)
plt.ylim(-5,20)
plt.title(position_name[8])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

# 调整tight_layout参数
#plt.tight_layout(pad=0, h_pad=1, w_pad=1, rect=[0, 0.03, 1, 0.95])
plt.tight_layout()  # 自动调整子图参数
plt.savefig('q_nod_vs_xi.png', dpi=500, bbox_inches='tight', format='png')








GRAD1['q_sat_1'] = 6.1078*np.exp(17.2693882*GRAD1['Ta_1']/(GRAD1['Ta_1']+273.15-35.86))    

plt.figure()
plt.plot(GRAD1['q_1']*1000)

plt.plot(GRAD1['q_sat_1'])
plt.ylabel('q(g/kg)')

plt.figure()
plt.scatter(GRAD1['q_1']*10000/GRAD1['q_sat_1'],GRAD1['RH_1'])




#%%风应力拖曳系数和10m风速之间关系

import pandas as pd

# 假设我们已经有了 FLUX1-FLUX9 和 GRAD1-GRAD9 的 DataFrames
# 这里演示如何计算并添加 Cd 列

def calculate_cd(flux_df, grad_df):
    """
    计算 Cd 并添加到 FLUX DataFrame
    :param flux_df: FLUX DataFrame (包含 USTAR 列)
    :param grad_df: GRAD DataFrame (包含 WS_3 列)
    :return: 添加了 Cd 列的 FLUX DataFrame
    """
    # 确保数据对齐 - 假设两个DataFrame的索引已经对齐
    # 或者可以通过共同的时间戳列合并
    
    # 方法1：如果索引已经对齐
    flux_df['Cd'] = np.sqrt( (flux_df['USTAR']**2) / (grad_df['WS_3']**2 )   )
    
    # 方法2：如果需要通过时间戳合并（更安全的方法）
    # merged = pd.merge(flux_df, grad_df[['timestamp', 'WS_3']], on='timestamp')
    # flux_df['Cd'] = (merged['USTAR'] ** 2) / (merged['WS_3'] ** 2)
    
    return flux_df

# 示例使用 - 处理 FLUX1 到 FLUX9
for i in range(1, 10):
    flux_name = f'FLUX{i}'
    grad_name = f'GRAD{i}'
    
    # 获取DataFrame - 这里假设它们已经在内存中
    # 实际使用时可能需要从文件或数据库加载
    flux_df = globals().get(flux_name)
    grad_df = globals().get(grad_name)
    
    if flux_df is not None and grad_df is not None:
        # 计算并添加Cd列
        flux_df = calculate_cd(flux_df, grad_df)
        
        # 将结果存回原变量
        globals()[flux_name] = flux_df
        
        print(f"成功为 {flux_name} 添加 Cd 列")
    else:
        print(f"警告: {flux_name} 或 {grad_name} 不存在")

# 处理后的DataFrames现在包含Cd列
#%%  Cd和U10的关系图

import matplotlib.pyplot as plt

# 创建图形
plt.figure(figsize=(12, 9))

# 绘制九个子图
for i in range(1, 10):
    ax = plt.subplot(3, 3, i)
    
    # 直接使用FLUX和GRAD变量
    ws = globals()[f'GRAD{i}']['WS_3']
    cd = globals()[f'FLUX{i}']['Cd']
    
    # 绘制散点图
    ax.scatter(ws, cd, s=12, alpha=0.6, c='steelblue', edgecolor='white', linewidth=0.3)
    
    # 仅边缘子图显示标签
    if i in [7, 8, 9]:
        ax.set_xlabel('U$_{10}$ [m/s]', fontsize=10)
    if i in [1, 4, 7]:
        ax.set_ylabel('Cd', fontsize=10)
    
    # 统一坐标范围（根据需要调整）
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.5)
    
    # 添加子图编号
    ax.text(0.05, 0.92, f'({chr(96+i)})', transform=ax.transAxes, 
            fontsize=12, fontweight='bold')
    
    # 添加网格线
    ax.grid(True, linestyle=':', alpha=0.4)

# 调整布局并保存
plt.tight_layout(pad=2.5, h_pad=1.5, w_pad=1.5)
plt.savefig('Cd_vs_U10_scatter.png', dpi=300, bbox_inches='tight')
plt.show()



#%%ustar/u10 vs u10
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# 1. 定义统一拟合函数（幂律）
def power_law(x, a, b):
    return a * np.power(x, b)

# 2. 创建图形
plt.figure(figsize=(12, 9))

for i in range(1, 10):
    ax = plt.subplot(3, 3, i)
    
    # 数据准备
    ws = globals()[f'GRAD{i}']['WS_3'].values
    cd = globals()[f'FLUX{i}']['Cd'].values
    valid = (ws > 0.1) & (cd > 0) & (~np.isnan(ws)) & (~np.isnan(cd))
    x, y = ws[valid], cd[valid]
    
    # 野点去除（基于IQR方法）
    if len(x) > 10:
        y_IQR = stats.iqr(y)
        y_median = np.median(y)
        y_valid = (y > (y_median - 3*y_IQR)) & (y < (y_median + 3*y_IQR))
        x, y = x[y_valid], y[y_valid]
    
    # 幂律拟合（带异常处理）
    try:
        popt, pcov = curve_fit(power_law, x, y, bounds=([1e-6, -10], [1, 0]), maxfev=5000)
        x_fit = np.linspace(0, 20, 100)
        y_fit = power_law(x_fit, *popt)
        
        # 计算R²
        residuals = y - power_law(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 绘制拟合曲线（带95%置信区间）
        ax.plot(x_fit, y_fit, 'r-', lw=1.5)
        ax.text(0.95, 0.9, f"$Cd^{{{1/2}}}={popt[0]:.2f}U10^{{{popt[1]:.2f}}}$\n$R²={r_squared:.2f}$", 
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    except Exception as e:
        print(f"Station {i} fit failed: {str(e)}")
    
    # 绘制散点（带透明度）
    ax.scatter(x, y, s=10, alpha=0.5, c='steelblue', edgecolor='white', linewidth=0.3)
    
    # 坐标轴设置
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.5)
    #ax.set_xticks(np.arange(0, 21, 5))
    #ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 仅边缘子图显示标签
    if i in [7, 8, 9]:
        ax.set_xlabel('$U_{10}$ (m/s)', fontsize=9)
    if i in [1, 4, 7]:
        ax.set_ylabel('$C_d^{1/2}$', fontsize=9)
    
    # 添加子图编号
    ax.text(0.05, 0.8, f"({chr(96+i)})", transform=ax.transAxes,
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

# 3. 调整布局并保存
plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
plt.savefig('Cd_vs_U10_cleaned.png', dpi=300, bbox_inches='tight')
#plt.close()


#%%Ri 和 L 之间关系对比（计算）

#计算Ri
import numpy as np
import pandas as pd

# 定义计算Richardson数的函数（使用T_SONIC作为参考温度）
def calculate_richardson_with_sonic_temp(df, g=9.81):
    """
    使用超声温度计算Richardson数并添加到DataFrame
    Ri = -g/T_SONIC * dtdz / (dudz**2)
    
    参数:
        df: 输入的DataFrame (必须包含dtdz、dudz和T_SONIC列)
        g: 重力加速度 (默认9.81 m/s²)
        
    返回:
        添加了Ri列的DataFrame
    """
    # 检查必要列是否存在
    required_cols = ['dtdz', 'dudz', 'T_SONIC']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"缺少必要列: {missing}")

    # 计算Ri（自动处理除零和无效值）
    with np.errstate(divide='ignore', invalid='ignore'):
        ri = (-g / df['T_SONIC']) * df['dtdz'] / (df['dudz']**2)
    
    # 替换无穷大和NaN
    df['Ri'] = np.where(np.isfinite(ri), ri, np.nan)
    
    return df

# 对FLUX1到FLUX9执行计算
for i in range(1, 10):
    df_name = f'FLUX{i}'
    try:
        if df_name in globals():
            # 执行计算
            globals()[df_name] = calculate_richardson_with_sonic_temp(globals()[df_name])
            print(f"成功为 {df_name} 添加Ri列")
            
            # 验证结果
            print(f"{df_name} Ri统计:\n", 
                  globals()[df_name]['Ri'].describe(), "\n")
        else:
            print(f"警告: {df_name} 不存在")
    except Exception as e:
        print(f"处理 {df_name} 时出错: {str(e)}")

# 示例输出（显示FLUX1的前5行Ri值）
if 'FLUX1' in globals():
    print("FLUX1前5行Ri计算结果示例:")
    print(FLUX1[['T_SONIC', 'dtdz', 'dudz', 'Ri']].head())



#%%Ri 和 L 之间关系对比（图）
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# 站点名称对应
folder_prefixes = ['MAWORS', 'NADORS', 'PLAN', 'QOMO_B', 'SETORS', 'NIMA', 'BANG', 'MANA', 'CHDU']

# 创建图形（设置期刊要求的尺寸和DPI）
fig, axes = plt.subplots(3, 3, figsize=(10, 8), dpi=300, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距

# 统一坐标轴范围
zL_limits = (-1, 1)  # 根据实际数据调整
Ri_limits = (-1, 1)   # Richardson数典型范围

# 循环处理每个站点
for i, (ax, site) in enumerate(zip(axes.flat, folder_prefixes), start=1):
    # 获取数据
    df = globals()[f'FLUX{i}']
    z_obs = df['z_obs'].values
    L = df['L'].values
    Ri = df['Ri'].values
    
    # 计算z/L（避免除以零）
    with np.errstate(divide='ignore', invalid='ignore'):
        z_over_L = z_obs / L
    
    # 过滤无效值
    valid = np.isfinite(z_over_L) & np.isfinite(Ri)
    z_over_L = z_over_L[valid]
    Ri = Ri[valid]
    
    # 绘制散点（使用科学配色）
    sc = ax.scatter(z_over_L, Ri, s=15, alpha=0.6, 
                   c='#1f77b4', edgecolor='w', linewidth=0.3)
    
    # 添加趋势线（可选）
    if len(z_over_L) > 10:
        try:
            # 使用局部加权散点平滑（LOWESS）
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(Ri, z_over_L, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', lw=1.5, zorder=3)
        except:
            pass
    
    # 设置站点标题
    ax.set_title(site, fontsize=10, pad=5, fontstyle='italic')
    
    # 添加参考线
    ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
    ax.axhline(0.25, color='k', linestyle='--', lw=0.8, alpha=0.7)
    ax.text(0.05, 0.28, 'Ri=0.25', fontsize=7, transform=ax.transAxes)
    
    # 坐标轴设置
    ax.set_xlim(zL_limits)
    ax.set_ylim(Ri_limits)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.3)
    
    # 仅边缘子图显示标签
    if i in [7, 8, 9]:
        ax.set_xlabel('$z/L$', fontsize=9, labelpad=2)
    if i in [1, 4, 7]:
        ax.set_ylabel('$Ri$', fontsize=9, labelpad=2)

# 添加整体标注
#fig.text(0.5, 0.95, 'Stability Analysis: $Ri$ vs $z/L$', 
#         ha='center', va='top', fontsize=11, fontweight='bold')
#fig.text(0.5, 0.91, 'Tibetan Plateau Flux Stations', 
#         ha='center', va='top', fontsize=9, alpha=0.7)

# 保存为出版质量图片
plt.savefig('Ri_vs_zL_all_stations.png', 
            dpi=600, 
            bbox_inches='tight', 
            facecolor='white',
            format='png')

plt.show()


# 站点名称对应
folder_prefixes = ['MAWORS', 'NADORS', 'PLAN', 'QOMO_B', 'SETORS', 'NIMA', 'BANG', 'MANA', 'CHDU']

# 创建图形（设置期刊要求的尺寸和DPI）
fig, axes = plt.subplots(3, 3, figsize=(10, 8), dpi=300, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距

# 统一坐标轴范围
zL_limits = (-1, 1)  # 根据实际数据调整
Ri_limits = (-1, 1)   # Richardson数典型范围

# 循环处理每个站点
for i, (ax, site) in enumerate(zip(axes.flat, folder_prefixes), start=1):
    # 获取数据
    df = globals()[f'FLUX{i}']
    z_obs = df['z_obs'].values
    L = df['L'].values
    Ri = df['Ri'].values
    
    # 计算z/L（避免除以零）
    with np.errstate(divide='ignore', invalid='ignore'):
        z_over_L = z_obs / L
    
    # 过滤无效值
    valid = np.isfinite(z_over_L) & np.isfinite(Ri)
    z_over_L = z_over_L[valid]
    Ri = Ri[valid]
    
    # 绘制散点（使用科学配色）
    sc = ax.scatter(Ri,z_over_L, s=15, alpha=0.6, 
                   c='#1f77b4', edgecolor='w', linewidth=0.3)
    
    # 添加趋势线（可选）
    if len(z_over_L) > 10:
        try:
            # 使用局部加权散点平滑（LOWESS）
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(z_over_L,Ri , frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', lw=1.5, zorder=3)
        except:
            pass
    
    # 设置站点标题
    ax.set_title(site, fontsize=10, pad=5, fontstyle='italic')
    
    # 添加参考线
    ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
    ax.axhline(0.25, color='k', linestyle='--', lw=0.8, alpha=0.7)
    ax.text(0.05, 0.28, 'Ri=0.25', fontsize=7, transform=ax.transAxes)
    
    # 坐标轴设置
    ax.set_xlim(zL_limits)
    ax.set_ylim(Ri_limits)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.3)
    
    # 仅边缘子图显示标签
    if i in [7, 8, 9]:
        ax.set_xlabel('$Ri$', fontsize=9, labelpad=2)
    if i in [1, 4, 7]:
        ax.set_ylabel('$z/L$', fontsize=9, labelpad=2)



# 添加整体标注
#fig.text(0.5, 0.95, 'Stability Analysis: $Ri$ vs $z/L$', 
#         ha='center', va='top', fontsize=11, fontweight='bold')
#fig.text(0.5, 0.91, 'Tibetan Plateau Flux Stations', 
#         ha='center', va='top', fontsize=9, alpha=0.7)

# 保存为出版质量图片
plt.savefig('zL_over_Ri_all_stations.png', 
            dpi=600, 
            bbox_inches='tight', 
            facecolor='white',
            format='png')

plt.show()
