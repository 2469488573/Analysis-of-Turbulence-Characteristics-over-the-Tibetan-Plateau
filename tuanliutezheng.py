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





#%%#以一个站点为例，考察湍流特征



#%%#计算无量纲动量梯度


FLUX1['u_no_dim'] =  kaman * FLUX1['z_obs'] * 0.25*(  (GRAD1['WS_5']-GRAD1['WS_1'])/(GRAD1['z5']-GRAD1['z1']) +
                                                    (GRAD1['WS_4']-GRAD1['WS_1'])/(GRAD1['z4']-GRAD1['z1'])+
                                                    (GRAD1['WS_3']-GRAD1['WS_1'])/(GRAD1['z3']-GRAD1['z1'])+
                                                    (GRAD1['WS_2']-GRAD1['WS_1'])/(GRAD1['z2']-GRAD1['z1'])) / FLUX1['USTAR'] 
FLUX1['xi'] = FLUX1['z_obs']/FLUX1['L']
#------------------------------------------------------------------------------
FLUX2['u_no_dim'] =  kaman * FLUX2['z_obs'] * 0.25*(  (GRAD2['WS_5']-GRAD2['WS_1'])/(GRAD2['z5']-GRAD2['z1']) +
                                                    (GRAD2['WS_4']-GRAD2['WS_1'])/(GRAD2['z4']-GRAD2['z1'])+
                                                    (GRAD2['WS_3']-GRAD2['WS_1'])/(GRAD2['z3']-GRAD2['z1'])+
                                                    (GRAD2['WS_2']-GRAD2['WS_1'])/(GRAD2['z2']-GRAD2['z1'])) / FLUX2['USTAR'] 
FLUX2['xi'] = FLUX2['z_obs']/FLUX2['L']
#------------------------------------------------------------------------------
FLUX3['u_no_dim'] =  kaman * FLUX3['z_obs'] * 0.25*(  (GRAD3['WS_5']-GRAD3['WS_1'])/(GRAD3['z5']-GRAD3['z1']) +
                                                    (GRAD3['WS_4']-GRAD3['WS_1'])/(GRAD3['z4']-GRAD3['z1'])+
                                                    (GRAD3['WS_3']-GRAD3['WS_1'])/(GRAD3['z3']-GRAD3['z1'])+
                                                    (GRAD3['WS_2']-GRAD3['WS_1'])/(GRAD3['z2']-GRAD3['z1'])) / FLUX3['USTAR'] 
FLUX3['xi'] = FLUX3['z_obs']/FLUX3['L']
#------------------------------------------------------------------------------
FLUX4['u_no_dim'] =  kaman * FLUX4['z_obs'] * 0.25*(  (GRAD4['WS_5']-GRAD4['WS_1'])/(GRAD4['z5']-GRAD4['z1']) +
                                                    (GRAD4['WS_4']-GRAD4['WS_1'])/(GRAD4['z4']-GRAD4['z1'])+
                                                    (GRAD4['WS_3']-GRAD4['WS_1'])/(GRAD4['z3']-GRAD4['z1'])+
                                                    (GRAD4['WS_2']-GRAD4['WS_1'])/(GRAD4['z2']-GRAD4['z1'])) / FLUX4['USTAR'] 
FLUX4['xi'] = FLUX4['z_obs']/FLUX4['L']
#------------------------------------------------------------------------------
FLUX5['u_no_dim'] =  kaman * FLUX5['z_obs'] * 0.25*(  (GRAD5['WS_5']-GRAD5['WS_1'])/(GRAD5['z5']-GRAD5['z1']) +
                                                    (GRAD5['WS_4']-GRAD5['WS_1'])/(GRAD5['z4']-GRAD5['z1'])+
                                                    (GRAD5['WS_3']-GRAD5['WS_1'])/(GRAD5['z3']-GRAD5['z1'])+
                                                    (GRAD5['WS_2']-GRAD5['WS_1'])/(GRAD5['z2']-GRAD5['z1'])) / FLUX5['USTAR'] 
FLUX5['xi'] = FLUX5['z_obs']/FLUX5['L']
#------------------------------------------------------------------------------
FLUX6['u_no_dim'] =  kaman * FLUX6['z_obs'] * 0.25*(  (GRAD6['WS_5']-GRAD6['WS_1'])/(GRAD6['z5']-GRAD6['z1']) +
                                                    (GRAD6['WS_4']-GRAD6['WS_1'])/(GRAD6['z4']-GRAD6['z1'])+
                                                    (GRAD6['WS_3']-GRAD6['WS_1'])/(GRAD6['z3']-GRAD6['z1'])+
                                                    (GRAD6['WS_2']-GRAD6['WS_1'])/(GRAD6['z2']-GRAD6['z1'])) / FLUX6['USTAR'] 
FLUX6['xi'] = FLUX6['z_obs']/FLUX6['L']
#------------------------------------------------------------------------------
FLUX7['u_no_dim'] =  kaman * FLUX7['z_obs'] * 0.25*(  (GRAD7['WS_5']-GRAD7['WS_1'])/(GRAD7['z5']-GRAD7['z1']) +
                                                    (GRAD7['WS_4']-GRAD7['WS_1'])/(GRAD7['z4']-GRAD7['z1'])+
                                                    (GRAD7['WS_3']-GRAD7['WS_1'])/(GRAD7['z3']-GRAD7['z1'])+
                                                    (GRAD7['WS_2']-GRAD7['WS_1'])/(GRAD7['z2']-GRAD7['z1'])) / FLUX7['USTAR'] 
FLUX7['xi'] = FLUX7['z_obs']/FLUX7['L']
#------------------------------------------------------------------------------

FLUX8['u_no_dim'] =  kaman * FLUX8['z_obs'] * 0.25*(  (GRAD8['WS_5']-GRAD8['WS_1'])/(GRAD8['z5']-GRAD8['z1']) +
                                                    (GRAD8['WS_4']-GRAD8['WS_1'])/(GRAD8['z4']-GRAD8['z1'])+
                                                    (GRAD8['WS_3']-GRAD8['WS_1'])/(GRAD8['z3']-GRAD8['z1'])+
                                                    (GRAD8['WS_2']-GRAD8['WS_1'])/(GRAD8['z2']-GRAD8['z1'])) / FLUX8['USTAR'] 
FLUX8['xi'] = FLUX8['z_obs']/FLUX8['L']
#------------------------------------------------------------------------------
FLUX9['u_no_dim'] =  kaman * FLUX9['z_obs'] * 0.25*(  (GRAD9['WS_5']-GRAD9['WS_1'])/(GRAD9['z5']-GRAD9['z1']) +
                                                    (GRAD9['WS_4']-GRAD9['WS_1'])/(GRAD9['z4']-GRAD9['z1'])+
                                                    (GRAD9['WS_3']-GRAD9['WS_1'])/(GRAD9['z3']-GRAD9['z1'])+
                                                    (GRAD9['WS_2']-GRAD9['WS_1'])/(GRAD9['z2']-GRAD9['z1'])) / FLUX9['USTAR'] 
FLUX9['xi'] = FLUX9['z_obs']/FLUX9['L']
#------------------------------------------------------------------------------
#%%无量纲动量和稳定度参数之间关系图


plt.figure(figsize=(10, 10)) 


plt.subplot(331)
plt.scatter(FLUX1['xi'], FLUX1['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[0])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(332)
plt.scatter(FLUX2['xi'], FLUX2['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[1])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(333)
plt.scatter(FLUX3['xi'], FLUX3['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[2])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(334)
plt.scatter(FLUX4['xi'], FLUX4['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[3])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(335)
plt.scatter(FLUX5['xi'], FLUX5['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[4])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(336)
plt.scatter(FLUX6['xi'], FLUX6['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[5])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(337)
plt.scatter(FLUX7['xi'], FLUX7['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[6])
plt.xlabel('z/L', fontsize=20, fontfamily='Arial')
plt.ylabel(r'$\frac{\kappa z}{u_*}\frac{du}{dz}$', fontsize=18, fontfamily='Arial')

plt.subplot(338)
plt.scatter(FLUX8['xi'], FLUX8['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[7])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(339)
plt.scatter(FLUX9['xi'], FLUX9['u_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.xlim(-10,10)
plt.ylim(-10,40)
plt.title(position_name[8])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

# 调整tight_layout参数
plt.tight_layout(pad=0, h_pad=1, w_pad=1, rect=[0, 0.03, 1, 0.95])

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


#%%温度无量纲梯度和稳定度参数之间相关系数


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

#%%计算梯度和ri等
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
FLUX2['dudz_2']  = (GRAD2['WS_2'] - GRAD2['WS_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX4['dudz_2']  = (GRAD4['WS_2'] - GRAD4['WS_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dudz_2']  = (GRAD5['WS_2'] - GRAD5['WS_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dudz_2']  = (GRAD6['WS_2'] - GRAD6['WS_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dudz_2']  = (GRAD7['WS_2'] - GRAD7['WS_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dudz_2']  = (GRAD8['WS_2'] - GRAD8['WS_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dudz_2']  = (GRAD9['WS_2'] - GRAD9['WS_1'] )/(GRAD9['z2']-GRAD9['z1'])

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
FLUX2['dtdz_2']  = (GRAD2['Ta_2'] - GRAD2['Ta_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX4['dtdz_2']  = (GRAD4['Ta_2'] - GRAD4['Ta_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dtdz_2']  = (GRAD5['Ta_2'] - GRAD5['Ta_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dtdz_2']  = (GRAD6['Ta_2'] - GRAD6['Ta_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dtdz_2']  = (GRAD7['Ta_2'] - GRAD7['Ta_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dtdz_2']  = (GRAD8['Ta_2'] - GRAD8['Ta_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dtdz_2']  = (GRAD9['Ta_2'] - GRAD9['Ta_1'] )/(GRAD9['z2']-GRAD9['z1'])


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
FLUX2['dqdz_2']  = (GRAD2['q_2'] - GRAD2['q_1'] )/(GRAD2['z2']-GRAD2['z1'])
FLUX4['dqdz_2']  = (GRAD4['q_2'] - GRAD4['q_1'] )/(GRAD4['z2']-GRAD4['z1'])
FLUX5['dqdz_2']  = (GRAD5['q_2'] - GRAD5['q_1'] )/(GRAD5['z2']-GRAD5['z1'])
FLUX6['dqdz_2']  = (GRAD6['q_2'] - GRAD6['q_1'] )/(GRAD6['z2']-GRAD6['z1'])
FLUX7['dqdz_2']  = (GRAD7['q_2'] - GRAD7['q_1'] )/(GRAD7['z2']-GRAD7['z1'])
FLUX8['dqdz_2']  = (GRAD8['q_2'] - GRAD8['q_1'] )/(GRAD8['z2']-GRAD8['z1'])
FLUX9['dqdz_2']  = (GRAD9['q_2'] - GRAD9['q_1'] )/(GRAD9['z2']-GRAD9['z1'])


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
plt.ylim(-5,10)
plt.title(position_name[0])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{q_*}\frac{dq}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(332)
plt.scatter(FLUX2['xi'], FLUX2['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[1])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(333)
plt.scatter(FLUX3['xi'], FLUX3['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[2])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(334)
plt.scatter(FLUX4['xi'], FLUX4['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[3])
#plt.xlabel('z/L')
plt.ylabel(r'$\frac{\kappa z}{q_*}\frac{dq}{dz}$', fontsize=20, fontfamily='Arial')

plt.subplot(335)
plt.scatter(FLUX5['xi'], FLUX5['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[4])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(336)
plt.scatter(FLUX6['xi'], FLUX6['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[5])
#plt.xlabel('z/L')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(337)
plt.scatter(FLUX7['xi'], FLUX7['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[6])
plt.xlabel('z/L', fontsize=20, fontfamily='Arial')
plt.ylabel(r'$\frac{\kappa z}{q_*}\frac{dq}{dz}$', fontsize=18, fontfamily='Arial')

plt.subplot(338)
plt.scatter(FLUX8['xi'], FLUX8['q_no_dim'], s=0.1,c='black')
plt.xlim(-10,10)
plt.xlim(-10,10)
plt.ylim(-5,10)
plt.title(position_name[7])
plt.xlabel('z/L', fontsize=18, fontfamily='Arial')
#plt.ylabel('kz*dudz/ustar')

plt.subplot(339)
plt.scatter(FLUX9['xi'], FLUX9['q_no_dim'], s=0.1,c='black')
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
plt.savefig('q_nod_vs_xi.png', dpi=500, bbox_inches='tight', format='png')


#计算一下相关，比湿无量纲梯度和稳定度参数之间相关系数。
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 FLUX1 到 FLUX9 已经是加载好的DataFrame
tables = {
    'FLUX1': FLUX1, 'FLUX2': FLUX2, 'FLUX3': FLUX3,
    'FLUX4': FLUX4, 'FLUX5': FLUX5, 'FLUX6': FLUX6,
    'FLUX7': FLUX7, 'FLUX8': FLUX8, 'FLUX9': FLUX9
}

# 1. 计算各表相关系数
results = []
for name, df in tables.items():
    # 数据清洗
    clean_df = df[['xi', 'q_no_dim']].dropna()
    clean_df = clean_df[clean_df['xi'].abs() <= 50]
    
    n_samples = len(clean_df)
    
    if n_samples >= 2:  # 至少需要2个样本计算相关
        # 计算三种相关系数
        pearson_r, pearson_p = stats.pearsonr(clean_df['xi'], clean_df['q_no_dim'])
        spearman_r, spearman_p = stats.spearmanr(clean_df['xi'], clean_df['q_no_dim'])
        kendall_tau, kendall_p = stats.kendalltau(clean_df['xi'], clean_df['q_no_dim'])
        
        results.append({
            'Table': name,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Kendall_tau': kendall_tau,
            'Sample_Size': n_samples
        })
    else:
        print(f"警告: {name} 有效样本不足({n_samples})，跳过计算")

# 转换为DataFrame
result_df = pd.DataFrame(results)

# 2. 打印美观的汇总表格
print("═"*50)
print(f"{'相关系数汇总表':^50}")
print("═"*50)
print(result_df.round(3).to_string(index=False))
print("═"*50)
print("注: p值<0.05表示相关性显著")

# 3. 保存结果到Excel
with pd.ExcelWriter('correlation_results.xlsx') as writer:
    result_df.to_excel(writer, sheet_name='汇总', index=False)
    
    # 添加统计摘要
    summary = result_df.describe().loc[['mean', 'std', 'min', 'max']]
    summary.to_excel(writer, sheet_name='统计摘要')
    
print("\n结果已保存到 correlation_results.xlsx")

# 4. 可视化（3x3子图）
plt.figure(figsize=(15, 12))
for i, (name, df) in enumerate(tables.items(), 1):
    plt.subplot(3, 3, i)
    
    # 绘制散点图+回归线
    sns.regplot(
        x='xi', y='q_no_dim', data=df,
        scatter_kws={'s': 20, 'alpha': 0.6, 'color': 'steelblue'},
        line_kws={'color': 'coral', 'lw': 2}
    )
    
    # 添加标注
    corr_data = result_df[result_df['Table']==name].iloc[0]
    plt.title(
        f"{name}\n"
        f"Pearson r={corr_data['Pearson_r']:.2f} (p={corr_data['Pearson_p']:.3f})\n"
        f"Spearman ρ={corr_data['Spearman_r']:.2f}",
        fontsize=9
    )
    plt.xlabel('xi', fontsize=8)
    plt.ylabel('q_no_dim', fontsize=8)

plt.tight_layout()
plt.savefig('correlation_plots.png', dpi=300, bbox_inches='tight')
print("可视化已保存为 correlation_plots.png")
plt.show()

# 5. 相关系数分布分析
plt.figure(figsize=(10, 5))
sns.boxplot(data=result_df[['Pearson_r', 'Spearman_r', 'Kendall_tau']], palette="Set2")
plt.title("三种相关系数分布比较", fontsize=14)
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel("相关系数值")
plt.savefig('correlation_distribution.png', dpi=300)
plt.show()






GRAD1['q_sat_1'] = 6.1078*np.exp(17.2693882*GRAD1['Ta_1']/(GRAD1['Ta_1']+273.15-35.86))    

plt.figure()
plt.plot(GRAD1['q_1']*1000)

plt.plot(GRAD1['q_sat_1'])
plt.ylabel('q(g/kg)')

plt.figure()
plt.scatter(GRAD1['q_1']*10000/GRAD1['q_sat_1'],GRAD1['RH_1'])


