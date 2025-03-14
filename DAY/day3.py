import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm


# 设置中文字体路径
font_path = 'F:/github/QuantPY/fonts/dingliezhuhaifont-20240831GengXinBan)-2.ttf'  # 示例路径
my_font = fm.FontProperties(fname=font_path)

# 创建 Ticker 对象
ticker = yf.Ticker("600036.SS")  # 招商银行的股票代码

# 获取今日的历史市场数据
today_data = ticker.history(period="1d", interval="1m")  # 获取今天的数据，以1分钟为间隔

print(today_data)
# 转换时间为北京时间
today_data.index = today_data.index.tz_convert('Asia/Shanghai')

# 筛选出有效的时间段，去掉休盘时间
morning_data = today_data.between_time('09:30', '11:30')  # 上午交易时间
afternoon_data = today_data.between_time('13:00', '15:00')  # 下午交易时间

# 使用 pd.concat() 合并两个时间段的数据
filtered_data = pd.concat([morning_data, afternoon_data])

# 绘制走势图
plt.figure(figsize=(12, 6))
plt.plot(filtered_data.index, filtered_data['Close'], label='收盘价', color='blue', linewidth=2)

# 添加标题和标签
plt.title('招商银行今日走势图', fontproperties=my_font, fontsize=16)
plt.xlabel('时间 (北京时间)', fontproperties=my_font, fontsize=14)
plt.ylabel('价格 (CNY)', fontproperties=my_font, fontsize=14)

# 设置 x 轴范围，确保时间格式正确
plt.xlim(filtered_data.index[0], filtered_data.index[-1])

# 将时间戳格式化为时:分
xticks_labels = [dt.strftime('%H:%M') for dt in filtered_data.index[::10]]
plt.xticks(filtered_data.index[::10], xticks_labels, rotation=45, fontproperties=my_font)  # 每10个点显示一个刻度

# 添加网格和图例
plt.grid()
plt.legend(prop={'size': 12, 'family': 'SimHei'})  # 设置图例字体
plt.tight_layout()

# 显示图形
plt.show()

# 显示上午和下午的数据
print("上午交易数据:")
print(morning_data)

print("\n下午交易数据:")
print(afternoon_data)
