import pandas as pd
import matplotlib.pyplot as plt

# 你的数据
data = [
    ['LOW', 9,9,9,8,6,4,4,3,3,3,3,3,3,3,2,2,2,2,3,3,4,4,3,3,3,3,2,2,2,2,3,2,2,2,2,2,1,1,1,1,5,5,4,4,3,3,3,3,3,3,5,3,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2],
    ['MEDIUM', 1,1,1,2,4,6,6,7,7,7,7,7,7,7,8,8,8,8,7,7,6,6,7,7,7,7,8,8,8,8,7,8,8,8,8,8,9,9,9,9,4,4,5,5,6,6,6,6,6,6,3,6,7,8,8,8,8,8,8,8,7,7,7,7,7,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7],
    ['HIGH', 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]]

# 将数据转换为 pandas DataFrame
df = pd.DataFrame(data, columns=['Type'] + [f'Move {i//10} Timestep {i%10}' for i in range(len(data[0]) - 1)])

# 转置 DataFrame，这样 Move 就会成为你的列
df = df.set_index('Type').T

# 创建折线图
plt.figure(figsize=[20,10])
plt.plot(df['LOW'], label='LOW', color='#5BDAF9')
plt.plot(df['MEDIUM'], label='MEDIUM', color='#F56C0D')
plt.plot(df['HIGH'], label='HIGH', color='#F71E10')

# 设置图表的其他元素
plt.legend(loc='upper right')
plt.ylabel('Value')
plt.xticks(range(0, len(df.index) + 1, 10), labels=[f'Move {i}' for i in range((len(df.index)//10)+1)], rotation=45)

# 展示图表
plt.show()
