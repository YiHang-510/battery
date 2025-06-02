import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载文件
i = 1
Vrlx = np.load(f'F:\\code\\battery\\data\\selected_data\\battery{i}\\Vrlx.npy', allow_pickle=True)
EndVrlx = np.load(f'F:\\code\\battery\\data\\selected_data\\battery{i}\\EndVrlx.npy', allow_pickle=True)

# 查看类型和长度
print(type(Vrlx), len(Vrlx))          # list of lists
print(type(EndVrlx), len(EndVrlx))    # list or ndarray

# 打印前几个
print("第1条弛豫电压序列：", Vrlx[0])
print("末端电压序列前10项：", EndVrlx[:10])

# 保存 EndVrlx 为 CSV
pd.DataFrame(EndVrlx).to_csv("EndVrlx.csv", index=False, header=["末端电压"])

# 保存 Vrlx（多列展开）为 CSV
pd.DataFrame(Vrlx).to_csv("Vrlx.csv", index=False)

plt.figure()
plt.plot(EndVrlx)
plt.xlabel('Cycle')
plt.ylabel('Relaxation End Voltage (V)')
plt.title('EndVrlx vs Cycle')
plt.grid(True)
plt.show()