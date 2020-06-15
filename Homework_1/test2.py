#%%
import pandas as pd
df=pd.DataFrame()

df["姓名"]=["张飞","关羽","刘备","典韦","许诸"]
df["语文"]=[69,95,98,90,80]
df["数学"]=[65,76,86,88,90]
df["英语"]=[30,98,88,77,90]
df.set_index("姓名")
# %%
dict_1={"总和":"sum","平均":"mean","最小":"min","最大":"max","方差":"var","标准差":"std"}
for k,vv in dict_1.items():
    for kemu in df.columns.tolist()[1:]:
        print(kemu,k,df[kemu].agg(vv))
    print("------------------")


# %%
df["总成绩"]=df.sum(axis=1)
df.sort_values("总成绩",ascending=False)