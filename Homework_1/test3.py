#%%

import pandas as pd
df=pd.read_csv("car_complain.csv")

# %% 按照品牌进行排序

C_brand=df.groupby("brand")["id"].count()
print(C_brand.sort_values(ascending=False))

# %%按照车型进行排序
C_car_model=df.groupby("car_model")["id"].count()
print(C_car_model.sort_values(ascending=False))

 

# %%按照
df1=df.groupby(['brand'],as_index=False)['id'].agg({'投诉总数':"count"})

# %%
df2=df.groupby(["brand"],as_index=False)["car_model"].agg({"车型总数":"nunique"})

# %%
df3=pd.merge(df1,df2,how='left',on="brand")
df3["平均投诉"]=df3['投诉总数']/df3["车型总数"]
df3.sort_values("平均投诉",ascending=False)
#%%
print("品牌投诉前5")
print(df1.head())
print("-------------------------")
print("车型投诉前5")
print(df2.head())
print("-------------------------")
print("车型平均投诉前5")
print(df3.head())
