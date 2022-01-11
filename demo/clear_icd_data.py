from typing import List, Tuple, Dict, Any
import pandas as pd
"""
合并数据集

clear data

"""
# df =pd.read_csv("data/icd_pre/all_min.csv")
# df =pd.read_csv("data/icd_pre/疾病总表工作簿3.csv")
df =pd.read_csv("data/icd_pre/etl_diagnosis.csv")
df2 =pd.read_csv("data/icd_pre/诊断合集无处理.csv")
print(df)
# 去除重复

# new_df=df["sent"]
new=pd.concat([df2['sent'].str.strip(),df['sent'].str.strip()])
new.drop_duplicates()
new.dropna()

new.to_csv("data/icd_pre/icd_pred.csv")