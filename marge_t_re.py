import pandas as pd

a = pd.read_csv('./Data/valid.csv')
print(a)
print(a.columns)

a['tmp'] = a['text']+a['reply']
del a['text']
del a['reply']
a=a[['idx', 'context_idx', 'tmp', 'label']]
a= a.rename(columns={"tmp": "text"})
a.to_csv('./Data/valid.csv', index=False)
