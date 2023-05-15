import pandas as pd
import re
def data_clean(claims):
    
    claims = claims.lower()
    
    claims = claims.replace('\n', ' ')
    claim = "".join(e for e in claims)
    
    claim = re.findall(r"\b\d+\s?(?:\.|\)|\:|-)\s?.+?\s*(?:\.\s*|$)", claim, flags=re.DOTALL)
    data_clean.count += 1

    if claim:
        return claim[0]
    else:
        print(data_clean.count)
        return claims
data_clean.count = -1
def process():
    #read csv
    df = pd.read_csv("~/patent_classification/data/ori_data.csv")

    #remove number < 10 in cpc_class
    value_counts = df['cpc_code'].value_counts()
    to_remove = value_counts[value_counts <= 10].index.tolist()
    df = df[~df['cpc_code'].isin(to_remove)]

    #cpc code column get dummy 
    dummies_df = pd.get_dummies(df['cpc_code'])
    label_list = df['cpc_code'].unique().tolist()
    new_df = pd.concat([df, dummies_df], axis=1)
    new_df = new_df.groupby('publication_number')[label_list].sum().applymap(lambda x: 1 if x >= 1 else 0)

    # 將原始DataFrame與groupby後的結果合併
    df_merged = pd.merge(df.drop_duplicates(subset='publication_number'), new_df, how='right',on='publication_number')
    df_merged = df_merged.drop(columns = ['cpc_code'], axis = 1)
    print('123')
    #clean data
    df_merged['claims'] = df_merged['claims'].apply(data_clean)
    df_merged['abstract&claim'] = df_merged['abstract'] + df_merged['claims']
    return df_merged
if __name__ == '__main__':
    out_df = process()
    out_df.to_csv('/home/p76101372/patent_classification/data/preprocess_df.csv',index=False)