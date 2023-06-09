import pandas as pd
import re
import nltk
def data_clean(claims):
    
    claims = claims.lower()
    
    claims = claims.replace('\n', ' ')
    claim = "".join(e for e in claims)
    #extract first claim
    claim = re.findall(r"\b\d+\s?(?:\.|\)|\:|-)\s?.+?\s*(?:\.\s*|$)", claim, flags=re.DOTALL)
    data_clean.count += 1

    if claim:
        return claim[0]
    else:
        print(data_clean.count)
        return claims
def extract_noun(claims):
    words = nltk.word_tokenize(claims)
    tags = nltk.pos_tag(words) # 对单个字符进行标注
    NN = [s1 for (s1,s2) in tags if s2 in ['NN', 'NNP']]
    #对list列表的tags的两个变量进行判断（s1代表第一个变量，s2代表第二个变量）
    #提取出tags的NN和NNP单词。NN表示普通名词，NNP表示专有名词
    result = ' '.join(NN)
    return result
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
    #clean data
    df_merged['claims'] = df_merged['claims'].apply(data_clean)
    #extract noun
    # df_merged['claims'] = df_merged['claims'].apply(extract_noun)
    #merge claim and abstract
    df_merged['abstract&claim'] = df_merged['abstract'] + df_merged['claims']
    return df_merged
if __name__ == '__main__':
    out_df = process()
    out_df.to_csv('/home/p76101372/patent_classification/data/preprocess_df_without_extract_noun.csv',index=False)
