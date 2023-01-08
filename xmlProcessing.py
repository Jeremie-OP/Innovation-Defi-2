import pandas as pd
from elasticsearch import Elasticsearch

def xmlToDf(xmlFile):
    # Read XML file
    df = pd.read_xml(xmlFile)
    return df

def DfToElastic(df):
    # Write to ElasticSearch
    es = Elasticsearch(hosts="https://localhost:9201", basic_auth=("elastic", "rKjHcYp17Rugtffw4K2i"))
    chunkSize = 50
    for i in range(0, len(df)):
        try:
            es.index(index="train", document=df.loc[i].to_json())
        except Exception as e:
            print(i)

df_train = xmlToDf('train.xml')
df_dev = xmlToDf('dev.xml')
# es = Elasticsearch(hosts="https://localhost:9201", basic_auth=("elastic", "rKjHcYp17Rugtffw4K2i"))
# es.index(index="train", document=df_train.loc[0].to_json())
DfToElastic(df_train)
print(list(df_train.columns))
print(list(df_dev.columns))