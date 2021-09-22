import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


############################################
# GÖREV 1- Veri Ön İşleme
############################################

def load_dataset():
    df_ = pd.read_excel("VERİSETLERİ/online_retail_II.xlsx", sheet_name="Year 2010-2011")
    return df_


df_ = load_dataset()
df = df_.copy()


def check_df(dataframe, head=5, tail=5):
    print("########### HEAD #############")
    print(dataframe.head(head))

    print("########### TAIL #############")
    print(dataframe.tail(tail))

    print("########### DESCRIBE #############")
    print(dataframe.describe().T)

    print("########### INFO #############")
    print(dataframe.info())

    print("########### SHAPE #############")
    print(dataframe.shape)

    print("########### DTYPES #############")
    print(dataframe.dtypes)


check_df(df)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)


############################################
# GÖREV 2- ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

def get_country(dataframe, country):
    print("######## Country Values #########")
    print(dataframe["Country"].value_counts().sort_values(ascending=False))

    df_gr = dataframe[dataframe["Country"] == country]
    return df_gr


df_gr = get_country(df, "Germany")


def create_invoice_product_df(datafame, id=False):
    if id:
        data = datafame.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        data = datafame.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

    return data


gr_inv_pro_df = create_invoice_product_df(df_gr, id=True)


def stockcode_to_name(dataframe, code):
    product = dataframe[dataframe["StockCode"] == code][["Description"]].values[0].tolist()
    print(f"id: {code}, name: {product}")


############################################
# GÖREV 3- Birliktelik Kurallarının Çıkarılması
############################################
# apriori
frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)

# association_rules
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

############################################
# GÖREV 4- ID'leri verilen ürünlerin isimleri nelerdir?
############################################

stockcode_to_name(df_gr, 21987)
stockcode_to_name(df_gr, 23235)
stockcode_to_name(df_gr, 22747)


############################################
# GÖREV 5- SEPETTEKİ KULLANICILARA ÜRÜN ÖNERİSİ
############################################

def recommender(rules, urun):
    rules_sorts = rules.sort_values("lift", ascending=False)
    recommendation_list = []

    for index, product in enumerate(rules_sorts["antecedents"]):
        for j in list(product):
            if j == urun:
                recommendation_list.append(list(rules_sorts.iloc[index]["consequents"]))
    recom_products = list({item for item_list in recommendation_list for item in item_list})

    return recom_products[:3]


# Ürün Önerme ve Önerilen Ürünlerin İsimleri
recommender_products = recommender(rules, 21987)
for id in recommender_products:
    stockcode_to_name(df_gr, id)
