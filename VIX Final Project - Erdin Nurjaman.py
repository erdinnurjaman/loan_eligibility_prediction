#!/usr/bin/env python
# coding: utf-8

# # Import Standard Packages

# In[1]:


import itertools
import joblib
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 1000)


# # Read the Dataset

# Membaca dataset yang akan digunakan

# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')


# # Data Understanding

# Cuplikan sekilas mengenai nilai yang terdapat didalam data

# In[5]:


df.head()


# ## Rincian Dataframe

# Melihat kelengkapan nilai dari semua kolom dalam dataframe

# In[6]:


df.info()


# ## Data Duplikat

# Melihat apakah ada data duplikat di dataframe

# In[7]:


df.duplicated().sum()


# ## Data Hilang

# Melihat berapa jumlah baris data yang hilang di setiap kolomnya

# In[8]:


df.isnull().sum()


# ## Dimensi Data

# Melihat dimensi data yaitu jumlah baris dan kolom dalam dataframe

# In[9]:


df.shape


# # Missing Values

# Cek persentase missing values pada dataframe

# In[10]:


feature_with_na = [features for features in df.columns if df[features].isnull().sum()>1]

for feature in feature_with_na:
    print(feature, np.round(df[feature].isnull().mean(), 4), '% Missing Values')


# # Define the Target Feature

# Melihat nilai unik pada target feature yaitu pada kolom **'loan_status'**

# In[11]:


df['loan_status'].value_counts()


# Dalam project ini, saya ditugaskan untuk menentukan pinjaman mana yang berkemungkinan untuk gagal bayar. Dengan menggunakan dataset yang diberikan, saya akan mengklasifikasikan
# pelanggan ke dalam 2 kategori target yaitu **Berhasil** dan **Gagal**. Dikarenakan nilai unik pada **'loan_status'** ada banyak, kita akan memasukan kategori tersebut
# pada angka biner 1 dan 0, dengan keterangan:<br>
#     1 = Berhasil Bayar (Value: **Fully Paid**)<br>
#     0 = Gagal Bayar (Value: **Charged Off, Default dan Does not meet the credit policy. Status**)<br>
# 
# Untuk value seperti **Late, Current dan in_grace_period** tidak akan dimasukan karena status dari pinjaman tersebut masih berjalan.

# In[12]:


# Tentukan kategori yang akan dimasukan
good = ['Fully Paid']
bad = ['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Fully Paid']

# Ganti label pada loan_status
def loan_status_label(value):
    if value in good:
        return 1
    return 0


# In[13]:


# Menerapkan Fungsi
df_new = df[df['loan_status'].isin(good + bad)].copy()
df_new['loan_status'] = df['loan_status'].apply(loan_status_label)


# # Exploratory Data Analysis

# Melihat distribusi dari Target Feature yaitu Status Pinjaman

# In[14]:


plt.figure(figsize=(10,5))
sns.countplot(data=df_new, x='loan_status')
plt.show()


# ## Korelasi Antar Feature

# In[15]:


# Hitung korelasi tiap variabel
correlations = (df_new.select_dtypes(exclude=object)
                         .corr()
                         .dropna(how="all", axis=0)
                         .dropna(how="all", axis=1)
)


# In[16]:


correlations["loan_status"].abs().sort_values(ascending=False)


# In[17]:


# Saring korelasi antara vmin - vmax
vmin, vmax = 0.1, 0.99

unstack_corr = correlations.unstack()
pos_corr = (unstack_corr > vmin) & (unstack_corr < vmax)
neg_corr = (unstack_corr > -vmax) & (unstack_corr < -vmin)
high_corr = unstack_corr[pos_corr | neg_corr]

trimmed_corr = high_corr.sort_values(ascending=False).unstack()


# In[18]:


# Buat mask untuk membentuk matriks segitiga bawah
mask = np.zeros_like(trimmed_corr)
mask[np.triu_indices_from(mask)] = True


# In[19]:


# Tampilkan heatmap
plt.figure(figsize=(20, 20))
plot = sns.heatmap(
    trimmed_corr, 
    annot=True, 
    mask=mask,
    fmt=".2f", 
    cmap="viridis", 
    annot_kws={"size": 14})

plot.set_xticklabels(plot.get_xticklabels(), size=18)
plot.set_yticklabels(plot.get_yticklabels(), size=18)
plt.show()


# Dari heatmap diatas terdapat beberapa variabel yang memiliki pengaruh terhadap status pinjaman, diantaranya:

# In[20]:


affect_loan = high_corr.loc["loan_status"].abs().sort_values(ascending=False)
affect_loan


# Sedangkan fitur yang saling berkorelasi dengan yang sebelumnya perlu kita identifikasi. Kita menggunakan nilai batas 0.9 untuk mencari fitur yang saling berkorelasi kuat.

# In[21]:


threshold = 0.9
affect_collision = (high_corr.abs()
                             .loc[high_corr > threshold]
                             .loc[affect_loan.index, affect_loan.index]
                             .sort_values(ascending=False)
)
affect_collision


# Berdasarkan besar pengaruhnya terhadap status pinjaman, fitur yang saling berkorelasi akan dipilih berdasarkan yang paling berpengaruh.

# In[22]:


left_index = affect_collision.index.get_level_values(0)
right_index = affect_collision.index.get_level_values(1)

def remove_collide_index(left_index, right_index):
    include, exclude = [], []

    for left, right in zip(left_index, right_index):
        if left not in include and left not in exclude:
            include.append(left)
        if right not in include and right not in exclude:
            exclude.append(right)
        
    return include, exclude


include_affect_col, exclude_affect_col = remove_collide_index(left_index, right_index)
include_affect_col, exclude_affect_col


# Fitur numerik berpengaruh yang akan kita gunakan

# In[23]:


affect_num_cols = affect_loan[~affect_loan.index.isin(exclude_affect_col)].index.to_list()
affect_num_cols


# ## Loan Status dan Besar Pinjaman Pokok yang Sudah Dibayarkan

# Principal adalah besar pinjaman pokok yang dipinjamkan kepada debitur. Dengan kata lain merupakan jumlah asli dari uang yang dipinjamkan. Orang yang mengalami gagal bayar kebanyakan belum dapat membayarkan uang pokok pembayaran hingga jatuh tempo, bisa dilihat dari distribusi pembayaran dibawah. Rata-ratanya hampir mencapai 0.

# In[24]:


title_font = dict(size=20, weight="bold")

def plot_count(df, y, title, **sns_kwargs):
    value_counts = df[y].value_counts()
    percentage = value_counts / value_counts.sum()
    percentage = percentage.apply("{:.2%}".format)

    plt.figure(figsize=(14, 10))
    plt.title(title, fontdict=title_font)
    sns.countplot(data=df, y=y, order=value_counts.index, **sns_kwargs)
    plt.ylabel("")
    plt.show()

    print(percentage)


def plot_distribution(df, x, title, **sns_kwargs):
    plt.figure(figsize=(14, 10))
    plt.title(title, fontdict=title_font)
    sns.histplot(data=df, x=x, kde=True, **sns_kwargs)
    plt.ylabel("")
    plt.show()


def plot_boxplot(df, x, y, title, **sns_kwargs):
    plt.figure(figsize=(14, 10))
    plt.title(title, fontdict=title_font)
    sns.boxplot(data=df, x=x, y=y, **sns_kwargs)
    plt.ylabel("")
    plt.show()


# In[25]:


plot_distribution(df=df_new, x="total_rec_prncp", hue="loan_status", title="")


# ## Loan Status dan Total Uang yang Tidak Ditagihkan

# Charged off recoveries adalah total uang yang tidak bisa dibayarkan kepada perusahaan peminjam karena sudah lewat masa jatuh tempo sehingga perusahaan peminjam bisa melepas hak tagih utang tersebut dengan menjualnya ke perusahaan lain. Dari sini terlihat jelas bahwa orang dengan status pinjaman buruk lah yang paling banyak memiliki charge off recoveries.

# In[26]:


plot_distribution(df=df_new, x="recoveries", hue="loan_status", title="")


# # Loan Status dan Jumlah Pinjaman

# Rata-rata jumlah pinjaman terbanyak berada pada status buruk.

# In[27]:


x, y = "loan_status", "loan_amnt"
plot_boxplot(df=df_new, x=x, y=y, title="Distribusi Total Pinjaman")
df_new.groupby(x)[y].describe()


# ## Loan Status dan Total Pembayaran yang Diterima

# Tampak jelas apabila total pembayaran terbanyak berada pada pinjaman yang berstatus baik.

# In[28]:


x, y = "loan_status", "total_pymnt"
plot_boxplot(df=df_new, x=x, y=y, title="Distribusi Total Pembayaran yang Diterima")
df_new.groupby(x)[y].describe()


# ## Tujuan Mengambil Pinjaman

# Lebih dari setengah peminjam memiliki tujuan untuk menutup pinjaman sebelumnya. Apabila dilihat dari jenisnya, tujuan untuk konsumsi lebih banyak daripada tujuan untuk bisnis, renovasi dan pendidikan.

# In[29]:


plot_count(df_new, y="purpose", title="Tujuan Pinjaman")


# ## Negara Asal Peminjam

# Peminjam sebagian besar berasal dari negara Kanada.

# In[30]:


plot_count(df=df_new, y="addr_state", title="Negara Asal Peminjam")


# ## Tingkat Pinjaman

# Pinjaman diberi tingkatan dari huruf abjad A sampai G, semakin mendekati G maka tingkat bunga yang dibayarkan lebih besar.

# In[31]:


x, y = "int_rate", "grade"
order = df_new[y].sort_values().unique()
plot_boxplot(df_new, x=x, y=y, title="Tingkat Pinjaman", order=order)
plot_count(df=df_new, y=y, title="")
df_new.groupby(y)[x].describe()


# ## Status Kepemilikan Rumah

# Sebagian besar peminjam mendelegasikan rumahnya sebagai jaminan pinjaman, sedangkan hanya sedikit dari peminjam yang memiliki rumah sendiri.

# In[33]:


y = "home_ownership"
order = df_new[y].sort_values().unique()
plot_count(df=df_new, y=y, title="")


# # Data Preprocessing

# Setelah melihat info dan deskripsi dari data diatas, terdapat fitur yang tidak perlu kita pakai karena tidak begitu signifikan untuk digunakan sebagai fitur dalam prediksi.

# In[34]:


# Informasi rinci mengenai kolom dan baris data
data_stat = pd.DataFrame()
data_stat.index = df_new.columns
data_stat["unique_value"] = df_new.nunique()
data_stat["missing_rate"] = df_new.isna().mean()
data_stat["dtype"] = df_new.dtypes
data_stat


# Kolom dengan data yang tidak bisa dipakai

# In[35]:


# Kolom yang semua datanya hilang
miss_col = data_stat[data_stat["missing_rate"] == 1].index.to_list()
print("Kolom yang semua datanya hilang:")
print(miss_col)
print()

# Kolom yang terlalu unik
vari_col = data_stat[data_stat["unique_value"] == df_new.shape[0]].index.to_list()
print("Kolom yang terlalu unik:")
print(vari_col)
print()

# Kolom dengan kategori yang banyak
cat_col_stat = data_stat[data_stat["dtype"] == "object"]
vari_cat_col = cat_col_stat[cat_col_stat["unique_value"] > 1000].index.to_list()
print("Kolom dengan kategori yang banyak:")
print(vari_cat_col)
print()

# Kolom yang terdiri dari satu nilai
single_valued_col = data_stat[data_stat["unique_value"] == 1].index.to_list()
print("Kolom yang terlalu unik:")
print(single_valued_col)
print()

removed_features = miss_col + vari_col + vari_cat_col + single_valued_col


# In[36]:


# Hilangkan fitur yang tidak terpakai
pre_df = df_new.loc[:, ~df_new.columns.isin(removed_features)].copy()
pre_df.shape


# ## Categorical Feature

# In[37]:


# Kolom-kolom yang berdata kategorik
cat_features = pre_df.select_dtypes(include=object).columns
cat_features


# ## Kolom berisikan tanggal

# In[38]:


date_cols = ["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d", "next_pymnt_d"]

for col in date_cols:
    print(pre_df[col].value_counts().iloc[:5])
    print()


# Tidak terdapat korelasi yang kuat antar tanggal serta tiap tanggal memiliki sedikit korelasi dengan status pinjaman. Namun kita akan menghapus fitur tanggal yang berkorelasi kurang dari 0.1 dengan status pinjaman

# In[39]:


# Fitur tanggal yang akan kita gunakan
affect_date_cols = ["issue_d", "last_pymnt_d", "last_credit_pull_d", "next_pymnt_d"]
affect_date_cols


# In[40]:


# Hapus fitur tanggal yang tidak memiliki korelasi kuat dengan status pinjaman
unused_cols = ["earliest_cr_line"]
pre_df = pre_df.drop(columns=unused_cols, errors="ignore")
pre_df.head()


# ##  Categorical Column yang tidak dipokai

# In[41]:


other_cat_cols = cat_features[~cat_features.isin(date_cols)]
other_cat_cols


# In[42]:


pre_df.loc[:, other_cat_cols].head()


# Beberapa kolom kategorikal yang tidak terpakai adalah:<br>
# 
# - desc dan title karena merupakan teks.
# - zip_code karena 3 angka dibelakangnya disensor
# - sub_grade karena sudah memiliki kolom yang mirip yaitu grade

# In[43]:


unused_cols = ["desc", "zip_code", "sub_grade", "title"]
pre_df = pre_df.drop(columns=unused_cols, errors="ignore")
pre_df.head()


# In[44]:


other_cat_cols = cat_features[~cat_features.isin(date_cols + unused_cols)]
other_cat_cols


# Terdapat korelasi yang kuat antara emp_title dengan status pinjaman, disusul dengan grade dan term. Fitur yang kurang berpengaruh lainnya tidak akan digunakan untuk prediksi.

# In[45]:


# Fitur kategorikal yang akan kita gunakan
affect_cat_cols = ["grade", "term"]
affect_cat_cols


# In[46]:


# Hapus fitur yang kurang berpengaruh
used_cols = ["emp_title", "grade", "term"]
unused_cols = other_cat_cols[~other_cat_cols.isin(used_cols)]
pre_df = pre_df.drop(columns=unused_cols, errors="ignore")
pre_df.head()


# ## Feature yang berkorelasi dengan target

# In[47]:


# Kolom-kolom yang akan kita gunakan
predictor_cols = affect_num_cols + affect_cat_cols + affect_date_cols
predictor_cols


# # Feature Engineering

# ## Imputasi pada kolom missing value

# Missing value terbanyak dimiliki oleh fitur next_pyment_d karena bisa jadi peminjam yang sudah melunasi utangnya tidak akan memiliki jadwal pembayaran lagi.

# In[48]:


pre_df[predictor_cols].isna().mean().sort_values(ascending=False)


# In[49]:


# Isi data dengan "no"
pre_df["next_pymnt_d"] = pre_df["next_pymnt_d"].fillna("no")
top_next_pyment_d = pre_df["next_pymnt_d"].value_counts().head()
top_next_pyment_d


# Lakukan hal yang sama pada kolom last_pymnt_d dan last_credit_pull_d

# In[50]:


pre_df["last_pymnt_d"] = pre_df["last_pymnt_d"].fillna("no")
pre_df["last_credit_pull_d"] = pre_df["last_credit_pull_d"].fillna("no")


# Isi missing value data numerik menggunakan nilai modus

# In[51]:


mode = pre_df["inq_last_6mths"].mode().values[0]
pre_df["inq_last_6mths"] = pre_df["inq_last_6mths"].fillna(mode)


# Cek kembali apakah masih ada data yang hilang

# In[52]:


pre_df[predictor_cols].isna().mean().sort_values(ascending=False)


# # Modeling

# ## Menentukan Target dan Feature data

# Label merupakan tingkat performa dari pinjaman yang berada pada kolom loan_status. Berhubung kolom tersebut memiliki beberapa kategori, kita sudah memilih dan menggabungkannya menjadi 2 kategori yaitu baik dan buruk.<br>
# 
# Sebelumnya, kita perlu memisahkan label dan fitur dari data untuk kemudian dapat dilakukan pemisahan data.

# In[53]:


label = pre_df["loan_status"].copy()
features = pre_df[predictor_cols].copy()

print("Label shape:")
print(label.shape)

print("Features shape:")
print(features.shape)


# In[54]:


num_features = features.select_dtypes(exclude="object")
cat_features = features.select_dtypes(include="object")


# In[55]:


# Normalisasi fitur numerik
num_features = (num_features - num_features.mean()) / num_features.std()
num_features


# In[56]:


# OneHotEncode fitur kategorik
cat_features = pd.get_dummies(cat_features)
cat_features


# In[57]:


# Gabungkan Fitur
features_full = pd.concat([num_features, cat_features], axis=1)


# features_full.shape

# ## Train Test Split

# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_full, label, test_size=0.2, random_state=42, stratify=label)


# In[59]:


X_train.shape, y_train.shape


# ## Fitting

# In[58]:


from sklearn.linear_model import LogisticRegression
logres = LogisticRegression(max_iter=500, solver="sag", class_weight="balanced", n_jobs=-1)
logres


# In[59]:


logres.fit(X_train, y_train)


# ##  Save Model

# In[61]:


joblib.dump(logres, "logres.z")


# In[62]:


logres = joblib.load("logres.z")


# # Model Evaluation

# Kita akan membuat model prediksi paling sederhana yaitu dengan memprediksi seluruh data kategori terbanyak. Hal ini dilakukan supaya kita mendapatkan patokan, berapa performa minimal yang harus dilalui oleh model machine learning kita nantinya.

# In[63]:


test_label_counts = y_test.value_counts()
test_label_counts


# In[64]:


test_label_counts.max() / test_label_counts.sum()


# ## Classification Metrics

# ### Training

# In[65]:


logres.score(X_train, y_train)


# In[66]:


report = classification_report(y_true=y_train, y_pred=logres.predict(X_train))
print(report)


# ### Testing

# In[67]:


logres.score(X_test, y_test)


# In[68]:


report = classification_report(y_true=y_test, y_pred=logres.predict(X_test))
print(report)


# ## Confusion Matrix

# In[69]:


conf = confusion_matrix(y_true=y_test, y_pred=logres.predict(X_test))


# In[70]:


plt.figure(figsize=(10, 10))
sns.heatmap(conf, annot=True, fmt="g")
plt.show()

