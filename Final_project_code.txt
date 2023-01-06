import pandas as pd
import numpy as np
# xgboost
from xgboost import XGBClassifier
# PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
# 繪圖
import matplotlib.pyplot as plt
import seaborn as sns
from operator import *
import talib

# 選擇幣種
df = pd.read_csv('Bitcoin.csv', encoding='utf-8', low_memory=False)

df.set_index('Date', inplace=True)
df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'CryptoFGI']]

# Labels
def triple_barrier(data, price, ub, lb, max_period):
    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0] / s[0]
    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period + 1)
    signal = pd.Series(1, p.index)
    # 漲過4%
    signal.loc[p > ub] = 2
    # 跌下2%
    signal.loc[p < lb] = 0
    # ret = pd.DataFrame({'triple_barrier': signal})
    data['triple_barrier'] = signal
    return data
def RSI(data, Close ,period):
    # 整理資料
    Chg = Close - Close.shift(1)
    Chg_pos = pd.Series(index=Chg.index, data=Chg[Chg > 0])
    Chg_pos = Chg_pos.fillna(0)
    Chg_neg = pd.Series(index=Chg.index, data=-Chg[Chg < 0])
    Chg_neg = Chg_neg.fillna(0)
    # 計算14日平均漲跌幅度
    up_mean = []
    down_mean = []
    for i in range(period + 1, len(Chg_pos) + 1):
        up_mean.append(np.mean(Chg_pos.values[i - period:i]))
        down_mean.append(np.mean(Chg_neg.values[i - period:i]))
    # 計算 RSI
    rsi = []
    for i in range(len(up_mean)):
        rsi.append(100 * up_mean[i] / (up_mean[i] + down_mean[i]))
    rsi_series = pd.Series(index=Close.index[period:], data=rsi)
    data['RSI'] = rsi_series
    return data
def on_balance_volume(data, periods, close_col='Close', vol_col='Volume'):
    OBV = []
    OBV.append(0)
    for i in range(1, len(close_col)):
        if close_col[i] > close_col[i - 1]:  # If the closing price is above the prior close price
            OBV.append(OBV[-1] + vol_col[i])  # then: Current OBV = Previous OBV + Current Volume
        elif close_col[i] < close_col[i - 1]:
            OBV.append(OBV[-1] - vol_col[i])
        else:
            OBV.append(OBV[-1])
    data['obv'] = OBV
    data['OBV_EMA'] = data['obv'].ewm(com=20).mean()

    return data
def on_balance_volume_FGI(data, periods, close_col='FGI', vol_col='Volume'):
    OBV = []
    OBV.append(0)
    for i in range(1, len(close_col)):
        if close_col[i] > close_col[i - 1]:  # If the closing price is above the prior close price
            OBV.append(OBV[-1] + vol_col[i])  # then: Current OBV = Previous OBV + Current Volume
        elif close_col[i] < close_col[i - 1]:
            OBV.append(OBV[-1] - vol_col[i])
        else:
            OBV.append(OBV[-1])
    data['123'] = OBV
    data['OBV_FGI'] = data['123'].ewm(com=20).mean()

    return data


def ema(data, period, column='Close'):
    data['ema_' + str(period)] = column.ewm(ignore_na=False, min_periods=period,
                                            com=period, adjust=True).mean()
    return data
def macd(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    remove_cols = []
    if not 'ema' + str(period_long) in data.columns:
        data = ema(data, period_long)
        remove_cols.append('ema' + str(period_long))

    if not 'ema' + str(period_short) in data.columns:
        data = ema(data, period_short)
        remove_cols.append('ema' + str(period_short))

    data['macd_val'] = data['ema' + str(period_short)] - data['ema' + str(period_long)]
    data['macd_signal_line'] = data['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal,
                                                    adjust=True).mean()

    data = data.drop(remove_cols, axis=1)

    return data
def FGI_bubble_14(data, period):
    p = data['CryptoFGI'].rolling(window=period, center=False).mean()
    data['FGI_' + str(period)] = p - data['CryptoFGI']

    return data
def FGI_class(data, FGI, ub, lb):
    p = FGI
    signal = pd.Series(1, p.index)
    # 漲過4%
    signal.loc[p > ub] = 2
    # 跌下2%
    signal.loc[p < lb] = 0
    data['FGI_class'] = signal
    return data

triple_barrier(df, df.Close, 1.08, 0.96, 14)
RSI(df, df.Close, 14)
on_balance_volume(df, 9, df.Close, df.Volume)
ema(df, 9, df.Close)
# Fear & Greed Index 相關
on_balance_volume_FGI(df, 9, df.CryptoFGI, df.Volume)
FGI_bubble_14(df, 14)
FGI_class(df, df.CryptoFGI, 80, 20)
df['RSI_FGI'] = talib.RSI(df.CryptoFGI)
# macd(df, 26, 12, 9, df.Close)

# Momentum Indicator Functions
df['ADX_14'] = talib.ADX(df.High, df.Low, df.Close, timeperiod=14)
df['ADXR_14'] = talib.ADXR(df.High, df.Low, df.Close, timeperiod=14)
df['APO'] = talib. APO(df.Close, fastperiod=12, slowperiod=26, matype=0)
# df['AROONOSC'] = talib.AROONOSC(df.High, df.Low, timeperiod=14)
# df['BOP'] = talib.BOP(df.Open, df.High, df.Low, df.Close)
# df['CCI'] = talib.CCI(df.High, df.Low, df.Close, timeperiod=14)
# df['CMO'] = talib.CMO(df.Close, timeperiod=14)
# df['DX'] = talib.DX(df.High, df.Low, df.Close, timeperiod=14)
# df['MFI'] = talib.MFI(df.High, df.Low, df.Close, df.Volume, timeperiod=14)
# df['MINUS_DI'] = talib.MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
# df['MINUS_DM'] = talib.MINUS_DM(df.High, df.Low, timeperiod=14)
# df['MOM'] = talib.MOM(df.Close, timeperiod=10)
# df['PLUS_DI'] = talib.PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
# df['PLUS_DM'] = talib.PLUS_DM(df.High, df.Low, timeperiod=14)
# df['PPO'] = talib.PPO(df.Close, fastperiod=12, slowperiod=26, matype=0)
# df['ROC'] = talib.ROC(df.Close, timeperiod=10)
# df['ROCP'] = talib.ROCP(df.Close, timeperiod=10)
# df['ROCR'] = talib.ROCR(df.Close, timeperiod=10)
# df['ROCR100'] = talib.ROCR100(df.Close, timeperiod=10)
# df['TRIX'] = talib.TRIX(df.Close, timeperiod=30)
# df['ULTOSC'] = talib.ULTOSC(df.High, df.Low, df.Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
# df['WILLR'] = talib.WILLR(df.High, df.Low, df.Close, timeperiod=14)

# 波動率指標
df['ATR_14'] = talib.ATR(df.High, df.Low, df.Close, timeperiod=14)
df['NATR_14'] = talib.NATR(df.High, df.Low, df.Close, timeperiod=14)
df['TRANGE'] = talib.TRANGE(df.High, df.Low, df.Close)

# # MA線
# MA5  = df['Close'].rolling(window=5, center=False).mean()
# MA10 = df['Close'].rolling(window=10, center=False).mean()
# MA20 = df['Close'].rolling(window=20, center=False).mean()
# MA60 = df['Close'].rolling(window=60, center=False).mean()
# df['Bios_MA5'] = (df.Close - MA5) / MA5
# df['Bios_MA10'] = (df.Close - MA10) / MA10
# df['Bios_MA20'] = (df.Close - MA20) / MA20
# df['Bios_MA60'] = (df.Close - MA60) / MA60
# df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

df['Bios_EMA9'] = (df.Close - df.ema_9) / df.ema_9
EMA_12 = df['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_DIF'] = EMA_12 - EMA_26
df['MACD_signal'] = df.MACD_DIF.ewm(span=9, adjust=False).mean()
df['MACD_histogram'] = df.MACD_DIF - df.MACD_signal

# 資料處理前面NAN列全刪，後14列全刪(因triple_barrier是預測未來14日)
df.drop(df.tail(14).index, inplace=True)
# 後方輸入要刪除的比例，剩下的為data
del_size = 14 + round((len(df) - 14) * 0.4)
df.drop(df.head(del_size).index, inplace=True)
all_data = df.dropna()
all_data.to_csv('all_data.csv')
features = all_data.drop(columns=['High', 'Low', 'Close', 'Open', 'Volume',
                                  'triple_barrier', 'ema_9', 'obv'])
features.to_csv('features.csv')

# 取出 Train & Test_data 、 Labels 、 Close
all_target = all_data.triple_barrier
Close_data = all_data.Close
Open_data = all_data.Open
triple_barrier_data = all_data.triple_barrier

# proportion為train的比例
# 分4季，proportion改成[0.6、0.7、0.8、0.9]
proportion = 0.6
train_size = round(len(all_data) * proportion)
test_tail = round(len(all_data) * (proportion + 0.1))

train_data = features[:train_size]
test_data = features[train_size: test_tail:]
train_target = all_target[:train_size]
test_target = all_target[train_size: test_tail:]

plt_train_label = all_data[:train_size]
plt_test_label = all_data[train_size: test_tail:]
### class count ###
total = len(plt_train_label)

plt.figure(figsize=(10, 7))
plt.subplot(121)
g = sns.countplot(x='triple_barrier', data=plt_train_label)
g.set_title("class Count", fontsize=14)
g.set_ylabel('Count', fontsize=14)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 1.5,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14, fontweight='bold')
plt.margins(y=0.1)
plt.show()

# # # 股價圖  # # # ok

## Train & Test data
fig = plt.figure(figsize=(18, 6))

Close_train = Close_data[0:train_size]
Close_Test = Close_data[train_size:test_tail:]
Close_Backtest = Close_data[train_size: test_tail]
Open_Test = Open_data[train_size: test_tail]
triple_barrier_Test = triple_barrier_data[train_size:test_tail]

ax = Close_data.plot()
plt.plot([None for i in Close_train] + [x for x in Close_Test])
ax.legend(['train', 'test'])
plt.title('Train & Test data')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid()
plt.show()

## train (Fear & Greed)
# fig = plt.figure(figsize=(18, 6))
# ax1 = fig.add_subplot(111)
# plt_train_label['Close'].plot(ax=ax1, label='BTC Price')
# plt.title('Bitcoin(Fear & Greed)')
# plt.xlabel('Date')
# ax1.set_ylabel('Price')
# plt.legend(loc=2)
# #
# ax2 = ax1.twinx()
# plt_train_label['CryptoFGI'].plot(ax=ax2, alpha=0.5, color='peru', label='Fear & Greed')
# ax2.set_yticks(np.arange(0, 100, 10))    # 设置右边纵坐标刻度
# ax2.set_ylabel('Fear & Greed')       # 设置右边纵坐标标签
# plt.legend(loc=1)
# ax1.grid()
# plt.show()

## train (label)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(111)
plt_train_label['Close'].plot(ax=ax1, label='Price')
plt.title('train data(labeling)')
plt.xlabel('Date')
ax1.set_ylabel('Price')
plt.legend(loc=2)

ax2 = ax1.twinx()
plt_train_label['triple_barrier'].plot(ax=ax2, alpha=0.3, color='r', label='label')
ax2.set_yticks(np.arange(0, 3, 1))    # 设置右边纵坐标刻度
ax2.set_ylabel('label')       # 设置右边纵坐标标签
plt.legend(loc=1)
ax1.grid()
plt.show()

## test (label)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(111)
plt_test_label['Close'].plot(ax=ax1, label='Price')
plt.title('test data(labeling)')
plt.xlabel('Date')
ax1.set_ylabel('Price')
plt.legend(loc=2)

ax2 = ax1.twinx()
plt_test_label['triple_barrier'].plot(ax=ax2, alpha=0.3, color='r', label='label')
ax2.set_yticks(np.arange(0, 3, 1))    # 设置右边纵坐标刻度
ax2.set_ylabel('label')       # 设置右边纵坐标标签
plt.legend(loc=1)
ax1.grid()
plt.show()

##############################

# XGBoost 調參 ok
''''''
# parameter
xgb_parameter = {'use_label_encoder': [False],
                 'eval_metric': ['mlogloss'],
                 'n_estimators': np.linspace(100, 1000, 10, dtype=int),
                 'max_depth': [1, 2],
                 'learning_rate': np.linspace(0.01, 0.2, 20, dtype=float),
                 'min_child_weight': [1, 2, 3, 4, 5]}
### Bitcoin ###
# 以下為每季的最終選擇參數
# Q1
# xgb_parameter = {'use_label_encoder': [False],
#                  'eval_metric': ['mlogloss'],
#                  'n_estimators': [900],
#                  'max_depth': [1],
#                  'learning_rate': [0.01],
#                  'min_child_weight': [1]}

# Q2
# xgb_parameter = {'use_label_encoder': [False],
#                  'eval_metric': ['mlogloss'],
#                  'n_estimators': [50],
#                  'max_depth': [2],
#                  'learning_rate': [0.01],
#                  'min_child_weight': [1]}

# Q3
# xgb_parameter = {'use_label_encoder': [False],
#                  'eval_metric': ['mlogloss'],
#                  'n_estimators': [600],
#                  'max_depth': [2],
#                  'learning_rate': [0.02],
#                  'min_child_weight': [1]}

# Q4
# xgb_parameter = {'use_label_encoder': [False],
#                  'eval_metric': ['mlogloss'],
#                  'n_estimators': [600],
#                  'max_depth': [2],
#                  'learning_rate': [0.01],
#                  'min_child_weight': [1]}

# grid_search

xgb = XGBClassifier()
# Instantiate the grid search model
my_cv = TimeSeriesSplit(n_splits=5).split(train_data)
grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_parameter,
                           scoring='accuracy', cv=my_cv, n_jobs=-1, verbose=2)
# grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_parameter,
#                            scoring='accuracy')
grid_search.fit(train_data, train_target)
# total time
mean_fit_time = grid_search.cv_results_['mean_fit_time']
mean_score_time = grid_search.cv_results_['mean_score_time']
n_splits = grid_search.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(grid_search.cv_results_).shape[0] #Iterations per split
print('grid_search total time :')
print(np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter)
# best parameter
print('best parameter :')
print(grid_search.best_params_)
# the best training score
print('the best training score :')
print(grid_search.score(train_data, train_target))
# testing score of best estimator
print('testing score of best estimator :')
print(grid_search.score(test_data, test_target))

test_predicted = grid_search.predict(test_data)
pred = pd.DataFrame(test_predicted, columns=['output'])
pred.to_csv('predicted.csv')

# #　# 模型診斷 # # #

## test (pred)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(111)
plt_test_label['Close'].plot(ax=ax1, label='Price')
plt.title('test data(pred)')
plt.xlabel('Date')
ax1.set_ylabel('Price')
plt.legend(loc=2)

ax2 = ax1.twinx()
pred['output'].plot(ax=ax2, alpha=0.3, color='r', label='label')
ax2.set_yticks(np.arange(0, 3, 1))    # 设置右边纵坐标刻度
ax2.set_ylabel('label')       # 设置右边纵坐标标签
plt.legend(loc=1)
ax1.grid()
plt.show()

## confusion_matrix     ok
cf_matrix = confusion_matrix(test_target, test_predicted)
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(3, 3)

plt.figure(figsize=(10, 7))
cmax = sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='')

cmax.set_title('Confusion Matrix\n')
cmax.set_xlabel('Model Predicted')
cmax.set_ylabel('Actual')

cmax.xaxis.set_ticklabels(['0(Sell)', '1(Hold)', '2(Buy)'])
cmax.yaxis.set_ticklabels(['0', '1', '2'])
plt.show()

## precision & recall    ok
tp = np.diag(cf_matrix)
prec = list(map(truediv, tp, np.sum(cf_matrix, axis=0)))
rec = list(map(truediv, tp, np.sum(cf_matrix, axis=1)))
# print('Precision: {}\nRecall: {}'.format(prec, rec))
print('sell_precision : ', prec[0])
print('hold_precision  : ', prec[1])
print('buy_precision  : ', prec[2])
print('sell_recall : ', rec[0])
print('hold_recall : ', rec[1])
print('buy_recall  : ', rec[2])


#############################
# Backtesting
## 起始: 自由金、比特畢 各半 ##
# labeling的信號做交易
def label_backtest(price_df, total):
    asset_values = []
    cash = total
    shares = round((cash / price_df.Open[0]) / 2)
    cash -= shares * price_df.Open[0]
    for i in range(len(price_df) - 1):
        close_t = price_df['Close'][i]
        close_tp1 = price_df['Close'][i + 1]
        signal = price_df['triple_barrier'][i]
        # daily_change = round((close_t / open_t), 2)
        # 買入賣出的現金流
        buy_cashflow = close_t * 1.002
        sell_cashflow = close_t * 0.998

        # 觸及布林通道上緣
        if (signal == 2) and (cash >= buy_cashflow):
            shares += 1
            cash -= buy_cashflow
        # 觸及布林通道下緣
        if (signal == 0) and (shares >= 1):
            shares -= 1
            cash += sell_cashflow
        # 記錄每日資產變化
        total_asset = shares * close_tp1 + cash
        asset_values.append(total_asset)

    total_return = round((total_asset / 1000000 - 1) * 100, 2)
    print(f'Label Return : {total_return}%')
    return asset_values
# XGBoost(買一支賣一支)
def Buy_one_Sell_one_backtest(price_df, total):
    asset_values = []
    cash = total
    shares = round((cash / price_df.Open[0]) / 2)
    cash -= shares * price_df.Open[0]
    for i in range(len(price_df) - 1):
        # open_t = price_df['Open'][i]
        close_t = price_df['Close'][i]
        close_tp1 = price_df['Close'][i + 1]
        signal = price_df['buy_sell'][i]
        # daily_change = round((close_t / open_t), 2)
        # 買入賣出的現金流
        buy_cashflow = close_t * 1.002
        sell_cashflow = close_t * 0.998

        # 觸及布林通道上緣
        if (signal == 2) and (cash >= buy_cashflow):
            shares += 1
            cash -= buy_cashflow
        # 觸及布林通道下緣
        if (signal == 0) and (shares >= 1):
            shares -= 1
            cash += sell_cashflow
        # 記錄每日資產變化
        total_asset = shares * close_tp1 + cash
        asset_values.append(total_asset)
        # BAHcaash *= daily_change
        # BAH_asset_values.append(BAHcaash)
    total_return = round((total_asset / 1000000 - 1) * 100, 2)
    print(f'Buy_1 Sell_1 Return : {total_return}%')
    return asset_values
# 全買全賣
def All_in_all_out(price_df, total):
    asset_values = []
    cash = total / 2
    inmarket = cash
    for i in range(len(price_df) - 1):
        open_t = price_df['Open'][i]
        close_t = price_df['Close'][i]
        signal = price_df['buy_sell'][i]
        daily_change = round((close_t / open_t), 2)
        # 買入賣出的現金流
        buy_cashflow = cash
        sell_cashflow = inmarket
        inmarket = inmarket * daily_change
        # 買
        if (signal == 2) and (cash >= buy_cashflow):
            inmarket += buy_cashflow
            cash -= buy_cashflow
        # 賣
        if (signal == 0) and (inmarket >= sell_cashflow):
            inmarket -= sell_cashflow
            cash += sell_cashflow
        # 記錄每日資產變化
        total_asset = inmarket + cash
        asset_values.append(total_asset)
    allin_return = round((total_asset / total - 1) * 100, 2)
    print(f'All in all out Return : {allin_return}%')
    return asset_values

# 百分比交易方式
def percent_deal(price_df, buysell_percent, total):
    asset_values = []
    cash = total / 2
    inmarket = cash
    for i in range(len(price_df) - 1):
        open_t = price_df['Open'][i]
        close_t = price_df['Close'][i]
        signal = price_df['buy_sell'][i]
        daily_change = round((close_t / open_t), 2)
        # 買入賣出的現金流
        buy_cashflow = cash * buysell_percent * 1.002
        sell_cashflow = inmarket * buysell_percent * 0.998
        inmarket = inmarket * daily_change
        # 買
        if (signal == 2) and (cash >= buy_cashflow):
            inmarket += buy_cashflow
            cash -= buy_cashflow
        # 賣
        if (signal == 0) and (inmarket >= sell_cashflow):
            inmarket -= sell_cashflow
            cash += sell_cashflow
        # 記錄每日資產變化
        total_asset = inmarket + cash
        asset_values.append(total_asset)
    percent_return = round((total_asset / total - 1) * 100, 2)
    print(f'percent Return : {percent_return}%')
    return asset_values

# Buy & Hold
def BAH(price_df, BAHcash):
    BAH_asset_values = []
    for i in range(len(price_df) - 1):
        open_t = price_df['Open'][i]
        close_t = price_df['Close'][i]
        daily_change = round((close_t / open_t), 2)
        # 記錄每日資產變化
        BAHcash *= daily_change
        BAH_asset_values.append(BAHcash)
    BAH_return = round((BAHcash / 1000000 - 1) * 100, 2)
    print(f'Buy & Hold Return : {BAH_return}%')
    return BAH_asset_values


price_df = pd.DataFrame(Open_Test, columns=['Open'])
price_df['Close'] = Close_Backtest
price_df['triple_barrier'] = triple_barrier_Test
price_df['buy_sell'] = test_predicted
price_df.to_csv('predicted & close.csv')

# 起始100W美金
All_Money = 1000000
Allin_asset_values = All_in_all_out(price_df, All_Money)
percent_asset_values = percent_deal(price_df, 0.1, All_Money)
BAH_asset_values = BAH(price_df, All_Money)

price_df.drop(price_df.head(1).index, inplace=True)
price_df['Allin_asset'] = Allin_asset_values
price_df['percent_asset'] = percent_asset_values
price_df['BAH_asset'] = BAH_asset_values
price_df.to_csv('backtest.csv')

plt.figure(figsize=(18, 6))
plt.title('Backtest')
plt.xlabel('Date')
plt.ylabel('Price')
# price_df.label_asset.plot(label='Label_backtest')
# price_df.B1S1_asset.plot(label='Buy one Sell one')
price_df.Allin_asset.plot(label='All in all out')
price_df.percent_asset.plot(label='percentage deal')
price_df.BAH_asset.plot(label='Buy & Hold')
plt.legend(loc=1)
plt.grid()
plt.show()
