## StockPool: generate feature data to RL

- Editor: youngmi huang 黄少瑄
- Update date: 2019/4/28
- version: 2.0.0


## Usage
### 数据更新与模型更新

1. 首次更新：设定选股日的起始年月 `batch_year`、`batch_month` 后，选股模型会跑到当前年月，形成股票池

   (`topN`  : 设定股票池选择的个数)

2. 当月更新：执行 `main.py` 当中的 `main()` 即可更新每日数据，当日若为每月最后一个星期五（选股日）则会跑选股模型以及更新产出 `codelist_info.csv`

   

```python
# 首次更新：从设定要跑的起始年月到当前年月 (首度生成 codelist_info.csv)
main(topN=7, batch_mode=True) # 初始设定 batch_year=2015, batch_month=12

# 当月更新：
main(topN=7)

```

PS. 首次更新：歷史回測的起始點，( year, month ) 設定須大於 (2015, 12），否則需更新 hs300 歷史成分股 `hist_hs300.csv` (目前最早的日期為 2015/6/30) 

## 

### 所花时间

2. 每日更新（行情、ola）：

    `daily_data.py` 爬全市场每日行情数据以及回写回 `Basic_Data_a` 约需 90 min。由于 tushare 有每分钟 200 次限制，设置了 time.sleep (0.5) 。

   

2. 每月更新（选股）

   (1) 历史更新

   使用 lightgbm 模型，更新 `2015/12 - 2019/3` 約花 90 min。

   (2) 当月更新：

   會依序更新季报数据 (quarterly_data.py) 、 创建新的特徵 (create_feature.py)、跑当月选股模型(run_model.py)

   | 所花时间 | 依序更新                                             |
   | -------- | ---------------------------------------------------- |
   | 7-8 min  | 1. 季报数据 (quarterly_data.py)                      |
   | 3-4 min  | 2. 创建新的特徵 (create_feature.py)                  |
   | 7 min    | 3. 跑当月选股模型与产出目标股票池文件 (run_model.py) |



## Structure

```python
├─ src      
│    ├──config.yml                     # 参数设定
│    ├──main.py                        # 主函数（数据更新、选股模型更新)
│    ├──daily_data.py                  # 每日市场、行情数据更新 (API获取)
│    ├──ola_connect.py                 # 每日更新 ola 舆情数据 (API获取)
│    ├──deliver_data.py                # 计算并输出 RL 训练所需特徵
│    ├──quarterly_data.py              # 每月更新最新的财报数据(虽是季报但因公司发布日不一)
│    ├──create_feature.py              # 选股模型的特徵构建
│    ├──run_model.py                   # 产生选股模型的每月上涨概率、更新后的股票池
│    ├──change_data.py                 # 根据当前股票池与每月选股模型产生的上涨概率，建立换股逻辑
│    └──tools       
│         ├──  utils.py                # 选股模型会用到的子函数 
│         └──  prepro.py               # 特徵预处理使用
│        
├─ a_return.csv                        # 沪深300市场行情 
├─ codelist_info.csv                   # 目标股票池的文件路径 
├─ sample_data_a                       # 输出 RL 训练所需特徵文件位置 
│    ├──  000001.SZ.csv       
│    ├──  000002.SZ.csv
│    ├──  000008.SZ.csv 
│    ├──  ...
│    
├─ csv 
│    ├──  CODES.txt                    # 全市場股票代碼文件 
│    ├──  fullname.csv                 # 全市场股票代码与企业中文名称匹配文件 
│    ├──  industry.csvt                # 全市场股票所属产业匹配文件
│    ├──  Basic_Data_a                 # 每日行情数据更新存放位置(历史+最新)
│            ├──  000001.SZ.csv       
│            ├──  000002.SZ.csv
│            ├──  .. 
│
│    ├──  New_Data_a                   # 每日行情数据更新存放位置(最新)
│            ├──  000001.SZ.csv       
│            ├──  000002.SZ.csv
│            ├──  .. 
│
│    ├──  mood_data                    # ola 舆情数据存放位置(历史＋最新) 
│            ├──  mood.csv
│            ├──  mood_hist.csv
│            └──  mood_daily           # ola 舆情数据更新(最新)
│                     ├──  000001.SZ.csv       
│                     ├──  000002.SZ.csv
│                     ├──  .. 
│
│    ├──  index_data                   # 计算 RL 训练所需全市场特徵存放位置
│            ├──  amount.csv     
│            ├──  volatility_5.csv
│            ├──  volatility_22.csv
│            └──  mkt_volatility_change.csv
│
│    ├──  model_data                   # 选股模型所需使用文件存放位置
│            ├──  hist_hs300.csv       # hs300 历史成分股 (自2015以来)        
│            ├──  hist_sh50.csv        # sh50  历史成分股 (自2015以来)        
│            ├──  hs300_average.csv    # 沪深300平权市场行情 (for选股模型评估)
│            ├──  sh50.csv             # 上证50市场行情 (for选股模型评估)
│            ├──  feature_total.pkl    # 选股模型构建的特徵文件存放位置 
│            ├──  dm_basic_5.csv       # 计算季度特徵的财报数据
│            ├──  performance_hist.csv # 选股模型上涨机率与模型成效表
│            └──  codelist_info_hist   # 备份目标股票池

```



## Additional

1. raw data 不包含停牌交易日期，所有数据处理补值判断皆在`deliver_output.py` 做

2. get_codelist_info()：拿上一期的股票池当做起始点， `performance_hist.csv` 为本期预测下个月各支股票上涨概率，换掉上一期股票池当中**本期预测上涨概率最小**的一支，并以不在上一期股票池当中**上涨概率最大**的一支做替换。
3. checkpoint點 
   - 一致性：交易日期需與大盤ㄧ致，每一支股票的數據量要相同 (X, 新股 退市股 不在此範圍限制) 
   - 檢查產出欄位有無缺失值/ 產出欄位順序需恆定 (欄位名稱已由 std 改為 stk) 
   - 缺失值發生情況：
     - 檢查除了date以外的欄位，各欄位是否有空值 
     - 判斷停牌的股票 => 若為空值補值要先補 close (補前面有值得部分） => 再由close由右往左補值 
     - ind_volatility：該交易日行業內包含的企業若都為空值 則要補 0 
     - daily 数据更新 dataflow：

![image-20190429005945425](/Users/youngmihuang/Library/Application Support/typora-user-images/image-20190429005945425.png) 

4. 每月選股日期為當月最後一個星期五

```python
# 获取(year, month) 最后一天
utils.getMonthFirstDayAndLastDay
# 获取(year, month) 最后一个星期五
utils.get_last_byday
# 获取选股日期 (当月为返回一笔, 回测返回多笔)
get_select_dt()
# 依照选股日期(最后一个星期五) 获得周一为 start_dt，下一个月的最后一个星期五为 end_dt
get_start_end_dt()

```





## Changelog

4/28

代碼重構；功能确立；股票池支援 hs300 与 sh50 两种股票池风格选股；更改为一次选7支

4/1

更新 `deliver_output.py` 中产生输出股票池的 get_code_list() 为历史股票池所有包含的股，以提供强化学习做历史建模与回测；更新 `ola_connect.py `中，讀取內容編碼 decode為 'utf=8' 以防止跨系統執行時報錯

3/31

-更新模型调用方式、产生每月更新股票池选股结果 (從2015/12 ~ 2019/3)、更新日为每月最后一个星期五(包含历史回测逻辑)

3/28

-初版

-----

##### To do

- 模型参数存放改寫 (lgb, xgboost. logistic regreesion)
- 2019/6, 2019/12 之後每半年的 hs300 組成替換 (目前使用 wind 手動下載與 append，須解決wind端 access token 問題 or 改找看 tushare pro 有無可用數據)
- 2018年以後新增的股票數據 (約70支)



##### Done

- 路径改为以 config 档设定为主
- model 月更新的部份正在重构中 => 已完成，可支援月更新與歷史更新 (2019/3/31)

- 嵌入quarterly data, features 更新流程
- 此文件 structure / Additional 待更新