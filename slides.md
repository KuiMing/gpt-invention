# Gpt Invention

Start from here!

---

<!-- .slide: data-background="#999999" -->

<!-- .slide: data-background-iframe="media/flow.html" -->

---

## Data

---

### Annual Report (10-k filings)

- 原作：
    - [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/#Annual-Reports-on-Form-10-K)
    - 費用：29 美元/月
- 建議：
    - 使用爬蟲 [sec-edgar](https://github.com/sec-edgar/sec-edgar) 在 [SEC 美國證券交易委員會](https://www.sec.gov/edgar/searchedgar/companysearch) 抓取 10-k filings
    - 免費
    - 免費

----

### Stock Price
- total 1505 stocks
    - S&P500, S&P400, S&P600
- 原作：
    - [`openbb`](https://openbb.co/)
- 建議：
    - [`yfinance`](https://finance.yahoo.com/)：Yahoo! Finance

----

<!-- .slide: data-background="https://hackmd.io/_uploads/BktMtfN1a.png" -->


---

### Sample

- train: 1k datapoints (out of 17.4k)
- test: 500 datapoints (out of 6.8k) 



## Cost: 

- Money: $60
- Time: 50 hours


---

## Embedding Model


### all-mpnet-base-v2
This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.
https://huggingface.co/sentence-transformers/all-mpnet-base-v2


```python!
embedding_model = LangchainEmbedding(   HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    ) 
```

---

### Massive Text Embedding Benchmark
https://huggingface.co/blog/mteb
![](https://hackmd.io/_uploads/Bkq8xozJT.png)

---

![](https://hackmd.io/_uploads/B1NrxsG16.png)

---

🏎 Maximum speed Models like Glove offer high speed, but suffer from a lack of context awareness resulting in low average MTEB scores.

⚖️ Speed and performance Slightly slower, but significantly stronger, ==all-mpnet-base-v2== or all-MiniLM-L6-v2 provide a good balance between speed and performance.

💪 Maximum performance Multi-billion parameter models like ST5-XXL, GTR-XXL or SGPT-5.8B-msmarco dominate on MTEB. They tend to also produce bigger embeddings like SGPT-5.8B-msmarco which produces 4096 dimensional embeddings requiring more storage!

---

## Vector DB

Chroma

---

## Questions

27 questions

```json
{
    "management":"Does the company have a clear strategy for growth and innovation? Are there any recent strategic initiatives or partnerships?"
}

---

```
1. feature_overall	
2. feature_revenue_1	
3. feature_revenue_2	
4. feature_revenue_3	
5. feature_profit_1	
6. feature_profit_2	
7. feature_debt_1	
8. feature_cashflow_2	
9. feature_dividend	
10. feature_management_1	
11. feature_management_2	
12. feature_industry_1	
13. feature_industry_2	
14. feature_research	
15. feature_guidance	
16. feature_leadership	
17. feature_macro	
18. feature_diversification	
19. feature_customerbase	
20. feature_esg	
21. feature_competition_1	
22. feature_competition_2	
23. feature_ip	
24. feature_digitaltransformation
25. feature_regulations	
26. feature_onlinepresence	
27. feature_legal


---

## Model

- Linear Regression: enforces non-negativity in the coefficients


- X
    - Confidence Scores from GPT-3.5-Turbo
- Y: percentage return is calculated between two successive annual report filing dates
    - target_custom_22 
        - Annual Return
        - target created from 12 month normalised returns per year (aka era) and then binned
    - target_custom_2
        - 98th percentile of return from the filing date represented as target max
        - target created from max normalised returns in the span of 12 months per year (aka era) and then binned


---

## Result

----


<!-- .slide: data-background="https://hackmd.io/_uploads/BksjiDtr6.png" -->

----

<!-- .slide: data-background="https://hackmd.io/_uploads/HyC4hwYST.png" -->


----


<!-- .slide: data-background="https://hackmd.io/_uploads/B1CVnPKHp.png" -->

----

<!-- .slide: data-background="https://hackmd.io/_uploads/HyCVnwFra.png" -->

----




<!-- .slide: data-background="https://hackmd.io/_uploads/rJCE2PFHa.png" -->

----

<!-- .slide: data-background="https://hackmd.io/_uploads/rJ0VhvFB6.png" -->

----

- 挑選股票
    - 每個月月底預測下個月最強的五檔股票
    - 若上個月所挑出來的股票比較強，則會被保留下來
    - 綜合上述做法，挑出五檔股票
- 月初開盤進場，月底收盤出場


----

<!-- .slide: data-background-iframe="media/cumprod_amount_2018_2022.html" -->


----

<!-- .slide: data-background-iframe="media/cumprod_amount_2002_2022.html" -->


----

- 避開突發的損失
    - 下跌 10 % 就停止交易
    - 觀察下個月的情況，若可獲利，就進場下單


----

<!-- .slide: data-background-iframe="media/cumprod_amount_2018_2022_trick.html" -->


----


<!-- .slide: data-background-iframe="media/cumprod_amount_2002_2022_trick.html" -->

----

<!-- .slide: data-background-iframe="media/annual_return_2002_2022.html" -->


----

- 年化報酬率：14%
- 最大交易回落：28%

----

<!-- .slide: data-background-iframe="media/cumprod_amount_2023.html" -->

---

分年分配目標值：
目的：為了在每年內相對排名股票的回報。
方法：每年單獨分配股票的目標值，這樣可以更好地比較同一年內不同股票的表現。

回報的排名和標準化：
首先：回報首先被排名，這意味著根據其回報率對股票進行排序。
然後：然後對排名後的回報進行標準化，這有助於消除數據中的任何潛在偏差或異常值。

目標值的範圍限制：
範圍：[0, 1]。
解釋：其中1表示更高的回報，這意味著目標值1表示該股票在該年度有更高的回報。

基於百分位數的分箱：
最後：對標準化的回報進行分箱，基於百分位數，這樣目標值的範圍是[0, 1]。
目的：這有助於創建一個更均勻和有序的目標值分布，可以更好地用於機器學習模型的訓練。

----

