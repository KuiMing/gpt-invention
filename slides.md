# Gpt Invention

Start from here!

---

<!-- .slide: data-background="#999999" -->

<!-- .slide: data-background-iframe="media/flow.html" -->

---

## Data

### Annual Report

- [Financial Modeling Prep](https://site.financialmodelingprep.com/)
    - https://site.financialmodelingprep.com/developer/docs/#Annual-Reports-on-Form-10-K

### Stock Price
- total 1505 stocks
    - S&P500
    - S&P400
    - S&P600
- package: `openbb`
    - https://openbb.co/
    - ![](https://hackmd.io/_uploads/BktMtfN1a.png)

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

![](https://hackmd.io/_uploads/ryQ3nlmk6.png)

---

![](https://hackmd.io/_uploads/BJX3ngXkp.png)

---

![](https://hackmd.io/_uploads/rkXhhx7J6.png)

---

![](https://hackmd.io/_uploads/Hy7hne7yT.png)

---

![](https://hackmd.io/_uploads/rkQh3e7J6.png)

---

![](https://hackmd.io/_uploads/Symnhl7J6.png)
