# Fake-EmoReact 2021 Challenge (Student)
https://competitions.codalab.org/competitions/31180?secret_key=2f97f399-8bba-4ed5-a0b7-99e17df1fe1b#participate


# model 
BERT-Base (L-12_H-768_A-12) + dropout + classifier

# Confusion_Matrix
![](https://i.imgur.com/6egfWc4.png)

# testing 

| method | accurracy | precision | recall |
| -------- | -------- | -------- | -------- |
| TF-IDF + Xgboost | 0.966 |0.971 | 0.82 |
| GloVe + LSTM | 0.971 | 0.976 | 0.989 |
| GloVe + Bi-LSTM | 0.971 | 0.971 | 0.989 |
| BERT-Mini + dropout + classifier | 0.971 | 0.976 | 0.989 | 
| BERT-Base + dropout + classifier | 0.993 | 0.997 | 0.998 |

# eval 
BERT-Base (L-12_H-768_A-12)

* Precision score : 0.7394(18)
* Recall score : 0.6664(13)
* F1 score: 0.6353(14)


# Detial
https://github.com/linzino7/Fake-EmoReact-2021-Challenge/blob/main/NLP_Fakenews.pdf


