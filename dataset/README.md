# Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

The dataset is not included in this repository because GitHub has file size limits.

## Download

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## After Download

Copy these files into this folder:

```
dataset/
│
├── Fake.csv
└── True.csv
```

Then train the model:

```bash
python train_model.py
```

Finally run:

```bash
python app.py
```