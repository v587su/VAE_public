## VAE filtering model
Code for the VAE filtering model in our paper "On the importance of Building High-quality Training Datasets for Neural Code Search".

### Data Preparation
We provide the Real Query Corpus in `./data`. You can make your own corpus using our data processing script `data_process.py`. 

### Train
To train our model:
```
python main.py --train_data_dir ./data
```

