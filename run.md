# CoVS

Source code of CoVS for Visual Storytelling task.

## Environments

- CUDA 10.1
- python 3.6
- pytorch 1.6.0

## Run

### Datasets

Download VIST ResNet152 features and put in project directory, make sure in `src_xxx/dataset.py` the path is correct. (Acutally you can generate ResNet152 features from original dataset)


### Generate Rake or LDA topics

```bash
cd src_xxx
python dataset.py
```

More details in `dataset.py`. For example, you need to call `VISTDataset.init_keywords_lda()` to generate LDA topics.

### Train or Test model

```bash
cd src_xxx
python main.py
```

You can change options(including 'train' or 'test') and hyperparameters at `src_xxx/main.py`.

## Acknowledgement

* [AREL](https://github.com/eric-xw/AREL)
* [VIST evaluation code](https://github.com/lichengunc/vist_eval), and [a python3 version used here](https://github.com/Sefaice/vist_eval)

## License

MIT