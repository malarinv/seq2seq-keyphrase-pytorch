
##### Setup
* `$ pip install -r requirements.txt`

##### TRAINING
* extract [test](https://drive.google.com/open?id=1Jh8Suuk6sTKuK-mbpvU5KfiQKi9zAGar) to `data/`
  - The above dataset is a subset of the training data to test the pipeline
  - to generate the actual dataset
    - extract [train](https://drive.google.com/file/d/1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR/view) to `./`
    - run `$ python preprocess.py -dataset_name kp20k -source_dataset_dir ./kp20k_new/`
* to start training run `$ python train.py -data ./data/kp20k -vocab ./data/kp20k.vocab.pt`
