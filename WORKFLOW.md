
##### Setup
* `$ pip install -r requirements.txt`

##### TRAINING
* extract [test](https://drive.google.com/open?id=1Jh8Suuk6sTKuK-mbpvU5KfiQKi9zAGar) to `data/`
  - The above dataset is a subset of the training data to test the pipeline
  - to generate the actual dataset
    - extract [train](https://drive.google.com/file/d/1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR/view) to `./`
    - run `$ python preprocess.py -dataset_name kp20k -source_dataset_dir ./kp20k_new/`
* set cuda visible devices `export CUDA_VISIBLE_DEVICES=0`
* to start training run `$ python -m train -data data/kp20k/kp20k -vocab data/kp20k/kp20k.vocab.pt -exp_path "./exp-1/attn_general.input_feeding.copy/%s.%s" -model_path "./model-1/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-1/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-1" -batch_size 256 -bidirectional -copy_attention -run_valid_every -1 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding`

##### PREDICTION
