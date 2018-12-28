
##### Setup
* `$ pip install -r requirements.txt`

##### TRAINING
* extract [test](https://drive.google.com/open?id=1Jh8Suuk6sTKuK-mbpvU5KfiQKi9zAGar) to `data/`
  - The above dataset is a subset of the training data to test the pipeline
  - to generate the actual dataset
    - extract [train](https://drive.google.com/file/d/1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR/view) to `./`
    - run `$ python preprocess.py -dataset_name kp20k -source_dataset_dir ./kp20k/`
* set cuda visible devices `export CUDA_VISIBLE_DEVICES=0`
* to start training run `$ python -m train -data data/kp20k/kp20k -vocab data/kp20k/kp20k.vocab.pt -exp_path "./exp-1/attn_general.input_feeding.copy/%s.%s" -model_path "./model-1/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-1/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-1" -batch_size 256 -bidirectional -copy_attention -run_valid_every -1 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding`
* to start training on small dataset run `$ python -m train -data data/kp20k_small/kp20k -vocab data/kp20k_small/kp20k.vocab.pt -exp_path "./exp-small/attn_general.input_feeding.copy/%s.%s" -model_path "./model/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-small" -batch_size 256 -bidirectional -copy_attention -run_valid_every -1 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding`

##### PREDICTION

* run `$ python -m predict_keyphrase -data data/kp20k/kp20k -vocab data/kp20k/kp20k.vocab.pt -exp_path "./exp-1/attn_general.input_feeding.copy/%s.%s" -model_path "./model-1/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-1/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-1" -batch_size 256 -bidirectional -copy_attention -run_valid_every -1 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding`


##### Custom TRAINING
* extract [test](https://drive.google.com/open?id=1Jh8Suuk6sTKuK-mbpvU5KfiQKi9zAGar) to `data/`
  - The above dataset is a subset of the training data to test the pipeline
  - to generate the actual dataset
    - create the json files
    - run `$ python custom_preprocess.py -dataset_name kp20k -source_dataset_dir product_extraction/ -output_path_prefix data_custom`
* set cuda visible devices `export CUDA_VISIBLE_DEVICES=0,1`
* to start training run `$ python -m train -data data_custom/kp20k/kp20k -vocab data_custom/kp20k/kp20k.vocab.pt -exp_path "./exp-custom/attn_general.input_feeding.copy/%s.%s" -model_path "./model/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-custom/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-custom" -batch_size 256 -bidirectional -copy_attention -run_valid_every 2000 -save_model_every 2000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding -vocab_size 2107 -device_ids 0 1`
* to start training on small custom dataset run `$ python -m train -data data_custom/kp20k_small/kp20k -vocab data_custom/kp20k_small/kp20k.vocab.pt -exp_path "./exp-custom/attn_general.input_feeding.copy/%s.%s" -model_path "./model/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-custom/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-custom" -batch_size 256 -bidirectional -copy_attention -run_valid_every -1 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding -vocab_size 2100 -device_ids 0 1`

##### Custom PREDICTION

* run `$ python -m predict_keyphrase -data data_custom/kp20k/kp20k -vocab data_custom/kp20k/kp20k.vocab.pt -exp_path "./exp-custom/attn_general.input_feeding.copy/%s.%s" -model_path "./model/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-custom/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-custom" -batch_size 256 -bidirectional -copy_attention -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding -min_src_seq_length 5`
