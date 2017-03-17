python -m translate \
       --data_dir ./data\
       --train_dir ./checkpoint\
       --from_train_data ./data/all_modern_train.ids\
       --to_train_data ./data/all_original_train.ids\
       --from_dev_data ./data/all_modern_dev.ids\
       --to_dev_data ./data/all_original_dev.ids\
       --num_layers 4 \
       --size 512 \
       --steps_per_checkpoint 1000 \
       --learning_rate 0.1 \
       --learning_rate_decay_factor 0.9 \
       --from_vocab_size 12359\
       --to_vocab_size 14580\
	   --decode 1\
	   --beam_search 1

