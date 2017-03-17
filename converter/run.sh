python -m translate \
       --data_dir ./data\
       --train_dir ./checkpoint\
       --from_train_data ./data/all_modern_train.ids\
       --to_train_data ./data/all_original_train.ids\
       --from_dev_data ./data/all_modern_dev.ids\
       --to_dev_data ./data/all_original_dev.ids\
       --num_layers 2 \
       --size 256 \
       --steps_per_checkpoint 500 \
       --learning_rate 0.5 \
       --learning_rate_decay_factor 0.9 \
       --from_vocab_size 14714\
       --to_vocab_size 17456\
	   --decode 1\
	   --beam_search 1

