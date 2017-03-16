python -m translate \
       --data_dir ./data\
       --train_dir ./checkpoint\
       --from_train_data ./data/all_modern_train.ids\
       --to_train_data ./data/all_original_train.ids\
       --from_dev_data ./data/all_modern_dev.ids\
       --to_dev_data ./data/all_original_dev.ids\
       --num_layers 2 \
       --size 256 \
       --steps_per_checkpoint 50 \
       --learning_rate 0.01 \
       --learning_rate_decay_factor 0.9 \
       --from_vocab_size 10000\
       --to_vocab_size 10000\

