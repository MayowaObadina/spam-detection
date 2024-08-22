export MAX_LENGTH=256
export BATCH_SIZE=16
export NUM_EPOCHS=1
export SAVE_STEPS=500000
export BERT_MODEL=google-bert/bert-base-uncased

for j in 1
do
  export SEED=$j
  export OUTPUT_FILE=test_result_$j
  export OUTPUT_PREDICTION=test_predictions_$j
  export DATA_DIR=../data
  export OUTPUT_DIR=spam_bert

  CUDA_VISIBLE_DEVICES=0 python3 train_textclass.py --data_dir $DATA_DIR \
  --model_type bert \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --output_result $OUTPUT_FILE \
  --output_prediction_file $OUTPUT_PREDICTION \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate 2e-5 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --gradient_accumulation_steps 2 \
  --do_train \
  --do_predict \
  --overwrite_output_dir
  done
