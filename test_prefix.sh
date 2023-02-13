TRAIN_METHOD=direct
TEST_METHOD=direct
LR=1e-2
N_PREFIX=10
DATASET=ethos-religion
TRAIN_TASK=tune
SPLIT=train
MODEL=gpt2-large
TRAIN_SIZE=100
STEP=100000
K=4
CUDA_VISIBLE_DEVICES=2 python test.py\
    --dataset $DATASET\
    --gpt $MODEL\
    --method $TEST_METHOD\
    --test_batch_size 16\
    --out_dir out/$MODEL-prefix\
    --n_prefix_tokens $N_PREFIX\
    --use_soft_prefix\
    --prefix_embed_file checkpoints/gpt2-large/$TRAIN_TASK-$SPLIT/prefix={$N_PREFIX}-{$TRAIN_METHOD}-lr={$LR}-initByVocab/soft_embeddings-$STEP.pt\
    # --seed 100\
