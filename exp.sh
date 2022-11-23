for dataset in 'mnist' 'mnist_m' 'svhn' 'syn'
do
    # Method 1
    python src/train.py --target_domain $dataset --domain_dim 64 --disentangle_layer 3 --logdir method1/$dataset

    # Method 2/3
    python src/train.py --target_domain $dataset --domain_dim 128 --disentangle_layer 2 --logdir method3/$dataset
    python src/train.py --target_domain $dataset --domain_dim 128 --disentangle_layer 2 --finetune --logdir method3-finetune/$dataset
done
