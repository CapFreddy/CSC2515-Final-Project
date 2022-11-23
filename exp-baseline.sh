for dataset in 'mnist' 'mnist_m' 'svhn' 'syn'
do
    # Method 1
    python src/train_baseline.py --target_domain $dataset --logdir baseline/$dataset
done
