export PYTHONPATH=$PWD:$PYTHONPATH
python src/baseline/baseline.py --target_domain mnist -s 1 -o output/target-mnist-seed-1.csv
python src/baseline/baseline.py --target_domain mnist -s 2 -o output/target-mnist-seed-2.csv
python src/baseline/baseline.py --target_domain mnist -s 3 -o output/target-mnist-seed-3.csv
python src/baseline/baseline.py --target_domain mnist_m -s 1 -o output/target-mnist_m-seed-1.csv
python src/baseline/baseline.py --target_domain mnist_m -s 2 -o output/target-mnist_m-seed-2.csv
python src/baseline/baseline.py --target_domain mnist_m -s 3 -o output/target-mnist_m-seed-3.csv
python src/baseline/baseline.py --target_domain svhn -s 1 -o output/target-svhn-seed-1.csv
python src/baseline/baseline.py --target_domain svhn -s 2 -o output/target-svhn-seed-2.csv
python src/baseline/baseline.py --target_domain svhn -s 3 -o output/target-svhn-seed-3.csv
python src/baseline/baseline.py --target_domain syn -s 1 -o output/target-syn-seed-1.csv
python src/baseline/baseline.py --target_domain syn -s 2 -o output/target-syn-seed-2.csv
python src/baseline/baseline.py --target_domain syn -s 3 -o output/target-syn-seed-3.csv