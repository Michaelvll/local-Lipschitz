python train.py --batch-size 64 ^
                --test-batch-size 32 ^
                --epochs 20 ^
                --lr 0.001 ^
                --momentum 0.5 ^
                --log-interval 5 ^
                --save-model models/mnist_cnn_lip1.0.pt ^
                --lip-lambda 1.0
