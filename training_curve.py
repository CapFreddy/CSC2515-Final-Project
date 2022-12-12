import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(npz_path, save_path):
    obj = np.load(npz_path)
    train_loss = obj['train_loss']
    val_object_acc = obj['val_object_acc']
    plt.plot(train_loss, label='train loss')
    plt.plot(val_object_acc, label='validation acc')

    if 'val_domain_acc' in obj:
        val_domain_acc = obj['val_domain_acc']
        plt.plot(val_domain_acc, label='domain acc')

    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=300)
    plt.clf()


method1_path = './result/method1/domain_dim_64-alpha_0.1/seed_42/mnist_m/phase1.npz'
method2_path = './result/method3/domain_dim_64-alpha_0.1/seed_42/mnist_m/phase1.npz'
method3_path = './result/method3/domain_dim_64-alpha_0.1/seed_42/mnist_m/phase2.npz'
plot_training_curve(method1_path, 'plot/curve/method1.png')
plot_training_curve(method2_path, 'plot/curve/method2.png')
plot_training_curve(method3_path, 'plot/curve/method3.png')
