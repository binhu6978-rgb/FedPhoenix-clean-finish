CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cifar10': ([6, 'R', 'M', 16, 'R', 'M', 'F'], 3, 10, 400, 120, 84, 0),
    'cifar100': ([6,'R', 'M', 16, 'R','M', 'F'], 3, 20, 400, 120, 84, 0),
    'femnist': ([16, 'M', 'R', 32, 'M', 'R', 'F'], 1, 62, 512, 256, 0),
}

GENERATORCONFIGS = {
    'cifar10': {
        'cnn': (512, 84, 3, 10, 100),
        'vgg': (512, 4096, 3, 10, 100),
        'vggdrop': (512, 4096, 3, 10, 100), 
        'resnet18': (512, 512, 3, 10, 100),
        'resnet_drop': (512, 512, 3, 10, 100),
        'mobnet': (512, 1024, 3, 10, 100),
    },
    'timage': {
        'resnet18': (512, 512, 3, 200, 100),
        'resnet_drop': (512, 512, 3, 200, 100),
        'resnet_drop25': (512, 512, 3, 200, 100),
        'vgg': (512, 4096, 3, 200, 100),
        'vggdrop': (512, 4096, 3, 200, 100),
        'mobnet': (512, 1024, 3, 200, 100),
    },
    'cifar100': {
        'cnn': (512, 84, 3, 20, 100),
        'vgg': (512, 4096, 3, 20, 100),
        'vggdrop': (512, 4096, 3, 20, 100),
        'resnet18': (512, 512, 3, 20, 100),
        'resnet_drop': (512, 512, 3, 20, 100),
        'mobnet': (512, 1024, 3, 20, 100),
    },
}
FedGenRUNCONFIGS = {
    'femnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10,  # used to regulate user training
            'weight_decay': 1e-2
        },
    'cifar10':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'generative_alpha': 0.2,
            'generative_beta': 0.2,
            'weight_decay': 1e-2
        },
    'cifar100':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'generative_alpha': 10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

}

FedGKDRUNCONFIGS = {
    'femnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10,  # used to regulate user training
            'weight_decay': 1e-2
        },
    'cifar10':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0.5,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'generative_alpha': 0,
            'generative_beta': 0.1,
            'weight_decay': 1e-2
        },
    'cifar100':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'generative_alpha': 10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

}

