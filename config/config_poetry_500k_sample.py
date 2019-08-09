
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 32,
    'test_nepoch': 4,
    'train_data': 'datasets/poetry_500k_sample_data/poetry_500k_sample.train.txt',
    'val_data': 'datasets/poetry_500k_sample_data/poetry_500k_sample.valid.txt',
    'test_data': 'datasets/poetry_500k_sample_data/poetry_500k_sample.test.txt'
}
