def get_dataset_name_and_rng(dataset_name):
    if 'rng' in dataset_name:
        if 'test' in dataset_name:
            raise ValueError(
                f'Dynamic mixing should not be activated on test '
                f'datasets to ensure reproducibility (i.e., no "rng" '
                f'in the dataset name: {dataset_name})'
            )

        try:
            dataset_name, seed = dataset_name.split('_rng')
        except ValueError:
            raise ValueError(
                f'Expected "<original_dataset_name>_rng[seed]" '
                f'(e.g., train_si284_rng), not {dataset_name}'
            ) from None

        if seed != '':
            rng = int(seed)
        else:
            rng = True
    else:
        rng = False
    return dataset_name, rng