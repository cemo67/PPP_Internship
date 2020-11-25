from Pertubations.numeric_class import Column, Scale, NoPertubation

def get_pertubation_list(X, data):
    # Pertubation
    DEFAULT_FRACTION = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    DEFAULT_SCALER = [1, 5, 10, 15, -0.3]

    PERTUBATION_LIST = [('No_Perturbation', NoPertubation(X))]
    for frac in DEFAULT_FRACTION:
        PERTUBATION_LIST.append(('Column_fraction_' + str(frac), Column(X, fraction=frac, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))
    for frac in DEFAULT_FRACTION:
        for sca in DEFAULT_SCALER:
            PERTUBATION_LIST.append(('Scale_fraction_' + str(frac) + '_scaler_' + str(sca) , Scale(X, fraction=frac, scaler=sca, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))

    return PERTUBATION_LIST

def get_config():
    file_list = ['blobs']
    k_range = 10
    samples = 100

    dict = { 'file_list': file_list,
             'k_range': k_range,
             'samples': samples}

    return dict