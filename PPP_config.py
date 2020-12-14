from Pertubations.numeric_class import Column, Scale, NoPertubation

def get_pertubation_list(X, data):
    # Pertubation
    #DEFAULT_FRACTION = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #DEFAULT_SCALER = [1, 5, 10, 15, -0.3]

    DEFAULT_FRACTION = [0.1, 0.5, 0.9]
    DEFAULT_SCALER = [2, 38]

    PERTUBATION_LIST = [('No_Perturbation', NoPertubation(X))]
    for frac in DEFAULT_FRACTION:
        PERTUBATION_LIST.append(('Column_fraction_' + str(frac), Column(X, fraction=frac, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))
    for frac in DEFAULT_FRACTION:
        for sca in DEFAULT_SCALER:
            PERTUBATION_LIST.append(('Scale_fraction_' + str(frac) + '_scaler_' + str(sca) , Scale(X, fraction=frac, scaler=sca, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))

    return PERTUBATION_LIST

def get_config():
    file_list = ['blobs', 'moons', 'circles']#, 'cosine']
    k_range = 15
    samples = [50, 100]#, 1000, 5000]#, 10000]
    rows_plot = 5
    cols_plot = 2
    PLOT_ALL_IN_ONE = False
    PLOT = True

    dict = { 'file_list': file_list,
             'k_range': k_range,
             'samples': samples,
             'rows_plot': rows_plot,
             'cols_plot': cols_plot,
             'PLOT_ALL_IN_ONE': PLOT_ALL_IN_ONE,
             'PLOT': PLOT}

    return dict