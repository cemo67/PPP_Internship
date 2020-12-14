import sys
sys.path.append("../")
from PPP_config import get_config
from Data.toy_data_class import toy_data_class


config_dict = get_config()
DATA_NAME_LIST = config_dict['file_list']
samples_ = config_dict['samples']

bl = False
mo = False
ci = False
co = False

for name in DATA_NAME_LIST:
    for sample in samples_:
        t = toy_data_class(samples=sample, name=name)

        if name == 'blobs':
            t.blobs()
            if bl:
                print(name)
                bl = False
                t.plot_data()
        elif name == 'moons':
            t.moons()
            if mo:
                print(name)
                mo = False
                t.plot_data()
        elif name == 'circles':
            t.circles()
            if ci:
                print(name)
                ci = False
                t.plot_data()
        elif name == 'cosine':
            t.cosine()
            if co:
                print(name)
                co = False
                t.plot_data()

        t.save()