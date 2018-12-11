import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

config = Config(*sys.argv[1:])

dl = DataLoader(config)
dl.load()

# for each epoch
    # check logs
    # load model
    # run stats (layer.cont_repr_weights)

model = Model(dl, config)
model.build_graph()
model.run_experiment()
model.close()
