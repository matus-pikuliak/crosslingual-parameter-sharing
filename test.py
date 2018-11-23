import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

config = Config(*sys.argv[1:])

dl = DataLoader(config)
dl.load()

model = Model(dl, config)
model.build_graph()
model.run_experiment()
model.temp_dep_faults()
# model.temp_export_representations()
model.close()
