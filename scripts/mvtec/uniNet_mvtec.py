from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import UniNet
from anomalib import TaskType
from anomalib.utils.normalization import NormalizationMethod

datamodule = MVTecAD(
    root=".\datasets\MVTecAD", 
    category="bottle", 
    train_batch_size=12, 
    eval_batch_size=12, 
    num_workers=3
    )

model = UniNet()

engine = Engine(
    max_epochs=10
    )

if __name__ == '__main__':
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(model=model, datamodule=datamodule)