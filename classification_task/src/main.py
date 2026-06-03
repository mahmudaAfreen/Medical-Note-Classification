from pytorch_lightning.cli import LightningCLI
from bert_models import BertClassificationModel
#from prototype_models import MultiProtoModule
from datamodule import MIMICClassificationDataModule

if __name__ == '__main__':
 cli = LightningCLI(BertClassificationModel, MIMICClassificationDataModule)
