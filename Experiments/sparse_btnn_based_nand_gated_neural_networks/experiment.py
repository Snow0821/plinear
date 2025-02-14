import torch
from torch import nn
from torchmetrics.classification import Accuracy

import lightning as L

class LitModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.example_input_array = torch.Tensor(64, 1, 28, 28)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "accuracy": acc}
    
    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        print(f"\nEpoch End - Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"val_accuracy": acc}

    def validation_step_epoch_end(self, outputs):
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        print(f"Validation Accuracy: {avg_acc:.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_accuracy", acc)
        return {"test_accuracy": acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)
        return optimizer

if __name__ == "__main__":
    from model import mnist_classifier, mnist_classifier_Compacted_Nand, mnist_classifier_conv2d
    from lightning.pytorch.callbacks import ModelSummary

    experiment = LitModule(mnist_classifier())
    # summary = ModelSummary(exp, max_depth=-1)
    # print(summary)

    from data import MNISTDataModule
    data_module = MNISTDataModule()

    print(data_module)

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("logs", name="mnist_nand_cnn")
    trainer = L.Trainer(callbacks=[ModelSummary(max_depth=-1)], max_epochs=20, logger = logger)
    trainer.fit(experiment, datamodule=data_module)

    # 테스트
    test_results = trainer.test(datamodule=data_module)
    print("Test Results:", test_results)