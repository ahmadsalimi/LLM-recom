from pytorch_lightning.cli import LightningCLI
from data.session.text_module import SessionTextDataModule
from training_free.module import TrainingFreeModule


def cli_main():
    cli = LightningCLI(TrainingFreeModule, SessionTextDataModule)
    # note: don't call fit!!


if __name__ == '__main__':
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
