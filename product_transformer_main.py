from pytorch_lightning.cli import LightningCLI

from data.session.vector_module import SessionVectorDataModule
from product_transformer.module import ProductTransformerModule


def cli_main():
    cli = LightningCLI(ProductTransformerModule, SessionVectorDataModule)
    # note: don't call fit!!


if __name__ == '__main__':
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
