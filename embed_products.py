from pytorch_lightning.cli import LightningCLI
from data.product.module import ProductDataModule
from product_embedding.module import ProductEmbeddingModule


def cli_main():
    cli = LightningCLI(ProductEmbeddingModule, ProductDataModule)
    # note: don't call fit!!


if __name__ == '__main__':
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
