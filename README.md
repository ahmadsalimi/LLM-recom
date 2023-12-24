# LLM-recom


LLM-recom is a recommendation system that leverages two types of models: one for generating embeddings for products and another for training transformers on these product embeddings.

## Generating Product Embeddings

You can use the Llama model to generate product embeddings with the following command:

```bash
python embed_products.py predict --config configs/embed_products_llama.yaml
```

After generating the embeddings, employ the numpy_to_parquet.py script to convert them to the Parquet format. This step organizes the data into big data chunk sizes, facilitating easier transfer to RAM.

## Training Transformer
To train the transformer model, execute the following command:

```bash
python embed_products.py predict --config configs/product_transformer_llama_uk.yaml
```

This command initiates the training process using the specified configuration file for the transformer model.

Before running these commands, ensure to customize the configuration files (**embed_products_llama.yaml** and **product_transformer_llama_uk.yaml**) to meet your specific requirements.


