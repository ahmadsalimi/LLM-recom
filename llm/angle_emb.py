from typing import List

import torch
from torch import nn
from angle_emb import AnglE, Prompts

class LLM(nn.Module):

    def __init__(self, model_name: str, pooling_type: str,prompt : str):
        super(LLM, self).__init__()

        self.angle_model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_type).cuda()
        self.angle_model.set_prompt(prompt=prompt)



    def forward(self, products: List[str]) -> torch.Tensor:
        """
        Arguments:
            products: List[str], list of formatted product strings.

        Returns:
            output: Tensor, shape ``[B, D]``.

            
        """

        input_data = [{'text': text} for text in products]
        tensor_output = self.angle_model.encode(input_data, to_numpy=False)
        return tensor_output
  
