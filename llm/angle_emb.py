from typing import List

import torch
from torch import nn
from angle_emb import AnglE, Prompts

class LLM(nn.Module):

    def __init__(self, model_name: str, pretrained_lora_path:str,pooling_type: str,prompt : str):
        super(LLM, self).__init__()

        if model_name.split("/")[-1] == "UAE-Large-V1":
            self.angle_model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_type).cuda()
            self.angle_model.set_prompt(prompt=prompt)
        if model_name.split("/")[-1] == "Llama-2-7b-hf":
            self.angle_model = AnglE.from_pretrained(model_name,pretrained_lora_path = pretrained_lora_path).cuda()
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
        return tensor_output.float()

class BERT(nn.Module):

    def __init__(self, model_name: str, pooling_type: str):
        super(BERT, self).__init__()
        self.angle_model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_type).cuda()



    def forward(self, products: List[str]) -> torch.Tensor:
        """
        Arguments:
            products: List[str], list of formatted product strings.

        Returns:
            output: Tensor, shape [B, D].
        """
        

        tensor_output = self.angle_model.encode(products, to_numpy=False)
        return tensor_output