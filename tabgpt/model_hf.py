from transformers import PretrainedConfig
import torch
from transformers import PreTrainedModel
from tabgpt.model import tabGPT


class tabGPTConfig(PretrainedConfig):
    model_type = "tabGPT"

    def __init__(self,
        n_layer = None,
        n_head = None,
        block_size = None,
        n_output_nodes = None,
        resid_pdrop = 0.1,
        attn_pdrop = 0.1, 
        **kwargs):
        super().__init__(**kwargs)
        self.model_type = None
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd =  768
        self.vocab_size = 50257
        self.block_size = block_size
        self.n_output_nodes = n_output_nodes
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop


class tabGPT_HF(tabGPT, PreTrainedModel):
    config_class = tabGPTConfig

    def __init__(self, config, *args, **kwargs):
        PreTrainedModel.__init__(self, config)
        tabGPT.__init__(self, config, *args, **kwargs)

    def forward(self, x=None, targets=None, return_dict=False, **kwargs):
        logits, loss = super(tabGPT_HF, self).forward(x=x, targets=targets)

        # If return_dict is True, return a dictionary
        if return_dict:
            return {"logits": logits, "loss": loss}

        # Otherwise, return logits directly (as a tuple)
        return logits, loss
    


if __name__ == '__main__':

    n_layer, n_head, n_embd = 4, 4, 768 # gpt-micro
    config = tabGPTConfig(n_layer=4, n_head=4, block_size=10, n_output_nodes=1)

    model = tabGPT_HF(config)

    logits, loss = model(x=torch.randn(16,10,768), targets=torch.randn(16))
   