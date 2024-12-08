# Bert2MoE

- 将BERT中的FFN换成MixtralSparseMoeBlock，基于huggingface transformers

## 1. Run Demo

```python
import torch

from Bert2MoE.modeling_bert2moe import Bert2MoELayer
from Bert2MoE.configuration_bert2moe import Bert2MoEConfig

from transformers.models.bert import modeling_bert
from transformers.models.bert.modeling_bert import BertModel

modeling_bert.BertLayer = Bert2MoELayer

device = torch.device("cuda:0")

config = Bert2MoEConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=6,
    intermediate_size=2048,
    num_attention_heads=8,
    num_local_experts=8,
    num_experts_per_tok=2
)

model = BertModel(config=config).to(device)
input_ids = torch.randint(0, 1000, size=(2, 10)).to(device)

outputs = model(input_ids)

print(outputs.last_hidden_state.shape)
```

