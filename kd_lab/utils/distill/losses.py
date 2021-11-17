from typing import Optional
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(
        self,
        distillation_type: Optional[str]='hard',
        tau: Optional[float]=1.0
    ) -> None:
        super().__init__()
        assert distillation_type in ['soft', 'hard']
        self.distillation_type = distillation_type
        self.tau = tau

    def forward(
        self,
        logits: Tensor,
        teacher_logits: Tensor
    ) -> Tensor:
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            logits: the outputs of the model to be trained. It is expected to be
                a Tensor
            teacher_logits: the output of the teacher model
        """
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / logits.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(logits, teacher_logits.argmax(dim=1))

        return distillation_loss
