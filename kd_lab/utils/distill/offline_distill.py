from typing import Any, Callable, Tuple
from cfg import Opts
from mlutils import gen, mod
from torch.functional import Tensor
from torch.optim import SGD
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import StepLR
from torch import nn
import torch

from .distill import BaseDistillTrainer
from .losses import DistillationLoss


__all__ = ['OfflineDistillTrainer']


class OfflineDistillTrainer(BaseDistillTrainer):
    @gen.synchrony
    def __init__(
        self,
        opt: Opts,
        student_model: nn.Module,
    ) -> None:
        assert opt.get('teacher_id', None) is not None
        opt.set('exp_id', opt.teacher_id)
        super().__init__(opt)
        teacher_model = self.model_manager.load_model()
        self.teacher_model = self.setup_teacher(teacher_model)
        self.student_model = student_model
        self.student_optimizer = create_optimizer_v2(
            student_model,
            opt.get('student_optimizer', 'sgd'),
            learning_rate=opt.student_lr,
            weight_decay=opt.get('student_weight_decay', 1.0e-4)
        )
        self.student_scheduler = StepLR(self.student_optimizer, 20, 0.95)
        self.teacher_model = yield self.to_gpu(self.teacher_model)
        self.student_model = yield self.to_gpu(self.student_model)
        self.actual_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = DistillationLoss(
            distillation_type=opt.get('distillation_type', 'hard'),
            tau=opt.get('distillation_tau', 1.0)
        )

    def train_teacher_step(
        self, *item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        images, labels = item
        with torch.no_grad():
            logits = self.teacher_model(images)
            loss = self.actual_loss_fn(logits, labels)
        return loss, logits

    def train_student_step(
        self, *item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        images, labels, teacher_logits = item

        self.student_optimizer.zero_grad()
        student_out = self.student_model(images)
        if isinstance(student_out, tuple):
            logits = student_out[0]
            dist_logits = student_out[1]
        else:
            logits = student_out
            dist_logits = student_out
        actual_loss = self.actual_loss_fn(logits, labels)
        distill_loss = self.distill_loss_fn(dist_logits, teacher_logits)
        loss = actual_loss + distill_loss
        loss.backward()
        self.student_optimizer.step()

        return loss, logits

    @gen.detach_cpu
    @gen.synchrony
    def train_step(self, item):
        images, labels = item
        images = yield self.to_gpu(images)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)

        teacher_loss, teacher_logits = self.train_teacher_step(
            images, labels
        )
        student_loss, student_logits = self.train_student_step(
            images, labels, teacher_logits
        )

        total_loss = teacher_loss + student_loss
        self.teacher_loss_meter.append(teacher_loss.detach())
        self.student_loss_meter.append(student_loss.detach())

        self.show_images('train_image', images)
        preds = self.logit_to_pred(student_logits)
        return total_loss, preds, labels

    @gen.detach_cpu
    @gen.synchrony
    def eval_step(self, item):
        images, labels = item
        images = yield self.to_gpu(images)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)

        logits = self.student_model(images)
        loss = self.actual_loss_fn(logits, labels)

        self.show_images('eval_image', images)
        preds = self.logit_to_pred(logits)
        return loss, preds, labels

    @gen.synchrony
    def inference(self, inp: Tensor) -> Tensor:
        inp = yield self.to_gpu(inp)

        if inp.ndim == 3:
            inp = inp.unsqueeze(0)

        with torch.no_grad():
            logits = self.net(inp)

        self.show_images('inference_image', inp)
        preds = self.logit_to_pred(logits)
        preds = yield self.to_cpu(preds.detach())
        return preds

    def on_epoch_end(self) -> None:
        self.student_scheduler.step()
        super().on_epoch_end()

    def setup_teacher(
        self,
        teacher_model: nn.Module
    ) -> nn.Module:
        for p in teacher_model.parameters():
            p.requires_grad = False
        return teacher_model
