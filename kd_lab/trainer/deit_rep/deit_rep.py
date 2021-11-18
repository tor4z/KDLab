from typing import Tuple, Any
import torch
from torch import Tensor
from cfg import Opts
from mlutils import mod, gen
from kd_lab.utils.distill import\
    OfflineDistillTrainer, OnlineDistillTrainer
from kd_lab.utils.distill.losses import DistillationLoss
from kd_lab.network.deit.deit import get_deit
from kd_lab.network.resnet import ResNet


__all__ = [
    'OfflineResNetDeitRepDistiller',
    'OnlineResNetDeitRepDistiller'
]



@mod.register('arch')
class OfflineResNetDeitRepDistiller(OfflineDistillTrainer):
    def __init__(
        self, opt: Opts
    ) -> None:
        student_model = get_deit(opt)
        super().__init__(opt, student_model)
        self.distill_loss_fn = DistillationLoss(
            distillation_type='soft',
            tau=opt.get('distillation_tau', 1.0)
        )

    def train_teacher_step(
        self, *item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        images, labels = item
        with torch.no_grad():
            logits, rep = self.teacher_model(images, return_rep=True)
            loss = self.actual_loss_fn(logits, labels)
        return loss, logits, rep

    def train_student_step(
        self, *item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        images, labels, teacher_rep = item

        self.student_optimizer.zero_grad()
        student_out = self.student_model(images)
        if isinstance(student_out, tuple):
            logits = student_out[0]
            dist_logits = student_out[1]
        else:
            logits = student_out
            dist_logits = student_out
        actual_loss = self.actual_loss_fn(logits, labels)
        distill_loss = self.distill_loss_fn(dist_logits, teacher_rep)
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

        teacher_loss, _, teacher_rep = self.train_teacher_step(
            images, labels
        )
        student_loss, student_logits = self.train_student_step(
            images, labels, teacher_rep
        )

        total_loss = teacher_loss + student_loss
        self.teacher_loss_meter.append(teacher_loss.detach())
        self.student_loss_meter.append(student_loss.detach())

        self.show_images('train_image', images)
        preds = self.logit_to_pred(student_logits)
        return total_loss, preds, labels


@mod.register('arch')
class OnlineResNetDeitRepDistiller(OnlineDistillTrainer):
    def __init__(
        self, opt: Opts
    ) -> None:
        resnet_kws = {
            'pretrained': True,
            'progress': True
        }
        teacher_model = ResNet(opt, 'resnet50', resnet_kws)
        student_model = get_deit(opt)
        super().__init__(opt, teacher_model, student_model)
        self.distill_loss_fn = DistillationLoss(
            distillation_type='soft',
            tau=opt.get('distillation_tau', 1.0)
        )

    def train_teacher_step(
        self, *item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        images, labels = item

        self.teacher_optimizer.zero_grad()
        logits, rep = self.teacher_model(images, return_rep=True)
        loss = self.actual_loss_fn(logits, labels)
        loss.backward()
        self.teacher_optimizer.step()

        return loss.detach(), logits.detach(), rep.detach()

    def train_student_step(
        self, *item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        images, labels, teacher_rep = item

        self.student_optimizer.zero_grad()
        student_out = self.student_model(images)
        if isinstance(student_out, tuple):
            logits = student_out[0]
            dist_logits = student_out[1]
        else:
            logits = student_out
            dist_logits = student_out
        actual_loss = self.actual_loss_fn(logits, labels)
        distill_loss = self.distill_loss_fn(dist_logits, teacher_rep)
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

        teacher_loss, _, teacher_rep = self.train_teacher_step(
            images, labels
        )
        student_loss, student_logits = self.train_student_step(
            images, labels, teacher_rep
        )

        total_loss = teacher_loss + student_loss
        self.teacher_loss_meter.append(teacher_loss.detach())
        self.student_loss_meter.append(student_loss.detach())

        self.show_images('train_image', images)
        preds = self.logit_to_pred(student_logits)
        return total_loss, preds, labels
