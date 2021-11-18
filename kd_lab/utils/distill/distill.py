from typing import Optional, Tuple, Any
from cfg import Opts
from torch.nn import functional as F
from torch import nn, Tensor
from mlutils import Trainer, AverageMeter
from cvutils import transform as tf
from kd_lab.utils.model_loader import ModelManager


class BaseTrainer(Trainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)
        self.model_manager = ModelManager(opt, self.saver)
        self.to_255 = tf.DeNormalize(
            mean=[189.47680262, 185.68525998, 177.09843632],
            std=[39.60534874, 39.47619922, 37.76661493])

    def show_images(self, title: str, images: Tensor) -> None:
        images = self.to_255(images)
        self.dashboard.add_image(title, images, rgb=True)



class BaseDistillTrainer(BaseTrainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)
        self.teacher_loss_meter = AverageMeter('teacher loss')
        self.student_loss_meter = AverageMeter('student loss')

    def setup_teacher(
        self,
        teacher_model: nn.Module
    ) -> nn.Module:
        raise NotImplementedError

    def train_teacher_step(
        self, item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def train_student_step(
        self, item: Tuple[Any]
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def logit_to_pred(self, logit):
        return F.softmax(logit, dim=1)
        # return pred.argmax(1)

    def on_epoch_begin(self):
        self.teacher_loss_meter.zero()
        self.student_loss_meter.zero()
        super().on_epoch_begin()

    def on_epoch_end(self):
        self.teacher_loss_meter.step()
        self.student_loss_meter.step()
        self.dashboard.add_trace_dict({
            'teacher_loss': self.teacher_loss_meter.avg,
            'student_loss': self.student_loss_meter.avg
        }, training=True)
        super().on_epoch_end()
