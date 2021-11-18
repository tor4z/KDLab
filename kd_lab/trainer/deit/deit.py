from cfg import Opts
from mlutils import mod
from kd_lab.utils.distill import\
    OfflineDistillTrainer, OnlineDistillTrainer
from kd_lab.network.deit.deit import get_deit
from kd_lab.network.resnet import ResNet


__all__ = [
    'OfflineResNetDeitDistiller',
    'OnlineResNetDeitDistiller'
]


@mod.register('arch')
class OfflineResNetDeitDistiller(OfflineDistillTrainer):
    def __init__(
        self, opt: Opts
    ) -> None:
        student_model = get_deit(opt)
        super().__init__(opt, student_model)


@mod.register('arch')
class OnlineResNetDeitDistiller(OnlineDistillTrainer):
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
