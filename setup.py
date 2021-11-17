from distutils.core import setup
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d%H%M%S')
    
    return f'dev{date_str}'


__version__ = f'0.0.1.{gen_code()}'


setup(name='kd_lab',
      version=__version__,
      description='INW Classification with SSL',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'torch',
            'einops',
            'numpy',
            'easydict',
            'timm',
            'ml-collections',
            'jupyterlab',
            'matplotlib',
      ],
      packages=['kd_lab',
                'kd_lab.dataset',
                'kd_lab.network',
                # classification networks
                'kd_lab.network.resnet',
                'kd_lab.network.vit',
                'kd_lab.network.iRPE',
                'kd_lab.network.iRPE.rpe_ops',
                # trainer
                'kd_lab.trainer',
                # classifier trainer
                'kd_lab.trainer.resnet',
                'kd_lab.trainer.vit',
                'kd_lab.trainer.iRPE',
                # dsitillation trainer
                'kd_lab.trainer.distill_resnet_iRPE',
                # utils
                'kd_lab.utils',
                'kd_lab.utils.data',
                'kd_lab.utils.distill'
      ]
)
