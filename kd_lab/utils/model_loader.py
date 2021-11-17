import os
import copy
import torch
from cfg import Opts
from mlutils import Saver, Log
from mlutils.saver import load_pickle


class ModelManager:
    def __init__(
        self,
        opt: Opts,
        saver: Saver
    ) -> None:
        self.exp_id = opt.get('exp_id', None)
        self.saver = saver
        if self.exp_id is not None:
            self.target_model = opt.target_model
            self.root = os.path.join(*saver.saver_dir.split('/')[:-1], self.exp_id)

    def load_model(self):
        assert self.exp_id is not None
        raw_model_path = os.path.join(self.root, 'raw_model.pkl')
        model = load_pickle(raw_model_path)

        weights = self.load_weight()
        model.load_state_dict(weights)
        return model

    def load_weight(self):
        assert self.exp_id is not None
        assert self.target_model in ['best', 'latest'], f'{self.target_model} error.'
        weight_path = os.path.join(self.root, f'{self.target_model}_0.pth')
        state_dict = torch.load(weight_path)
        epoch = state_dict['epoch']
        Log.info(f'Load model from {epoch}_th epoch.')
        return state_dict['net']

    def save_model(self, model):
        model = copy.deepcopy(model)
        model = model.cpu()
        self.saver.save_object(model, 'raw_model.pkl')
