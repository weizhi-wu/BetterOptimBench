from time import sleep
from typing import List

import numpy as np
import torch
from tqdm import tqdm


torch.manual_seed(69420)

torch.clear_autocast_cache()
torch.cuda.empty_cache()
torch.set_default_device('cuda')




def create_card(object: callable, options: dict = None) -> dict:
    if options is None:
        options = {}
    return {
        'object': object,
        'type': object.__bases__,
        'options': options,
    }




class Experiment:
    def __init__(
            self,
            test_card: dict,
            optimizer_card: dict,
            epochs: int,
            multi_run_optim: bool = False,
            experiment_name: str = None,
    ) -> None:

        test_function = test_card['object']
        optimizer = optimizer_card['object']
        test_options = test_card['options']
        optim_options = optimizer_card['options']

        self.test_instance = test_function(**test_options)
        self.optimizer = optimizer([self.test_instance.params], **optim_options)

        self.max_epochs = epochs

        self.timestamps = np.empty(self.max_epochs + 1)
        self.losses = np.empty(self.max_epochs + 1)

        self.timestamps[0] = 0
        self.losses[0] = self.test_instance.loss()

        self.multi_run_optim = multi_run_optim

        if experiment_name is None:
            experiment_name = f'{type(optimizer.__name__)} Optimization on {type(test_function.__name__)} Function'
        self.experiment_name = experiment_name


    @staticmethod
    def torch_to_numpy(tensor: any) -> any:
        return tensor.cpu().numpy()


    @staticmethod
    def train(optimizer: any, test: any) -> any:
        optimizer.zero_grad()
        loss = test.loss()
        loss.backward()
        step_loss = optimizer.step(closure=test.loss)
        return step_loss


    def run(self) -> None:
        print(f'Running {self.experiment_name}')
        sleep(0.01)

        for epoch in tqdm(range(self.max_epochs), desc='Training Progress'):
            if self.multi_run_optim and 'cumulative_runs' in getattr(self.optimizer, 'misc', {}):
                self.timestamps[epoch + 1] = self.optimizer.misc['cumulative_runs']
            else:
                self.timestamps[epoch + 1] = epoch + 1

            current_loss = self.train(self.optimizer, self.test_instance)
            self.losses[epoch + 1] = current_loss


    def plot(self):
        from toolkit_graph import plot_loss
        plot_loss(self)




class CompareOptimizers:
    def __init__(
            self,
            test_card: dict,
            optimizer_cards: List[dict],
            epochs: int
    ):
        self.collection = []
        for i in range(len(optimizer_cards)):
            self.collection.append(Experiment(test_card, optimizer_cards[i], epochs))


    def run(self):
        for i in range(len(self.collection)):
            self.collection[i].run()


    def plot(self):
        from toolkit_graph import plot_loss
        plot_loss(self.collection)
