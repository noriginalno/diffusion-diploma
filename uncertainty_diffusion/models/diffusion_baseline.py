from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from diffusers import DDPMScheduler
from diffusers.models.embeddings import get_timestep_embedding


class ConditionalLayer(nn.Module):
    def __init__(self, num_in, num_out, time_emb_size, cond_emb_size, max_period):
        super(ConditionalLayer, self).__init__()
        self.num_in = num_in
        self.time_emb_size = time_emb_size
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.ReLU(),
            nn.Linear(time_emb_size, time_emb_size),
        )
        self.cond_emb_size = cond_emb_size
        self.cond_enc = nn.Sequential(
            nn.LazyLinear(cond_emb_size),
            nn.BatchNorm1d(cond_emb_size),
            nn.ReLU(),
            nn.Linear(cond_emb_size, cond_emb_size),
            nn.BatchNorm1d(cond_emb_size),
            nn.ReLU(),
            nn.Linear(cond_emb_size, cond_emb_size),
        )
        self.num_out = num_out
        self.lin = nn.Sequential(
            nn.Linear(num_in + time_emb_size + cond_emb_size, num_out),
            nn.BatchNorm1d(num_out),
            nn.ReLU(),
            nn.Linear(num_out, num_out),
            nn.BatchNorm1d(num_out),
            nn.ReLU(),

        )
        self.max_period = max_period

    def forward(self, x, y, t):
        y_enc = self.cond_enc(y)
        time_enc = get_timestep_embedding(t, self.time_emb_size, max_period=self.max_period)
        time_enc = self.time_emb(time_enc)
        out = self.lin(torch.concat([x, y_enc, time_enc], dim=-1))
        return out


class LastLayer(ConditionalLayer):
    def __init__(self, num_in, num_out, time_emb_size, cond_emb_size, max_period):
        super(LastLayer, self).__init__(num_in, num_out, time_emb_size, cond_emb_size, max_period)
        self.lin = nn.Linear(num_in + time_emb_size + cond_emb_size, num_out)


class BaseDiffusionBody(nn.Module):
    def __init__(
            self, max_period, num_features=10,
            n_hidden=100, depth=1,
            time_emb_size=10, cond_emb_size=10,
    ):
        super(BaseDiffusionBody, self).__init__()
        self.max_period = max_period
        self.num_features = num_features
        self.cond_emb_size = cond_emb_size
        self.time_emb_size = time_emb_size
        layers = [ConditionalLayer(num_features, n_hidden, time_emb_size, cond_emb_size, max_period)]
        for i in range(depth):
            layers.append(ConditionalLayer(n_hidden, n_hidden, time_emb_size, cond_emb_size, max_period))
        self.layers = nn.ModuleList(layers)
        self.out = LastLayer(n_hidden, num_features, time_emb_size, cond_emb_size, max_period)

    def forward(self, x, y, t):
        generator = enumerate(self.layers)
        i, layer = next(generator)
        x = layer(x, y, t)
        for i, layer in generator:
            x = x + layer(x, y, t)
        return self.out(x, y, t)


class BaseDiffusion(nn.Module):
    def __init__(
            self,
            num_features=10,
            max_period=100,
            step_scheduler=None,
            scheduler_kwargs=None,
            cond_emb_size=10,
            time_emb_size=10,
            body_kwargs=None,
            device=None,
            wandb_project=None
    ):
        """
        Basic class for diffusion.
        """
        super(BaseDiffusion, self).__init__()

        self.num_features = num_features
        if step_scheduler is None:
            step_scheduler = DDPMScheduler
        if scheduler_kwargs is None:
            scheduler_kwargs = {'clip_sample': False}

        self.step_scheduler = step_scheduler(num_train_timesteps=max_period, **scheduler_kwargs)

        if body_kwargs is None:
            body_kwargs = {'n_hidden': 100, 'depth': 2}
        body_kwargs['cond_emb_size'] = cond_emb_size
        body_kwargs['time_emb_size'] = time_emb_size
        body_kwargs['num_features'] = num_features

        self.body = BaseDiffusionBody(max_period=max_period, **body_kwargs)

        self.loss = nn.MSELoss()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.wandb_project = wandb_project

    def forward(self, X, y, t):
        """
        X: [bs, num_features]
        y: [bs]
        t: [bs]
        """
        return self.body(X, y, t)  # [bs, num_features]

    @torch.no_grad()
    def sample(self, y, show_progress=False):
        x = torch.randn(y.shape[0], self.num_features, dtype=torch.float, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).to(self.device)
        iterator = enumerate(self.step_scheduler.timesteps)
        iterator = tqdm(iterator) if show_progress else iterator
        for i, t in iterator:
            tt = t.expand(y.shape[0]).to(self.device)
            residual = self(x, y, tt)
            x = self.step_scheduler.step(residual, t, x).prev_sample
        return x

    @torch.no_grad()
    def _pred_list(self, y, show_progress=False):
        x = torch.randn(y.shape[0], self.num_features, dtype=torch.float, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).to(self.device)
        x_list = list()
        residual_list = list()
        iterator = enumerate(self.step_scheduler.timesteps)
        iterator = tqdm(iterator) if show_progress else iterator
        for i, t in iterator:
            tt = t.expand(y.shape[0]).to(self.device)
            residual = self(x, y, tt)
            step = self.step_scheduler.step(residual, t, x)
            x = step.prev_sample
            x_list.append(step.pred_original_sample)
        return x_list

    @torch.no_grad()
    def sample_best_steps(self, y, best_steps, return_last_step=False, show_progress=False):
        x = torch.randn(y.shape[0], self.num_features, dtype=torch.float, device=self.device)
        x_return = [torch.zeros_like(x) for _ in best_steps] if isinstance(best_steps, list) else torch.zeros_like(x)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).to(self.device)

        iterator = enumerate(self.step_scheduler.timesteps)
        iterator = tqdm(iterator) if show_progress else iterator

        max_timestep = self.step_scheduler.timesteps[0]
        for i, t in iterator:
            tt = t.expand(y.shape[0]).to(self.device)
            residual = self(x, y, tt)
            step = self.step_scheduler.step(residual, t, x)
            x = step.prev_sample

            if isinstance(x_return, list):
                for i, best_step in enumerate(best_steps):
                    step_condition = (best_step == max_timestep - t)
                    x_return[i][step_condition] = step.pred_original_sample[step_condition]
            else:
                step_condition = (best_step == max_timestep - t)
                x_return[step_condition] = step.pred_original_sample[step_condition]

        if return_last_step == True:
            return x_return, x
        return x_return

    def configure_optimizer(self, learning_rate=1e-3, optimizer=None, lr_scheduler=None):
        self.optimizer = (
            torch.optim.AdamW(self.parameters(), lr=learning_rate)
            if optimizer is None else optimizer
        )
        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    500, 700, 1000, 1500,
                    2000, 3000, 4000, 5000,
                    7000, 10000, 13000, 16000
                ],
                gamma=0.15
            )
            if optimizer is None else optimizer
        )

    def fit(self, X, y, learning_rate=1e-3,
            epochs=10, batch_size=32):
        """TODO"""
        self.configure_optimizer(learning_rate=learning_rate)
        # numpy to tensor
        X_real = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_cond = torch.tensor(y, dtype=torch.float32, device=self.device)

        # tensor to dataset
        dataset_real = TensorDataset(X_real, y_cond)

        # Turn on training
        self.train(True)

        self.loss_history = []

        if self.wandb_project is not None:
            wandb.login()
            wandb.init(project=self.wandb_project)
            wandb.watch(self)

        for epoch in trange(epochs):
            loss_epoch = 0

            for i, (real_batch, cond_batch) in enumerate(
                    DataLoader(dataset_real, batch_size=batch_size, shuffle=True)
            ):
                noise = torch.randn_like(real_batch)
                timesteps = torch.randint(0, self.step_scheduler.timesteps[0] + 1, [cond_batch.shape[0]]).long().to(
                    self.device)
                noisy_batch = self.step_scheduler.add_noise(real_batch, noise, timesteps)

                self.optimizer.zero_grad()
                noise_pred = self(noisy_batch, cond_batch, timesteps)
                gen_loss = self.loss(noise, noise_pred)
                gen_loss.backward()
                self.optimizer.step()

                loss_epoch += gen_loss

            self.lr_scheduler.step()
            loss = loss_epoch.detach().cpu() / (i + 1)
            self.loss_history.append(loss)

            wandb.log({'train_loss': loss})

        wandb.finish()
        # Turn off training
        self.train(False)
