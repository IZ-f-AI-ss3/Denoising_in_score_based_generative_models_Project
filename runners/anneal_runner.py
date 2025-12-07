import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image

__all__ = ['AnnealRunner']


class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                     transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=2, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images


    # def custom_anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    #     images = []
    #     with torch.no_grad():
    #         for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='custom Langevin dynamics sampling'):
    #             labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
    #             labels = labels.long()
    #             # Calculate eps
    #             step_size = step_lr * (sigma / sigmas[-1]) ** 2
    #             for s in range(n_steps_each):
    #                 images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
    #                 noise = torch.randn_like(x_mod)
    #                 # score of current x_mod (which is xt tilde)
    #                 grad = scorenet(x_mod, labels)
    #                 # xt+1 = xt tild + eps/2 * score(xt tilde)
    #                 x_next = x_mod + (step_size / 2) * grad
    #                 # xt + sqt(eps) * zt
    #                 x_noise_component = x_next + np.sqrt(step_size) * noise
    #                 x_mod = torch.max(x_mod, x_noise_component)
    #         return images

    def half_denoising_anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='half-denoising Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))

                    noise = torch.randn_like(x_mod)
                    # x_tilde = xt + sigma * zt
                    x_tilde = x_mod + np.sqrt(step_size) * noise
                    # x_tilde = x_mod + sigma * noise

                    # score on x_tilde
                    grad = scorenet(x_tilde, labels)

                    # xt+1 = x_tilde + (sigma^2 / 2) * score(x_tilde)
                    x_mod = x_tilde + (step_size / 2) * grad
                    # x_mod = x_tilde + (sigma ** 2 / 2) * grad

            return images

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])


        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        grid_size = 5
        sampling_method = self.args.sampling_type

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sub_folder = os.path.join(self.args.image_folder,sampling_method)
        os.makedirs(sub_folder , exist_ok=True)

        imgs = []
        if self.config.data.dataset == 'MNIST':
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
            # Using custom_anneal_Langevin_dynamics for MNIST 
            if sampling_method == 'ordinary':
                all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
            elif sampling_method == 'half_denoising' :
                all_samples = self.half_denoising_anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
            else:
                raise ValueError("You can only choose among ordinary and half_denoising methods")

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(sub_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(sub_folder, 'image_raw_{}.pth'.format(i)))

        else:
            samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

            if sampling_method == 'ordinary':
                all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
            elif sampling_method == 'half_denoising' :
                all_samples = self.half_denoising_anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002) 
            else:
                raise ValueError("You can only choose among ordinary and half_denoising methods")

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(sub_folder, 'image_{}.png'.format(i)), nrow=10)
                torch.save(sample, os.path.join(sub_folder, 'image_raw_{}.pth'.format(i)))

        imgs[0].save(os.path.join(sub_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)


    def batched_test(self):
        load_path = os.path.join(self.args.log, 'checkpoint.pth')
        print(f"DEBUG: Looking for checkpoint at: {load_path}") 
        print(f"DEBUG: Does it exist? {os.path.exists(load_path)}") 
        
        states = torch.load(load_path, map_location=self.config.device)

        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])


        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()

        n_samples = self.args.n_samples
        batch_size = 100

        sampling_method = self.args.sampling_type

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sub_folder = os.path.join(self.args.image_folder,sampling_method)
        os.makedirs(sub_folder , exist_ok=True)

        img_id = 0  

        for k in tqdm.tqdm(range(n_samples // batch_size)):

            if self.config.data.dataset == 'MNIST':
                samples = torch.rand(batch_size, 1, 28, 28, device=self.config.device)
            else:
                samples = torch.rand(batch_size, 3, 32, 32, device=self.config.device)

            if sampling_method == 'ordinary':
                all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
            elif sampling_method == 'half_denoising' :
                all_samples = self.half_denoising_anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002) 
            else:
                raise ValueError("You can only choose among ordinary and half_denoising methods")
            
            final_batch = all_samples[-1] # only the final image
            for i in range(batch_size):
                sample = final_batch[i]

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)
                
                save_image(sample, os.path.join(sub_folder, f'img_{img_id}.png'))
                img_id += 1
                # print(f"Saved batch {img_id} images to {sub_folder}. Ready for FID calculation.")
            print(f"Finished batch {k} % {n_samples // batch_size} ")


    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32 ,32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def half_denoising_anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32 ,32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="half-denoising annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = sigma 

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod)
                    x_tilde = x_mod + sigma * noise                    
                    x_tilde[:, :, :, :16] = half_refer_image + sigma * noise[:, :, :, :16]
                    grad = scorenet(x_tilde, labels)                    
                    x_mod = x_tilde + (sigma ** 2 / 2) * grad                    
                    x_mod[:, :, :, :16] = half_refer_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        imgs = []
        if self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=2)
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            # all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            all_samples = self.half_denoising_anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                # save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                # torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=2)
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            # all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            all_samples = self.half_denoising_anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))


        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
