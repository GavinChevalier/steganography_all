import os, sys

sys.path.append(os.getcwd())
import click
import time

import numpy as np
import torch
import torchvision
from training_utils import *
# to fix png loading
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@click.command()
@click.option('--output_path', default=r'F:\Gavin\0000\WGAN\wganoutput',
              help='Output path where result (.e.g drawing images, cost, chart) will be stored')
@click.option('--dim', default=64, help='Model dimensionality or image resolution, tested with 64.')
@click.option('--batch_size', default=64, help='Training batch size. Must be a multiple of number of gpus')


# def test(output_path, dim, batch_size):
#
#     output_path = Path(output_path)
#     sample_path = output_path / "samples_test"
#     mkdir_path(sample_path)
#
#     cuda_available = torch.cuda.is_available()
#     device = torch.device("cuda" if cuda_available else "cpu")
#     fixed_noise = gen_rand_noise(batch_size).to(device)
#
#     aG = GoodGenerator(dim, dim * dim * 3)
#     g_state_dict = torch.load(str(output_path / "generator.pt"))
#     aG.load_state_dict(remove_module_str_in_state_dict(g_state_dict))
#
#     aG = torch.nn.DataParallel(aG).to(device)
#
#     gen_images = generate_image(aG, dim=dim, batch_size=batch_size, noise=fixed_noise)
#     torchvision.utils.save_image(gen_images, str(sample_path / 'samples_{}.png').format('01'), nrow=8,
#                                          padding=2)


def test2(output_path, dim, batch_size):

    output_path = Path(output_path)
    sample_path = output_path / "samples_test"
    mkdir_path(sample_path)

    fixed_noise = gen_rand_noise(batch_size).cuda()

    aG = GoodGenerator(dim, dim * dim * 3)
    g_state_dict = torch.load(str(output_path / "generator.pt"))
    aG.load_state_dict(remove_module_str_in_state_dict(g_state_dict))

    aG = torch.nn.DataParallel(aG).cuda()

    gen_images = generate_image(aG, dim=dim, batch_size=batch_size, noise=fixed_noise)
    torchvision.utils.save_image(gen_images, str(sample_path / 'samples_{}.png').format('02'), nrow=8,
                                         padding=2)




if __name__ == '__main__':
    test2()




