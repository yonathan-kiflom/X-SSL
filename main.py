import argparse
import datetime
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
from torchvision.datasets import ImageFolder
import torchvision
import matplotlib.pyplot as plt

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


class CustomImageFolder(ImageFolder):
    """ImageFolder that also passes the image name to the transform."""

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image, Path(path).stem)
        return image


class DataAugmentationDINO(object):
    """
    Generates two global crops from the original image and a local crop loaded
    from the cropped images directory.
    """

    def __init__(self, global_crops_scale: Sequence[float], cropped_images_dir: str):
        self.cropped_images_dir = cropped_images_dir

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.local_transfo = transforms.Compose([
            transforms.Resize((96, 96), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image, image_name: str):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        try:
            local_crop_path = glob.glob(os.path.join(self.cropped_images_dir, "*", f"{image_name}_*.png"))[0]
        except IndexError:
            local_crop_path = glob.glob(os.path.join(self.cropped_images_dir, "*", f"{image_name}_*.jpg"))[0]
        try:
            local_crop = Image.open(local_crop_path).convert("RGB")
            crops.append(self.local_transfo(local_crop))
        except FileNotFoundError:
            print(f"Local crop file not found: {local_crop_path}")
        return crops


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        ncrops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        nepochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch: int):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch: int,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}/{args.epochs}]"
    for it, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_dino_backbones(args):
    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load("facebookresearch/xcit:main", args.arch, pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load("facebookresearch/xcit:main", args.arch, pretrained=False)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    return student, teacher, embed_dim


def train_dino(args):
    utils.fix_random_seeds(args.seed)
    print(f"git:\n  {utils.get_sha()}\n")
    print("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    cudnn.benchmark = True

    transform = DataAugmentationDINO(args.global_crops_scale, args.cropped_data_path)
    dataset = CustomImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    student, teacher, embed_dim = build_dino_backbones(args)
    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    if args.pretrained_checkpoint:
        checkpoint = torch.load(args.pretrained_checkpoint)
        teacher.backbone.load_state_dict(checkpoint, strict=False)
        student.backbone.load_state_dict(checkpoint, strict=False)

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training!")
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def list_missing_crops(data_path: str, cropped_data_path: str) -> List[str]:
    data_list = glob.glob(os.path.join(data_path, "*", "*"))
    crop_list = glob.glob(os.path.join(cropped_data_path, "*", "*"))

    data_list_without_path = [Path(d).stem for d in data_list]
    crop_list_without_path = [Path(d).stem for d in crop_list]

    not_there_in_crop: List[str] = []
    for data in data_list_without_path:
        is_part_of_string = any(data in s for s in crop_list_without_path)
        if not is_part_of_string:
            print(data)
            not_there_in_crop.append(data)
    return not_there_in_crop


def load_teacher_for_attention(
    checkpoint_path: str,
    arch: str,
    patch_size: int,
    out_dim: int,
    use_bn_in_head: bool,
    drop_path_rate: float,
    device: torch.device,
):
    teacher_backbone = vits.__dict__[arch.replace("deit", "vit")](patch_size=patch_size, drop_path_rate=drop_path_rate)
    embed_dim = teacher_backbone.embed_dim
    teacher = utils.MultiCropWrapper(
        teacher_backbone,
        DINOHead(embed_dim, out_dim, use_bn=use_bn_in_head),
    )
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["teacher"] if isinstance(state, dict) and "teacher" in state else state
    teacher.load_state_dict(state_dict, strict=False)
    teacher.to(device)
    teacher.eval()
    return teacher


def generate_attention_maps(
    teacher,
    image_path: str,
    output_dir: str,
    threshold: float,
    patch_size: int,
    device: torch.device,
):
    model = teacher.backbone
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = Image.open(image_path).convert("RGB")
    img = transform(image).unsqueeze(0)

    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :, :w, :h]
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if threshold is not None:
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    else:
        th_attn = None

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(output_dir, "img.png"),
    )
    for j in range(nh):
        fname = os.path.join(output_dir, f"attn-head{j}.png")
        plt.imsave(fname=fname, arr=attentions[j], format="png")
        print(f"{fname} saved.")
    if th_attn is not None:
        for j in range(nh):
            fname = os.path.join(output_dir, f"attn-thresholded-head{j}.png")
            plt.imsave(fname=fname, arr=th_attn[j], format="png")
            print(f"{fname} saved.")


def build_parser():
    parser = argparse.ArgumentParser(description="DINO local-to-global training and utilities (cells 1-39).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DINO with local and global crops.")
    train_parser.add_argument("--data-path", required=True, help="Root folder with training images.")
    train_parser.add_argument("--cropped-data-path", required=True, help="Root folder with cropped local patches.")
    train_parser.add_argument("--output-dir", required=True, help="Where checkpoints and logs will be written.")
    train_parser.add_argument("--arch", default="vit_base", help="Backbone architecture.")
    train_parser.add_argument("--patch-size", type=int, default=16)
    train_parser.add_argument("--out-dim", type=int, default=128)
    train_parser.add_argument("--use-bn-in-head", action=argparse.BooleanOptionalAction, default=False)
    train_parser.add_argument("--norm-last-layer", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--momentum-teacher", type=float, default=0.996)
    train_parser.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    train_parser.add_argument("--teacher-temp", type=float, default=0.04)
    train_parser.add_argument("--warmup-teacher-temp-epochs", type=int, default=0)
    train_parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--weight-decay", type=float, default=0.04)
    train_parser.add_argument("--weight-decay-end", type=float, default=0.4)
    train_parser.add_argument("--clip-grad", type=float, default=3.0)
    train_parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--freeze-last-layer", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=0.0005)
    train_parser.add_argument("--min-lr", type=float, default=1e-6)
    train_parser.add_argument("--warmup-epochs", type=int, default=1)
    train_parser.add_argument("--optimizer", choices=["adamw", "sgd", "lars"], default="adamw")
    train_parser.add_argument("--drop-path-rate", type=float, default=0.1)
    train_parser.add_argument("--global-crops-scale", type=float, nargs=2, default=(0.8, 1.0))
    train_parser.add_argument("--local-crops-number", type=int, default=1)
    train_parser.add_argument("--local-crops-scale", type=float, nargs=2, default=(0.05, 0.4))
    train_parser.add_argument("--saveckp-freq", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--num-workers", type=int, default=1)
    train_parser.add_argument("--dist-url", default="env://")
    train_parser.add_argument("--local-rank", type=int, default=0)
    train_parser.add_argument("--gpu", type=int, default=0)
    train_parser.add_argument("--pretrained-checkpoint", default="./backbone/dino_vitbase16_pretrain.pth")

    crops_parser = subparsers.add_parser("check-crops", help="Report images missing cropped counterparts.")
    crops_parser.add_argument("--data-path", required=True)
    crops_parser.add_argument("--cropped-data-path", required=True)

    attn_parser = subparsers.add_parser("attention", help="Generate attention maps for a single image.")
    attn_parser.add_argument("--checkpoint", required=True, help="Checkpoint containing the teacher state.")
    attn_parser.add_argument("--image-path", required=True)
    attn_parser.add_argument("--output-dir", default="attnetion_maps_current")
    attn_parser.add_argument("--threshold", type=float, default=0.6)
    attn_parser.add_argument("--patch-size", type=int, default=16)
    attn_parser.add_argument("--arch", default="vit_base")
    attn_parser.add_argument("--out-dim", type=int, default=128)
    attn_parser.add_argument("--use-bn-in-head", action="store_true", default=False)
    attn_parser.add_argument("--drop-path-rate", type=float, default=0.1)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_dino(args)
    elif args.command == "check-crops":
        missing = list_missing_crops(args.data_path, args.cropped_data_path)
        print(f"{len(missing)} images have no cropped counterpart.")
    elif args.command == "attention":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = load_teacher_for_attention(
            checkpoint_path=args.checkpoint,
            arch=args.arch,
            patch_size=args.patch_size,
            out_dim=args.out_dim,
            use_bn_in_head=args.use_bn_in_head,
            drop_path_rate=args.drop_path_rate,
            device=device,
        )
        generate_attention_maps(
            teacher=teacher,
            image_path=args.image_path,
            output_dir=args.output_dir,
            threshold=args.threshold,
            patch_size=args.patch_size,
            device=device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
