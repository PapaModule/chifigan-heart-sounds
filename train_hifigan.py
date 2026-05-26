"""
train_hifigan.py
================
Conditional HiFi-GAN (cHiFi-GAN) dla dźwięków serca.

Kluczowe różnice vs WaveGAN:
  - Upsampling: nearest-neighbor + Conv1d zamiast ConvTranspose1d → brak aliasingu
  - Multi-Period Discriminator (MPD) — karze artefakty przy różnych periodycznościach
  - Multi-Scale Discriminator (MSD) — karze artefakty na różnych rozdzielczościach
  - Feature matching loss — stabilizuje trening bez WGAN-GP

Klasy: normal | murmur | extrastole
Hardware: PyTorch z automatycznym wyborem MPS / CUDA / CPU
"""

import os
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import soundfile as sf

# Reużyj datasetu z WaveGAN
from train_wavegan import HeartSoundDataset, make_weighted_sampler, get_device

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    DATASET_DIR    = Path("dataset")
    OUTPUT_DIR     = Path("output_hifigan")

    SAMPLE_RATE    = 4000
    AUDIO_LEN      = 12000           # 3 s × 4000 Hz; generator daje 12288, przycinane do 12000

    LATENT_DIM     = 128
    CLASS_EMB_DIM  = 64
    CLASSES        = ["normal", "murmur", "extrastole"]
    N_CLASSES      = len(CLASSES)

    # Generator
    G_INIT_LEN     = 12              # 12 × 1024 = 12288 → przycinane do 12000
    G_INIT_CH      = 256
    UPSAMPLE_RATES = [4, 4, 4, 4, 4]
    UPSAMPLE_CH    = [128, 64, 32, 16, 8]
    MRF_KERNELS    = [3, 7, 11]
    MRF_DILATIONS  = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    # Dyskryminatory
    MPD_PERIODS    = [2, 3, 5]

    # Dyskryminator spektrogramowy
    SD_N_FFT       = 256
    SD_HOP         = 64
    LAMBDA_SD      = 1.0              # waga strat SD względem MPD+MSD

    # Trening
    BATCH_SIZE     = 8
    N_EPOCHS       = 200
    LR_G           = 2e-4
    LR_D           = 1e-4
    BETA1          = 0.8
    BETA2          = 0.99
    LAMBDA_FM      = 2.0             # waga feature matching loss

    SAVE_EVERY     = 1
    SAMPLE_EVERY   = 5
    N_SAMPLES      = 4

    SEED           = 42


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class MRFBlock(nn.Module):
    """Multi-Receptive Field fusion — równoległe konwolucje różnych rozmiarów."""

    def __init__(self, channels, kernels, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for k, dils in zip(kernels, dilations):
            layer = nn.Sequential(*[
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.utils.weight_norm(nn.Conv1d(
                        channels, channels, k,
                        dilation=d, padding=k * d // 2
                    ))
                )
                for d in dils
            ])
            self.convs.append(layer)

    def forward(self, x):
        out = sum(conv(x) for conv in self.convs)
        return out / len(self.convs)

    def remove_weight_norm(self):
        for conv_seq in self.convs:
            for block in conv_seq:
                nn.utils.remove_weight_norm(block[1])


class HiFiGANGenerator(nn.Module):
    """
    (z, class_label) → audio [1, AUDIO_LEN]

    Upsampling: Upsample(nearest) + Conv1d — zero aliasingu.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.class_emb = nn.Embedding(cfg.N_CLASSES, cfg.CLASS_EMB_DIM)

        in_dim  = cfg.LATENT_DIM + cfg.CLASS_EMB_DIM
        out_dim = cfg.G_INIT_CH * cfg.G_INIT_LEN
        self.fc = nn.Linear(in_dim, out_dim)
        self.init_ch = cfg.G_INIT_CH

        self.ups   = nn.ModuleList()
        self.mrfs  = nn.ModuleList()

        ch = cfg.G_INIT_CH
        for i, (rate, out_ch) in enumerate(zip(cfg.UPSAMPLE_RATES, cfg.UPSAMPLE_CH)):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=rate, mode='nearest'),
                nn.utils.weight_norm(nn.Conv1d(
                    ch, out_ch,
                    kernel_size=rate * 2,
                    padding=rate // 2
                )),
            ))
            self.mrfs.append(MRFBlock(out_ch, cfg.MRF_KERNELS, cfg.MRF_DILATIONS))
            ch = out_ch

        self.post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.utils.weight_norm(nn.Conv1d(ch, 1, kernel_size=7, padding=3)),
            nn.Tanh(),
        )

    def forward(self, z, label):
        emb = self.class_emb(label)
        x   = torch.cat([z, emb], dim=-1)
        x   = self.fc(x).view(x.size(0), self.init_ch, self.cfg.G_INIT_LEN)

        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)

        x = self.post(x)
        # Przytnij lub dopełnij do dokładnej długości AUDIO_LEN
        if x.size(-1) > self.cfg.AUDIO_LEN:
            x = x[..., :self.cfg.AUDIO_LEN]
        elif x.size(-1) < self.cfg.AUDIO_LEN:
            x = F.pad(x, (0, self.cfg.AUDIO_LEN - x.size(-1)))
        return x

    def remove_weight_norm(self):
        for up in self.ups:
            nn.utils.remove_weight_norm(up[1])
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        nn.utils.remove_weight_norm(self.post[1])


# ─────────────────────────────────────────────────────────────────────────────
# DYSKRYMINATORY
# ─────────────────────────────────────────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """Jeden sub-dyskryminator MPD dla okresu p."""

    def __init__(self, period, class_emb_dim=16, n_classes=3):
        super().__init__()
        self.period = period
        self.class_emb = nn.Embedding(n_classes, class_emb_dim)

        ch = [1 + class_emb_dim, 16, 64, 256, 512, 512]
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(
                ch[i], ch[i+1],
                kernel_size=(5, 1),
                stride=(3, 1),
                padding=(2, 0)
            ))
            for i in range(len(ch) - 1)
        ])
        self.post = nn.utils.weight_norm(
            nn.Conv2d(512, 1, kernel_size=(3, 1), padding=(1, 0))
        )

    def forward(self, x, label):
        B, C, T = x.shape
        # Padding do wielokrotności okresu
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad), mode='replicate')
            T = T + pad
        x = x.view(B, C, T // self.period, self.period)  # [B, 1, T/p, p]

        # Dołącz embedding klasy jako dodatkowy kanał
        emb = self.class_emb(label)                        # [B, emb_dim]
        emb = emb.unsqueeze(-1).unsqueeze(-1)              # [B, emb_dim, 1, 1]
        emb = emb.expand(-1, -1, x.size(2), x.size(3))    # [B, emb_dim, T/p, p]
        x = torch.cat([x, emb], dim=1)

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, cfg.CLASS_EMB_DIM, cfg.N_CLASSES)
            for p in cfg.MPD_PERIODS
        ])

    def forward(self, x, label):
        outs, fmaps = [], []
        for d in self.discriminators:
            o, f = d(x, label)
            outs.append(o)
            fmaps.extend(f)
        return outs, fmaps


class ScaleDiscriminator(nn.Module):
    """Jeden sub-dyskryminator MSD."""

    def __init__(self, use_spectral_norm=False, class_emb_dim=16, n_classes=3):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.class_emb = nn.Embedding(n_classes, class_emb_dim)

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1 + class_emb_dim, 64, 15, 1, padding=7)),
            norm(nn.Conv1d(64, 64, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(64, 128, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 512, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(512, 512, 5, 1, padding=2)),
        ])
        self.post = norm(nn.Conv1d(512, 1, 3, 1, padding=1))

    def forward(self, x, label):
        emb = self.class_emb(label).unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat([x, emb], dim=1)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=False,
                               class_emb_dim=cfg.CLASS_EMB_DIM,
                               n_classes=cfg.N_CLASSES)
            for i in range(3)
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x, label):
        outs, fmaps = [], []
        for pool, d in zip(self.pools, self.discriminators):
            xp = pool(x)
            o, f = d(xp, label)
            outs.append(o)
            fmaps.extend(f)
        return outs, fmaps


# ─────────────────────────────────────────────────────────────────────────────
# DYSKRYMINATOR SPEKTROGRAMOWY
# ─────────────────────────────────────────────────────────────────────────────

class SpectrogramDiscriminator(nn.Module):
    """
    Dyskryminator operujący na log-STFT magnitude.
    Każda próbka oceniana indywidualnie — daje gradient o strukturze spektralnej.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_fft      = cfg.SD_N_FFT
        self.hop_length = cfg.SD_HOP
        self.class_emb  = nn.Embedding(cfg.N_CLASSES, cfg.CLASS_EMB_DIM)

        in_ch = 1 + cfg.CLASS_EMB_DIM
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(in_ch, 32,  (3, 3), stride=(1, 1), padding=1)),
            nn.utils.weight_norm(nn.Conv2d(32,    64,  (3, 3), stride=(2, 2), padding=1)),
            nn.utils.weight_norm(nn.Conv2d(64,    128, (3, 3), stride=(2, 2), padding=1)),
            nn.utils.weight_norm(nn.Conv2d(128,   256, (3, 3), stride=(2, 2), padding=1)),
        ])
        self.post = nn.utils.weight_norm(nn.Conv2d(256, 1, (3, 3), padding=1))

    def compute_spec(self, audio):
        # audio: [B, 1, T] → log-magnitude STFT [B, 1, freq, time]
        x = audio.squeeze(1)
        window = torch.hann_window(self.n_fft, device=audio.device)
        stft = torch.stft(x, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = torch.abs(stft)                  # [B, freq, time]
        mag = torch.log(mag + 1e-5)
        return mag.unsqueeze(1)                 # [B, 1, freq, time]

    def forward(self, x, label):
        spec = self.compute_spec(x)             # [B, 1, freq, time]
        emb  = self.class_emb(label)            # [B, emb_dim]
        emb  = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spec.shape[2], spec.shape[3])
        x    = torch.cat([spec, emb], dim=1)    # [B, 1+emb_dim, freq, time]

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


# ─────────────────────────────────────────────────────────────────────────────
# FUNKCJE STRAT
# ─────────────────────────────────────────────────────────────────────────────

def discriminator_loss(real_outs, fake_outs):
    loss = 0
    for r, f in zip(real_outs, fake_outs):
        loss += torch.mean((r - 1) ** 2) + torch.mean(f ** 2)
    return loss


def generator_adv_loss(fake_outs):
    loss = 0
    for f in fake_outs:
        loss += torch.mean((f - 1) ** 2)
    return loss


def feature_matching_loss(real_fmaps, fake_fmaps):
    loss = 0
    for r, f in zip(real_fmaps, fake_fmaps):
        loss += F.l1_loss(f, r.detach())
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# GENEROWANIE PRÓBEK
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(G, cfg, device, epoch, out_dir):
    G.eval()
    sample_dir = out_dir / f"samples/epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for cls_idx, cls_name in enumerate(cfg.CLASSES):
            for i in range(cfg.N_SAMPLES):
                z     = torch.randn(1, cfg.LATENT_DIM, device=device)
                label = torch.tensor([cls_idx], device=device)
                audio = G(z, label).squeeze().cpu().numpy()
                sf.write(str(sample_dir / f"{cls_name}_{i+1:02d}.wav"),
                         audio, cfg.SAMPLE_RATE, subtype="PCM_16")
    G.train()
    print(f"    Próbki zapisane → {sample_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# TRENING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    device = get_device()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset
    print("\n[1/4] Ładowanie datasetu...")
    dataset = HeartSoundDataset(cfg.DATASET_DIR, cfg.CLASSES, cfg.AUDIO_LEN)
    sampler = make_weighted_sampler(dataset)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                         sampler=sampler, num_workers=0, drop_last=True)
    print(f"  Batche na epokę: {len(loader)}")

    # Modele
    print("\n[2/4] Budowanie modeli...")
    G   = HiFiGANGenerator(cfg).to(device)
    MPD = MultiPeriodDiscriminator(cfg).to(device)
    MSD = MultiScaleDiscriminator(cfg).to(device)
    SD  = SpectrogramDiscriminator(cfg).to(device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in list(MPD.parameters()) + list(MSD.parameters()) + list(SD.parameters()))
    print(f"  Generator:          {g_params:,} parametrów")
    print(f"  Dyskryminatory:     {d_params:,} parametrów")

    opt_G = optim.AdamW(G.parameters(),
                        lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    opt_D = optim.AdamW(list(MPD.parameters()) + list(MSD.parameters()) + list(SD.parameters()),
                        lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    # Checkpoint
    ckpt_path   = cfg.OUTPUT_DIR / "checkpoint_latest.pt"
    start_epoch = 1
    if ckpt_path.exists():
        print(f"\n  Wczytuję checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        MPD.load_state_dict(ckpt["MPD"])
        MSD.load_state_dict(ckpt["MSD"])
        SD.load_state_dict(ckpt["SD"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Wznawiam od epoki {start_epoch}")

    print(f"\n[3/4] Trening — {cfg.N_EPOCHS} epok, batch={cfg.BATCH_SIZE}")
    print("      Ctrl+C zatrzymuje trening i zapisuje checkpoint.\n")

    G.train(); MPD.train(); MSD.train(); SD.train()

    try:
        for epoch in range(start_epoch, cfg.N_EPOCHS + 1):
            t0 = time.time()
            loss_D_sum = loss_G_sum = loss_adv_sum = loss_fm_sum = loss_sd_sum = 0.0

            for batch_idx, (real_audio, labels) in enumerate(loader):
                if batch_idx % 100 == 0:
                    print(f"    batch {batch_idx}/{len(loader)}", flush=True)
                real_audio = real_audio.to(device)   # [B, 1, AUDIO_LEN]
                labels     = labels.to(device)

                z          = torch.randn(real_audio.size(0), cfg.LATENT_DIM, device=device)
                fake_audio = G(z, labels).detach()

                # ── Krok dyskryminatora ──
                opt_D.zero_grad()

                real_mpd_out, real_mpd_fm = MPD(real_audio, labels)
                fake_mpd_out, _           = MPD(fake_audio, labels)
                real_msd_out, real_msd_fm = MSD(real_audio, labels)
                fake_msd_out, _           = MSD(fake_audio, labels)
                real_sd_out,  real_sd_fm  = SD(real_audio, labels)
                fake_sd_out,  _           = SD(fake_audio, labels)

                loss_D = (discriminator_loss(real_mpd_out, fake_mpd_out) +
                          discriminator_loss(real_msd_out, fake_msd_out) +
                          cfg.LAMBDA_SD * discriminator_loss(real_sd_out, fake_sd_out))
                loss_D.backward()
                opt_D.step()

                # ── Krok generatora ──
                opt_G.zero_grad()

                fake_audio = G(z, labels)
                fake_mpd_out, fake_mpd_fm = MPD(fake_audio, labels)
                fake_msd_out, fake_msd_fm = MSD(fake_audio, labels)
                fake_sd_out,  fake_sd_fm  = SD(fake_audio, labels)

                loss_adv = (generator_adv_loss(fake_mpd_out) +
                            generator_adv_loss(fake_msd_out) +
                            cfg.LAMBDA_SD * generator_adv_loss(fake_sd_out))
                loss_fm  = (feature_matching_loss(real_mpd_fm, fake_mpd_fm) +
                            feature_matching_loss(real_msd_fm, fake_msd_fm) +
                            cfg.LAMBDA_SD * feature_matching_loss(real_sd_fm, fake_sd_fm))
                loss_G   = loss_adv + cfg.LAMBDA_FM * loss_fm
                loss_G.backward()
                opt_G.step()

                loss_D_sum   += loss_D.item()
                loss_G_sum   += loss_G.item()
                loss_adv_sum += loss_adv.item()
                loss_fm_sum  += loss_fm.item()
                loss_sd_sum  += (fake_sd_out.mean() - real_sd_out.mean()).abs().item()
                if device.type == 'mps':
                    torch.mps.empty_cache()

            elapsed = int(time.time() - t0)
            avg_D   = loss_D_sum   / len(loader)
            avg_G   = loss_G_sum   / len(loader)
            avg_adv = loss_adv_sum / len(loader)
            avg_fm  = loss_fm_sum  / len(loader)
            avg_sd  = loss_sd_sum  / len(loader)
            print(f"  Epoka {epoch:4d}/{cfg.N_EPOCHS}"
                  f"  loss_D={avg_D:+.4f}"
                  f"  loss_G={avg_G:+.4f}"
                  f"  adv={avg_adv:+.4f}"
                  f"  fm={avg_fm:+.4f}"
                  f"  sd={avg_sd:+.4f}"
                  f"  ({elapsed}s)")

            if epoch in (1, 2) or epoch % cfg.SAMPLE_EVERY == 0:
                generate_samples(G, cfg, device, epoch, cfg.OUTPUT_DIR)

            # Checkpoint z rotacją backupu
            ckpt = {
                "epoch": epoch,
                "G":     G.state_dict(),
                "MPD":   MPD.state_dict(),
                "MSD":   MSD.state_dict(),
                "SD":    SD.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }
            prev_ckpt   = cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch-1:04d}.pt"
            prev_backup = cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch-1:04d}.pt.backup"
            old_backup  = cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch-2:04d}.pt.backup"
            if old_backup.exists():
                old_backup.unlink()
            if prev_ckpt.exists():
                prev_ckpt.rename(prev_backup)
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch:04d}.pt")
            print(f"    Checkpoint zapisany.")

    except KeyboardInterrupt:
        print("\n  Przerwano — zapisuję checkpoint...")
        torch.save({
            "epoch": epoch,
            "G":     G.state_dict(),
            "MPD":   MPD.state_dict(),
            "MSD":   MSD.state_dict(),
            "SD":    SD.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        }, ckpt_path)
        print("  Checkpoint zapisany.")

    # Finalny generator
    print("\n[4/4] Generowanie finalnych próbek...")
    generate_samples(G, cfg, device, cfg.N_EPOCHS, cfg.OUTPUT_DIR)
    torch.save(G.state_dict(), cfg.OUTPUT_DIR / "generator_final.pt")
    print(f"\nGotowe! Wyniki w: {cfg.OUTPUT_DIR.resolve()}")


def generate_dataset(n_per_class: int = 500, checkpoint: str = None):
    cfg    = Config()
    device = get_device()
    G = HiFiGANGenerator(cfg).to(device)
    ckpt_file = checkpoint or str(cfg.OUTPUT_DIR / "generator_final.pt")
    G.load_state_dict(torch.load(ckpt_file, map_location=device))
    G.eval()

    out_dir = Path("synthetic_dataset_hifigan")
    print(f"Generuję {n_per_class} próbek na klasę → {out_dir}/")

    with torch.no_grad():
        for cls_idx, cls_name in enumerate(cfg.CLASSES):
            cls_dir = out_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                z     = torch.randn(1, cfg.LATENT_DIM, device=device)
                label = torch.tensor([cls_idx], device=device)
                audio = G(z, label).squeeze().cpu().numpy()
                sf.write(str(cls_dir / f"{cls_name}_synth_{i+1:04d}.wav"),
                         audio, cfg.SAMPLE_RATE, subtype="PCM_16")
            print(f"  {cls_name}: {n_per_class}/{n_per_class}")
    print("Gotowe!")


if __name__ == "__main__":
    train(Config())
