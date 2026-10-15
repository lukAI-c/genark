# GenArk Workspace Backup Manifest

This file records the source revisions and external artifacts required to
rebuild `/home/clk/genark_ws`. Large model and dataset files are intentionally
not committed to Git.

## Source repositories

| Directory | Repository | Revision |
| --- | --- | --- |
| `genark` | <https://github.com/lukAI-c/genark> | Use the latest `master` commit containing this file. |
| `Genesis` | <https://github.com/Genesis-Embodied-AI/Genesis> | Tag `v0.4.3`, commit `b7bb647`; then apply `patches/genesis-v0.4.3-local.patch`. |
| `Uni-NaVid` | <https://github.com/jzhzhang/Uni-NaVid> | Commit `79ef5ea`. |
| `NaVid-VLN-CE` | <https://github.com/jzhzhang/NaVid-VLN-CE> | Commit `ce2f804`. |

Restore the Genesis changes after cloning:

```bash
git -C Genesis checkout v0.4.3
git -C Genesis apply ../genark/patches/genesis-v0.4.3-local.patch
```

## Model artifacts

Download the Uni-NaVid checkpoint from:

<https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/uninavid-7b-full-224-video-fps-1-grid-2>

Recommended command:

```bash
hf download Jzzhang/Uni-NaVid \
  --include "uninavid-7b-full-224-video-fps-1-grid-2/*" \
  --local-dir NaVid-VLN-CE/model_zoo/
```

Download the EVA ViT-G checkpoint from:

<https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth>

Expected SHA-256 checksums:

```text
f187421f7cf15151de2a05075f1ceccd0ff4958ab61bf52f9b1c3152aab8b838  pytorch_model-00001-of-00002.bin
8e892b7e4697c9024c328ebf93c99113a829b0ca5f1318de0c77903c4f23d1ac  pytorch_model-00002-of-00002.bin
99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f  eva_vit_g.pth
```

## Navigation datasets

The local `data/datasets.tar.gz` is not stored in Git. Its SHA-256 is:

```text
94c3487d98f34b8dd0e66c10c0b61db529a0bf89c46b99ee584aa344be8a9391  datasets.tar.gz
```

Public source and setup instructions:

- VLN-CE R2R data: <https://github.com/jacobkrantz/VLN-CE#data>
- RxR data mirror referenced by NaVid-VLN-CE: <https://1drv.ms/u/c/aa19f644cf9d8afb/ETQ8Co-hGLFMjwd5HckKsvABjWvZ3cbPsWwdzbhmQDoL1g>
- Matterport3D scenes require accepting the Matterport3D dataset license; they cannot be redistributed through this repository.

The exact local dataset archive should be copied off the server before it is
reclaimed if byte-for-byte reproducibility is required.
