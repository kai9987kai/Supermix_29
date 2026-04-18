
# Supermix

Supermix is the active monorepo for the Supermix desktop app, the Omni Collective training line, specialist model experiments, benchmark tooling, and local-first packaging.

It is intentionally a mixed workspace: source code, training scripts, build helpers, generated manifests, research outputs, and packaging metadata all live in the same repository so the full desktop + training workflow can be developed, tested, packaged, and published from one place.

## Latest release

**Supermix Studio X — V48 Frontier Edition** is the latest desktop release.

The large installer and model bundles are hosted on Hugging Face due to repository asset size constraints and Git LFS quotas. The desktop release page links to the installer, model directory, and release assets.

## What this repository contains

- `source/`  
  Active development workspace: model definitions, training scripts, dataset builders, benchmark runners, desktop/web app code, and release helpers.

- `runtime_python/`  
  A lighter packaged runtime path for quick local use.

- `datasets/`  
  Local conversation, coding, reasoning, science, and specialist training inputs.

- `output/`  
  Generated graphs, summaries, manifests, previews, logs, and Hugging Face upload staging.

- `installer/`  
  Inno Setup definitions and post-install notes for the desktop app.

- `dist/`  
  Locally built EXEs and installer outputs.

- `web_static/`  
  Lightweight browser-only metadata/static bundle.

## Main capabilities

- Multimodel desktop app with model selector, `Auto` routing, collective mode, and agent mode.
- Local chat, omni, image, vision, math, protein, 3D, and materials specialist model families.
- Curated desktop installer with built-in models plus one-click downloadable optional models.
- Benchmark sweeps and local graph generation for the model zoo.
- Hugging Face publishing helpers for models, installers, datasets, and downloadable model-store artifacts.
- Training pipelines for omni, frontier, native-image, and specialist lines.

## Current status

- The latest finished full omni checkpoint in this workspace is `omni_collective_v7`.
- `omni_collective_v8` remains the long-running local experiment.
- The desktop line uses a curated built-in bundle plus an in-app Hugging Face model store instead of shipping the full legacy model zoo.
- The most recent desktop release is the V48 Frontier Edition.

## Quick start

### Run the packaged runtime

```bash
python runtime_python/chat_web_app.py
````

Windows launchers:

```bat
runtime_python\launch_chat_web_supermix.bat
runtime_python\launch_chat_terminal_supermix.bat
```

### Run the active source web app

```bash
python source/chat_web_app.py
```

### Run the multimodel desktop app from source

```bash
python source/supermix_multimodel_desktop_app.py
```

### Run the training monitor

```bash
python source/training_monitor_gui.py --root .
```

### Open the browser-only static bundle

Open:

```text
web_static/index.html
```

## Desktop downloads

### Latest release assets

* Release page:
  `https://github.com/kai9987kai/Supermix/releases`
* Hugging Face desktop installer mirror:
  `https://huggingface.co/datasets/Kai9987kai/supermix-studio-desktop-installer`
* Model zoo:
  `https://huggingface.co/datasets/Kai9987kai/supermix-model-zoo`

### Local build outputs

* `dist/SupermixStudioDesktop/SupermixStudioDesktop.exe`
* `dist/installer/SupermixStudioDesktopSetup.exe`
* `dist/installer/SupermixStudioDesktopReleaseSHA256.txt`

## Curated bundle and model store

The desktop app ships with a curated core bundle and can also download optional models from the in-app model store.

The current workspace includes code and/or packaged artifacts for:

### Qwen adapter line

* `v28`
* `v30`

### Champion / frontier line

* `v31`
* `v32`
* `v33`
* `v34`
* `v35`
* `v39`
* `v40_benchmax`
* `v41`
* `v46`
* `v47`
* `v48`

### Native-image line

* `v36`
* `v37`
* `v38`

### Omni-collective line

* `v1`
* `v2`
* `v3`
* `v4`
* `v5`
* `v6`
* `v7`
* `v8_preview`
* `v8` in progress locally

### Specialist lines

* `math_equation_micro_v1`
* `science_vision_micro_v1`
* `protein_folding_micro_v1`
* `mattergen_micro_v1`
* `three_d_generation_micro_v1`
* `dcgan_mnist_model`
* `dcgan_v2_in_progress`

## Latest finished omni model

The latest finished full omni checkpoint in this workspace is `omni_collective_v7`.

The latest preview checkpoint is `omni_collective_v8_preview`.

`omni_collective_v8` is still training locally and is not yet the published stable omni release.

## Benchmarks

Representative current results from the local benchmark inventory:

* `v40_benchmax`: `0.2433` common benchmark score
* `omni_collective_v8_preview`: `0.2167` on the reduced preview benchmark sweep
* `omni_collective_v7`: `0.1067` on the current 6-benchmark local sweep

Specialist-only models remain in the graph inventory even when the common text benchmark is not the right evaluation fit.

## Training entry points

Representative training and continuation scripts:

```text
source/train_omni_collective_v2.py
source/train_omni_collective_v3.py
source/train_omni_collective_v4.py
source/train_omni_collective_v5.py
source/train_omni_collective_v6.py
source/train_omni_collective_v7.py
source/train_omni_collective_v8.py
source/train_math_equation_model.py
source/train_image_recognition_model.py
source/train_protein_folding_model.py
source/train_three_d_generation_model.py
source/train_mattergen_generation_model.py
source/benchmark_all_models_common.py
```

If you want the active experimental path, start in `source/`.

## Desktop build entry points

Primary desktop build helpers:

```text
source/build_supermix_studio_desktop_exe.ps1
source/build_supermix_studio_desktop_installer.ps1
SupermixStudioDesktop.spec
installer/SupermixStudioDesktop.iss
```

The current desktop app supports:

* curated built-in bundle seeding
* in-app Hugging Face model-store downloads
* richer chat UI with drafts, context bank, compare bench, and dispatch preview
* richer training monitor with recovery posture, rescue guidance, and fleet spotlight summaries

## Hugging Face publishing

Public Hugging Face repos used by this workspace include:

* `Kai9987kai/supermix-model-zoo`
* `Kai9987kai/supermix-studio-desktop-installer`

Previously published model repos from this workspace include:

* `Kai9987kai/supermix-v33-frontier`
* `Kai9987kai/supermix-omni-collective-v1`
* `Kai9987kai/supermix-v38-native-image-xlite-fp16`
* `Kai9987kai/supermix-v39-frontier-reasoning-plus`
* `Kai9987kai/supermix-omni-collective-v2-frontier`
* `Kai9987kai/supermix-math-equation-micro-v1`
* `Kai9987kai/supermix-omni-collective-v4-frontier`
* `Kai9987kai/supermix-omni-collective-v7-frontier`

Public dataset repos include:

* `Kai9987kai/supermix-conversation-datasets`
* `Kai9987kai/supermix-science-vision-dataset`

## Notes

* This repo is intentionally not a clean source-only model repo.
* Large local artifacts, checkpoints, and logs are often kept locally and mirrored selectively to releases or Hugging Face.
* The safest way to consume the desktop app is through the curated-core release plus the in-app model store.

## License

MIT

```


::contentReference[oaicite:1]{index=1}
```

[1]: https://github.com/kai9987kai/Supermix "GitHub - kai9987kai/Supermix: Supermix is the active monorepo for the current Supermix desktop app, the Omni Collective training line, specialist model experiments, benchmark tooling, and local-first packaging flow. · GitHub"
