#!/bin/bash
source .venv/bin/activate

echo "1. Running baseline..."
python -m surface.generate --model baseline --size 128 --output outputs/model_baseline.png

echo "2. Running naive_bayes..."
python -m surface.generate --model naive_bayes --size 128 --output outputs/model_naive_bayes.png

echo "3. Running mrf (1 sweep for speed)..."
python -m surface.generate --model mrf --size 128 --sweeps 1 --output outputs/model_mrf.png

echo "4. Running gmm..."
python -m surface.generate --model gmm --size 128 --output outputs/model_gmm.png

echo "5. Running ensemble (1 sweep for speed)..."
python -m surface.generate --model ensemble --size 128 --sweeps 1 --output outputs/model_ensemble.png

echo "6. Running gan..."
python -m surface.generate --model gan --size 128 --load_model checkpoints/gan_150_epochs.pth --output outputs/model_gan.png

echo "7. Running gan512..."
python -m surface.generate --model gan512 --size 512 --load_model checkpoints/gan512_100_epochs.pth --output outputs/model_gan512.png

echo "8. Running unet_height..."
python -m surface.generate --model unet_height --size 128 --load_model checkpoints/unet_height10.pth --output outputs/model_unet_height.png

echo "All tests finished!"
