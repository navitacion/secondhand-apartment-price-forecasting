# Second-Hand Apartment Price Forecasting

Nishika Competition

[Site](https://www.nishika.com/competitions/11/summary)

## Env

- WSL2

- CUDA 11.0

## Ready for Using

Build Docker Container

```
# Build Container
docker build -t my_env .
```

Run Container

```
# Run Container
docker run -it --rm --gpus all -v $(pwd):/workspace my_env bash
# Install Library on RapidsAI
sh setup.sh
```
