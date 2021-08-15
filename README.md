
# IMRT Plan Generation Prototypes

## Folder structure

- **cuda**: CUDA implementation of Gradient Descent.
- **multicore**: MKL implementation of Gradient Descent.
- **fluences**: Sample starting fluences.
- **notebooks**: Jupyter notebooks for some preprocessing and analysis operations.
- **util**: Python scripts for analysis.

## Requirements

For the CUDA version, [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 10 or newer. For the Multicore version, [Intel Math Kernel Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.8gqn8f) 2019 or newer.

To compile, just run ``make``.

## Launch

```
./run.sh
```

On that script you can see the parameters used to call the program and a running example. For now, the CUDA version is the most updated. A full rewrite and feature parity with the Multicore version is on the works. 

## Modifying the initial fluence

The initial fluence is stored as a text file, with as many lines as beamlets. The number of beamlets has to match the model:

- Plan 3: 25298 beamlets.
- Plan 5: 30265 beamlets.

Each beamlet must be in the range [0, 1]. The files in the  `fluences` folder are examples of valid fluences for Plan 5.

