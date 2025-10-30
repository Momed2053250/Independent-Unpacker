For specifying which alpaka backend to use, use:
``` cmake -DUSE_CUDA=ON ```
``` cmake -DUSE_HIP=ON ```
Default one is set ti CPU/OpenMP backend if no option is selected the code falls back to this  
``` Confused where to find this ?```
Look for in the Unpacker.h file 
