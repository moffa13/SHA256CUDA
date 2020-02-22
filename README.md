# SHA256CUDA

Because of std::cin ignoring spaces, including newlines, use getline(std::cin, in) instead.

To compile, rename main.cpp to main.cu, which nvcc recognizes.  Then compile:

`$ nvcc -o hash_program main.cu`

Example run:

```
james@acer-nitro5:~/src/cuda/SHA256CUDA$ ./hash_program 
Entrez un message : This is a test
Nonce : 0
Difficulte : 7

Shared memory is 16400B
772009 hash(es)/s
Nonce : 8388608
30226098This is a test
00000001adff67cab9570a236d8490c0f5efee91e0303562e2d95ff7c3b7f3ec
james@acer-nitro5:~/src/cuda/SHA256CUDA$
```

If you start it with a higher nonce, and give it more work to do, the hashrate will be higher.
On my system with a GTX 1050 GPU, I get 6.5 million hash/s.

