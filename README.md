# SHA256CUDA

This is a fork of the original designed for mining Fortuna.

1. Compile. Google to figure out how to get `nvcc` on your system.
`$ nvcc -o hash_program main.cu`

2. Run
`./hash_program` <last 20 hex encoded bytes of datum> <hex encoded nonce> <leading zeroes>`

Example:

```
./hash_program 1944115820000000000743b91d82f30188f55777051ae87aae1aa32dfe90d5564a0bfbf73e0919fffc1a03578b88ff 078f489ebdb8e90d69993e250566c204 5
> 000008a833f27d46323ec3750e9830e5822a394446f9441c6ce9c5a8bb5c8d64|078f489ee5b93d6469993e250566c204
```

## Important Notes:
1. Difficulty checking is not implemented! TODO.
2. There is no communication mechanism back to a host program; the host program must "shell out" to the compiled executable and parse the `<hash>|<nonce>` result from stdin.

Original readme below.

---


Because of std::cin ignoring spaces, including newlines, use getline(std::cin, in) instead.

`$ nvcc -o hash_program main.cu`

Example run:

```
james@acer-nitro5:~/src/cuda/SHA256CUDA$ ./hash_program 
Enter a message : This is a test
Nonce : 0
Difficulty : 7

Shared memory is 16400B
772009 hash(es)/s
Nonce : 8388608
30226098This is a test
00000001adff67cab9570a236d8490c0f5efee91e0303562e2d95ff7c3b7f3ec
james@acer-nitro5:~/src/cuda/SHA256CUDA$
```

On my system with a GTX 1050 GPU, I get 6.5-12 million hash/s.
On my GTX 970, I get 135 millions hashes/s in Release mode and max speed optimization (-O2)

With sha256d:

```
james@acer-nitro5:~/src/cuda/SHA256CUDA/SHA256CUDA$ ./hash_test 
Enter a message : Hello, world
Nonce : 0
Difficulty : 7

Shared memory is 16793600KB
608525 hash(es)/s
Nonce : 8388608
2926219 hash(es)/s
Nonce : 41943040
5081728 hash(es)/s
Nonce : 75497472
7091294 hash(es)/s
Nonce : 109051904
8966579 hash(es)/s
Nonce : 142606336
10721864 hash(es)/s
Nonce : 176160768
12366523 hash(es)/s
Nonce : 209715200
209884948Hello, world
0000000af41bfb840272cf865799484235d775c343f6a7f0435828e1b17b2ff4

james@acer-nitro5:~/src/cuda/SHA256CUDA/SHA256CUDA$ echo -n "209884948Hello, world" | sha256sum | cut -d' ' -f1 | xxd -r -p | sha256sum
0000000af41bfb840272cf865799484235d775c343f6a7f0435828e1b17b2ff4  -

james@acer-nitro5:~/src/cuda/SHA256CUDA/SHA256CUDA$
```
