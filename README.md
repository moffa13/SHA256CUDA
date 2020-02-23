# SHA256CUDA

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
