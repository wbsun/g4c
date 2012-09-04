
all: lib main

lib: g4c.cu g4c.h
	nvcc -arch=sm_20 -O2 --shared --compiler-options '-fPIC' -o libg4c.so g4c.cu

main: lib main.c
	gcc -O2 -o main main.c -L./ -lg4c

clean:
	rm -f *.o *.so main