
all: lib main

lib: g4c.cu g4c.h
	nvcc -arch=sm_20 -O2 --shared --compiler-options '-fPIC' -o libg4c.so g4c.cu

main: lib main.c
	gcc -O2 -o main main.c -L./ -lg4c

clean-main:
	rm -f main

ac.o: ac.cc ac.hh ac.h
	g++ -D_G4C_BUILD_AC_ -O2 -c ac.cc

ac: ac.o
	g++ -D_G4C_BUILD_AC_ -O2 ac.o -o ac

clean-ac:
	rm -f ac.o

clean: clean-ac clean-main
	rm -f *.o *.so