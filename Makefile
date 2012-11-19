
all: lib

mm: g4c_mm.cc g4c_mm.h g4c_mm.hh
	g++ -O2 -fPIC -c g4c_mm.cc

mm-test.o: g4c_mm.hh g4c_mm.cc g4c.h
	g++ -D_G4C_TEST_MM_ -O2 -c g4c_mm.cc -o g4c_mm-test.o

mm-test: mm-test.o
	g++ -D_G4C_TEST_MM_ -O2 g4c_mm-test.o -o mmtest

clean-mm-test:
	rm -f g4c_mm-test.o
	rm -f mmtest

clean-mm:
	rm -f g4c_mm.o

ac-test:
	nvcc -arch=sm_20 -O2 g4c.cu -o g4c-ac-test.o
	g++ -O2 -c g4c_mm.cc -o g4c_mm-ac-test.o
	nvcc -arch=sm_20 -O2 --compiler-options '-D_G4C_BUILD_AC_' ac_dev.cu -o ac_dev-ac-test.o
	g++ -O2 -D_G4C_BUILD_AC_ -c ac.cc -o ac-test.o
	nvcc -arch=sm_20 -O2 --compiler-options '-D_G4C_BUILD_AC_' ac-test.o ac_dev-ac-test.o g4c-ac-test.o g4c_mm-ac-test.o -o ac-test


ac-test.o: ac.cc ac.hh ac.h ac_dev.cu g4c.hh g4c.h g4c.cu
	nvcc -arch=sm_20 -O2 -c ac_dev.cu -o ac_dev.o
	g++ -D_G4C_BUILD_AC_ -O2 -c ac.cc -o ac-test.o

ac-test: ac-test.o ac_dev.o
	nvcc -arch=sm_20 -D_G4C_BUILD_AC_ -O2 ac-test.o ac_dev.o -o ac-test

ac: ac.cc ac.hh ac.h
	g++ -O2 -fPIC -c ac.cc

clean-ac:
	rm -f ac.o

clean-ac-test:
	rm -f ac-test.o
	rm -f ac-test


clean: clean-ac-test clean-mm-test clean-mm clean-ac
	rm -f *.o *.so

lib: g4c.cu g4c.h mm ac
	nvcc -arch=sm_20 -O2 --shared --compiler-options '-fPIC' -o libg4c.so g4c.cu g4c_mm.o

install-lib: lib
	sudo cp libg4c.so /usr/lib/
	sudo cp g4c.h /usr/include/

uninstall-lib:
	sudo rm -f /usr/lib/libg4c.so
	sudo rm -f /usr/include/g4c.h
