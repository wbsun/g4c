
all: lib

lib: g4c.cu g4c.h
	nvcc -arch=sm_20 -O2 --shared --compiler-options '-fPIC' -o libg4c.so g4c.cu

ac.o: ac.cc ac.hh ac.h
	g++ -D_G4C_BUILD_AC_ -O2 -c ac.cc

ac: ac.o
	g++ -D_G4C_BUILD_AC_ -O2 ac.o -o ac

install-lib: lib
	sudo cp libg4c.so /usr/lib/
	sudo cp g4c.h /usr/include/

uninstall-lib:
	sudo rm -f /usr/lib/libg4c.so
	sudo rm -f /usr/include/g4c.h

mm-test.o: g4c_mm.hh g4c_mm.cc g4c.h
	g++ -D_G4C_TEST_MM_ -O2 -c g4c_mm.cc

mm-test: mm-test.o
	g++ -D_G4C_TEST_MM_ -O2 g4c_mm.o -o mmtest

clean-mm-test:
	rm -f g4c_mm.o
	rm -f mmtest

clean-ac:
	rm -f ac.o
	rm -f ac

clean: clean-ac clean-mm-test
	rm -f *.o *.so