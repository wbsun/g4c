# G4C Makefile
# Naive solution to deal with test and lib build.

CXX=g++
NVCC=nvcc
CXXFLAGS=-O2
CXXLIBFLAGS=-fPIC
NVCCFLAGS=-arch=sm_20 -O2
NVCCLIBFLAGS=--shared --compiler-options '-fPIC'

ACTESTFLAGS=-D_G4C_AC_TEST_
MMTESTFLAGS=-D_G4C_MM_TEST_

SRCS=g4c_mm.cc g4c.cu ac.cc ac_dev.cu lookup.cu
LIBG4COBJS=$(addsuffix -lib.o, $(basename $(SRCS)))
ACTESTOBJS=$(addsuffix -ac-test.o, $(basename $(SRCS)))
MMTESTOBJS=g4c_mm-mm-test.o

G4CDEPS=g4c.cu g4c.hh g4c.h
G4CMMDEPS=g4c_mm.cc g4c_mm.h g4c_mm.hh g4c.h
ACDEPS=ac.cc ac.hh g4c_ac.h g4c.h
ACDEVDEPS=ac_dev.cu ac.hh g4c_ac.h g4c.hh g4c.h
LPMDEPS=lookup.cu g4c_lookup.h g4c.hh g4c.h


all: libg4c

lpm: lookup-lpm-test.o

lookup-lpm-test.o: $(LPMDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

lookup-lib.o: $(LPMDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

lookup-ac-test.o:


libg4c: $(LIBG4COBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) $^ -o $@.so

g4c_mm-lib.o: $(G4CMMDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

g4c-lib.o: $(G4CDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

ac-lib.o: $(ACDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

ac_dev-lib.o: $(ACDEVDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@


ac-test: $(ACTESTOBJS)
	$(NVCC) $(NVCCFLAGS) $(ACTESTFLAGS) $^ -o $@

g4c_mm-ac-test.o: $(G4CMMDEPS)
	$(CXX) $(CXXFLAGS) $(ACTESTFLAGS) -c $< -o $@

g4c-ac-test.o: $(G4CDEPS)
	$(NVCC) $(NVCCFLAGS) $(ACTESTFLAGS) -c $< -o $@

ac-ac-test.o: $(ACDEPS)
	$(CXX) $(CXXFLAGS) $(ACTESTFLAGS) -c $< -o $@

ac_dev-ac-test.o: $(ACDEVDEPS)
	$(NVCC) $(NVCCFLAGS) $(ACTESTFLAGS) -c $< -o $@


mm-test: $(MMTESTOBJS)
	$(CXX) $(CXXFLAGS) $(MMTESTFLAGS) $^ -o $@

g4c_mm-mm-test.o: $(G4CMMDEPS)
	$(CXX) $(CXXFLAGS) $(MMTESTFLAGS) -c $< -o $@


install-lib: libg4c.so
	cp libg4c.so /usr/lib/
	cp g4c.h /usr/include/
	cp g4c_lookup.h /usr/include/
	cp g4c_ac.h /usr/include/

uninstall-lib:
	rm -f /usr/lib/libg4c.so
	rm -f /usr/include/g4c.h
	rm -f /usr/include/g4c_lookup.h
	rm -r /usr/include/g4c_ac.h


clean:
	rm -f *.o *.so
	rm -f ac-test
	rm -f mm-test


