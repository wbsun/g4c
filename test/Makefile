# G4C submodule testing makefile.

CXX=g++
NVCC=nvcc
CXXFLAGS=-O2
NVCCFLAGS=-arch=sm_20 -O2

EACSRCS=../g4c_mm.cc ../g4c.cu ../ac.cc ../ac_dev.cu evalac.cc
EACOBJS=$(addsuffix -eac.o, $(basename $(notdir $(EACSRCS))))

G4CDEPS=../g4c.cu ../g4c.hh ../g4c.h
G4CMMDEPS=../g4c_mm.cc ../g4c_mm.h ../g4c_mm.hh ../g4c.h
ACDEPS=../ac.cc ../ac.hh ../ac.h ../g4c.h
ACDEVDEPS=../ac_dev.cu ../ac.hh ../ac.h ../g4c.hh ../g4c.h
EACDEPS=evalac.cc ../g4c.h ../ac.h utils.h

eac: $(EACOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

evalac-eac.o: $(EACDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

g4c_mm-eac.o: $(G4CMMDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

g4c-eac.o: $(G4CDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ac-eac.o: $(ACDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

ac_dev-eac.o: $(ACDEVDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

clean:
	rm -rf *.o
	rm -rf eac
