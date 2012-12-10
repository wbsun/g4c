# G4C submodule testing makefile.

CXX=g++
NVCC=nvcc
CXXFLAGS=-O2
NVCCFLAGS=-arch=sm_20 -O2

COMMSRCS=../g4c_mm.cc ../g4c.cu

EACSRCS=$(COMMSRCS) ../ac.cc ../ac_dev.cu evalac.cc
EACOBJS=$(addsuffix -eval.o, $(basename $(notdir $(EACSRCS))))

G4CDEPS=../g4c.cu ../g4c.hh ../g4c.h
G4CMMDEPS=../g4c_mm.cc ../g4c_mm.h ../g4c_mm.hh ../g4c.h
ACDEPS=../ac.cc ../ac.hh ../ac.h ../g4c.h
ACDEVDEPS=../ac_dev.cu ../ac.hh ../ac.h ../g4c.hh ../g4c.h
EACDEPS=evalac.cc ../g4c.h ../ac.h utils.h

ELUSRCS=$(COMMSRCS) ../lookup.cu evallu.cc
ELUOBJS=$(addsuffix -eval.o, $(basename ($notdir $(ELUSRCS))))
LUDEPS=../lookup.cu ../lookup.h ../g4c.h
ELUDEPS=evallu.cc ../g4c.h ../lookup.h utils.h

all: elu eac

elu: $(ELUOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

eac: $(EACOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

elu-eval.o: $(ELUDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

lookup-eval.o: $(LUDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

evalac-eval.o: $(EACDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

g4c_mm-eval.o: $(G4CMMDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

g4c-eval.o: $(G4CDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ac-eval.o: $(ACDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

ac_dev-eval.o: $(ACDEVDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

clean:
	rm -rf *.o
	rm -rf eac
	rm -rf elu
