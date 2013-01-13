# G4C submodule testing makefile.

CXX=g++
NVCC=nvcc
CXXFLAGS=-g
NVCCFLAGS=-arch=sm_20 -g

COMMSRCS=../g4c_mm.cc ../g4c.cu

EACSRCS=$(COMMSRCS) ../ac.cc ../ac_dev.cu evalac.cc
EACOBJS=$(addsuffix -eval.o, $(basename $(notdir $(EACSRCS))))

ECLSRCS = $(COMMSRCS) ../g4c_cl.cu evalcl.cc
ECLOBJS = $(addsuffix -eval.o, $(basename $(notdir $(ECLSRCS))))

CLDEPS = ../g4c_cl.cu ../g4c_cl.h ../g4c.h ../g4c.hh
ECLDEPS = evalcl.cc ../g4c.h ../g4c_cl.h utils.h

G4CDEPS=../g4c.cu ../g4c.hh ../g4c.h
G4CMMDEPS=../g4c_mm.cc ../g4c_mm.h ../g4c_mm.hh ../g4c.h
ACDEPS=../ac.cc ../ac.hh ../g4c_ac.h ../g4c.h
ACDEVDEPS=../ac_dev.cu ../ac.hh ../g4c_ac.h ../g4c.hh ../g4c.h
EACDEPS=evalac.cc ../g4c.h ../g4c_ac.h utils.h

ELUSRCS=$(COMMSRCS) ../lookup.cu evallu.cc
ELUOBJS=$(addsuffix -eval.o, $(basename $(notdir $(ELUSRCS))))
LUDEPS=../lookup.cu ../g4c_lookup.h ../g4c.h
ELUDEPS=evallu.cc ../g4c.h ../g4c_lookup.h utils.h

all: elu eac

elu: $(ELUOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

eac: $(EACOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

ecl: $(ECLOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

evalcl-eval.o: $(ECLDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

g4c_cl-eval.o: $(CLDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


evallu-eval.o: $(ELUDEPS)
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
	rm -rf ecl
