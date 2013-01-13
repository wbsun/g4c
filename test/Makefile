# G4C submodule testing makefile.

CXX=g++
NVCC=nvcc
CXXFLAGS=-O2
NVCCFLAGS=-arch=sm_20 -O2

COMMSRCS=../mm.cc ../main.cu

EACSRCS=$(COMMSRCS) ../ac.cc ../ac_dev.cu evalac.cc
EACOBJS=$(addsuffix -eval.o, $(basename $(notdir $(EACSRCS))))

ECLSRCS = $(COMMSRCS) ../cl.cu evalcl.cc
ECLOBJS = $(addsuffix -eval.o, $(basename $(notdir $(ECLSRCS))))

CLDEPS = ../cl.cu ../g4c_cl.h ../g4c.h ../internal.hh
ECLDEPS = evalcl.cc ../g4c.h ../g4c_cl.h utils.h

G4CDEPS=../main.cu ../internal.hh ../g4c.h
G4CMMDEPS=../mm.cc ../mm.h ../mm.hh ../g4c.h
ACDEPS=../ac.cc ../ac.hh ../g4c_ac.h ../g4c.h
ACDEVDEPS=../ac_dev.cu ../ac.hh ../g4c_ac.h ../internal.hh ../g4c.h
EACDEPS=evalac.cc ../g4c.h ../g4c_ac.h utils.h

ELUSRCS=$(COMMSRCS) ../lpm.cu evallu.cc
ELUOBJS=$(addsuffix -eval.o, $(basename $(notdir $(ELUSRCS))))
LUDEPS=../lpm.cu ../g4c_lpm.h ../g4c.h
ELUDEPS=evallu.cc ../g4c.h ../g4c_lpm.h utils.h

all: elu eac ecl

elu: $(ELUOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

eac: $(EACOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

ecl: $(ECLOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

evalcl-eval.o: $(ECLDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

cl-eval.o: $(CLDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


evallu-eval.o: $(ELUDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

lpm-eval.o: $(LUDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

evalac-eval.o: $(EACDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

mm-eval.o: $(G4CMMDEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

main-eval.o: $(G4CDEPS)
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
