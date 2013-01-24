# G4C Makefile
# Naive solution to deal with test and lib build.

CXX          = g++
NVCC         = nvcc
CXXFLAGS     = -O2
CXXLIBFLAGS  = -fPIC
NVCCFLAGS    = -arch=sm_20 -O3
NVCCLIBFLAGS = --shared --compiler-options '-fPIC'

MMSRCS   = mm.cc
CORESRCS = $(MMSRCS) main.cu
ACSRCS   = ac.cc ac_dev.cu
CLSRCS   = cl.cu
LUSRCS   = lpm.cu

ALLSRCS  = $(CORESRCS) $(ACSRCS) $(CLSRCS) $(LUSRCS)

LIBOBJS  = $(addsuffix -lib.o, $(basename $(ALLSRCS)))

MMDEPS    = mm.cc mm.h mm.hh g4c.h
COREDEPS  = main.cu internal.hh g4c.h
ACDEPS    = ac.cc g4c_ac.h g4c.h
ACDEVDEPS = ac_dev.cu g4c_ac.h internal.hh g4c.h
LPMDEPS   = lpm.cu g4c_lpm.h internal.hh g4c.h
CLDEPS    = cl.cu g4c_cl.h internal.hh g4c.h

all: libg4c

libg4c: $(LIBOBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) $^ -o $@.so

lpm-lib.o: $(LPMDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

cl-lib.o: $(CLDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

mm-lib.o: $(MMDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

main-lib.o: $(COREDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

ac-lib.o: $(ACDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

ac_dev-lib.o: $(ACDEVDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@


install-lib: libg4c.so
	cp libg4c.so /usr/lib/
	cp g4c.h /usr/include/
	cp g4c_lpm.h /usr/include/
	cp g4c_ac.h /usr/include/
	cp g4c_cl.h /usr/include/

uninstall-lib:
	rm -f /usr/lib/libg4c.so
	rm -f /usr/include/g4c.h
	rm -f /usr/include/g4c_lpm.h
	rm -r /usr/include/g4c_ac.h


clean:
	rm -f *.o *.so
	rm -f ac-test
	rm -f mm-test
	rm -f cl-tset





##########################################################
# Deprecated:
##########################################################
ACTESTFLAGS = -D_G4C_AC_TEST_
MMTESTFLAGS = -D_G4C_MM_TEST_
CLTESTFLAGS = -D_G4C_CL_TEST_

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

