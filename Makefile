# G4C Makefile
# Naive solution to deal with test and lib build.

CXX          = g++
NVCC         = nvcc
CXXFLAGS     = -O2
CXXLIBFLAGS  = -fPIC
NVCCFLAGS    = -arch=sm_20 -O2
NVCCLIBFLAGS = --shared --compiler-options '-fPIC'

MMSRCS   = g4c_mm.cc
CORESRCS = $(MMSRCS) g4c.cu
ACSRCS   = ac.cc ac_dev.cu
CLSRCS   = g4c_cl.cu
LUSRCS   = lookup.cu

ALLSRCS  = $(CORESRCS) $(ACSRCS) $(CLSRCS) $(LUSRCS)

LIBOBJS  = $(addsuffix -lib.o, $(basename $(ALLSRCS)))

MMDEPS    = g4c_mm.cc g4c_mm.h g4c_mm.hh g4c.h
COREDEPS  = g4c.cu g4c.hh g4c.h
ACDEPS    = ac.cc ac.hh g4c_ac.h g4c.h
ACDEVDEPS = ac_dev.cu ac.hh g4c_ac.h g4c.hh g4c.h
LPMDEPS   = lookup.cu g4c_lookup.h g4c.hh g4c.h
CLDEPS    = g4c_cl.cu g4c_cl.h g4c.hh g4c.h

all: libg4c

libg4c: $(LIBOBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) $^ -o $@.so

lookup-lib.o: $(LPMDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

g4c_cl-lib.o: $(CLDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

g4c_mm-lib.o: $(MMDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

g4c-lib.o: $(COREDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@

ac-lib.o: $(ACDEPS)
	$(CXX) $(CXXFLAGS) $(CXXLIBFLAGS) -c $< -o $@

ac_dev-lib.o: $(ACDEVDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBFLAGS) -c $< -o $@


install-lib: libg4c.so
	cp libg4c.so /usr/lib/
	cp g4c.h /usr/include/
	cp g4c_lookup.h /usr/include/
	cp g4c_ac.h /usr/include/
	cp g4c_cl.h /usr/include/

uninstall-lib:
	rm -f /usr/lib/libg4c.so
	rm -f /usr/include/g4c.h
	rm -f /usr/include/g4c_lookup.h
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

