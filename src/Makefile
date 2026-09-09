# XLSDFT build with LVTX backend and SVE/SME CheFSI optimized kernels.
include Makefile.config
include objects.mk

# Explicit SME/SVE feature macros for header-only LVTX + locally-streaming TU's.
LVTX_CXXFLAGS := -D__ARM_FEATURE_SME=1 \
                 -D__ARM_FEATURE_SME_F64F64=1 \
                 -D__ARM_FEATURE_SVE=1
# Vendored lvtx_blas_common.hpp (SYGVD phase struct) shadows the cluster tree.
LVTX_INCLUDES := -Iinclude/lvtx_dsygvd_local -I$(LVTX_BLAS_ROOT)

.PHONY: all clean
all: a.out

ifeq ($(USE_LVTX_BACKEND), 1)
.PHONY: pin-check
pin-check: check_lvtx_blas_pin.py
	@LVTX_BLAS_ROOT="$(LVTX_BLAS_ROOT)" python3 ./check_lvtx_blas_pin.py

backend_free.o: backend_free.cpp pin-check
	$(CC) -c $(CFLAGS) $(LVTX_CXXFLAGS) $(CPPFLAGS) $(LVTX_INCLUDES) $< -o $@

lvtx_backend.o: lvtx_backend.cpp pin-check
	$(CC) -c $(CFLAGS) $(LVTX_CXXFLAGS) $(CPPFLAGS) $(LVTX_INCLUDES) $< -o $@
endif

%.o : %.cpp
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

a.out: $(OBJSC)
	$(CC) -o $(BINBASE) $^ $(CFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(OBJSC) main.o $(BINBASE) backend_free.o lvtx_backend.o
