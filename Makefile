# makefile for vmath

CRATE = libvmath-68a2c114141ca-0.0.dylib libvmath-68a2c114141ca-0.0.dylib.dSYM
EXE = main main.dSYM
SRC = vec2.rs vec3.rs vec4.rs mat2.rs

$(EXE): libvmath-68a2c114141ca-0.0.dylib main.rs
	rustc main.rs -L.

$(CRATE): $(SRC) vmath.rc
	rustc --lib vmath.rc

clean:
	rm -rf $(CRATE) $(EXE)

.PHONY: all
