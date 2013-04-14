# makefile for vmath

EXE = main main.dSYM
SRC = vmath.rs main.rs

$(EXE): $(SRC)
	rustc main.rs

clean:
	rm -rf $(EXE)

.PHONY: all
