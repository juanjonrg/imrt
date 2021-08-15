CC=nvcc
FLAGS=-g -std=c++14 -arch=sm_70
LIBS=-lcusparse

gradient: gradient.cu
	$(CC) $(FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f gradient
