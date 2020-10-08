CFLAGS = -Wno-deprecated-gpu-targets
CC = nvcc

all: 01 01b 02 03 04 05 06 factorise_3_4_block factorise_3_cuda

01: 01.cu
	$(CC) $(CFLAGS) -o 01 01.cu -lrt

01b: 01b.cu
	$(CC) $(CFLAGS) -o 01b 01b.cu -lrt

02: 02.cu
	$(CC) $(CFLAGS) -o 02 02.cu -lrt

03: 03.cu
	$(CC) $(CFLAGS) -o 03 03.cu -lrt

04: 04.cu
	$(CC) $(CFLAGS) -o 04 04.cu -lrt

05: 05.cu
	$(CC) $(CFLAGS) -o 05 05.cu -lrt

06: 06.cu
	$(CC) $(CFLAGS) -o 06 06.cu -lrt

factorise_3_4_block: factorise_3_4_block.c
	cc -o factorise_3_4_block factorise_3_4_block.c -lrt -pthread

factorise_3_cuda: factorise_3_cuda.cu
	$(CC) $(CFLAGS) -o factorise_3_cuda factorise_3_cuda.cu -lrt

clean:
	rm -f 01 01b 02 03 04 05 06 factorise_3_4_block factorise_3_cuda *~
